"""
Curriculum target sampler for staged training.

Stages the target difficulty from simple fixed anchors to full-workspace random
targets while the reward function stays stable across all stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import yaml


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""
    name: str
    description: str
    mode: str
    timesteps: int
    jitter_radius: float = 0.0
    inner_margin: float = 0.15
    anchor_weight_uniform: bool = True


@dataclass
class CurriculumConfig:
    """Loaded curriculum configuration."""
    enabled: bool
    anchor_box_min: np.ndarray
    anchor_box_max: np.ndarray
    fixed_anchors: List[np.ndarray]
    stages: List[CurriculumStage]
    log_interval_episodes: int = 50
    initial_stage_index: int = 0


class CurriculumTargetSampler:
    """
    Stage-aware target sampler for curriculum training.

    Tracks the global training step and determines the current stage.
    Each call to ``sample()`` returns a target appropriate for the active stage.
    """

    def __init__(
        self,
        config: CurriculumConfig,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.config = config
        self._ws_min = np.asarray(workspace_min, dtype=np.float32)
        self._ws_max = np.asarray(workspace_max, dtype=np.float32)

        self._seed = seed
        self._next_anchor_idx = 0

        self._current_stage_idx = config.initial_stage_index
        self._stage_changed = False

        self._episode_count = 0
        self._episodes_in_stage = 0
        self._worker_rank: int = 0

    @classmethod
    def from_yaml(
        cls,
        curriculum_yaml_path: Path,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        seed: int = 42,
    ) -> "CurriculumTargetSampler":
        """Load curriculum from a YAML file."""
        raw = yaml.safe_load(curriculum_yaml_path.read_text())
        cur = raw.get("curriculum", {})

        if not cur.get("enabled", False):
            raise ValueError("curriculum.enabled must be True")

        anchor_box_min = np.array(cur["anchor_box"]["min"], dtype=np.float32)
        anchor_box_max = np.array(cur["anchor_box"]["max"], dtype=np.float32)

        anchors_raw = cur.get("fixed_anchors", [])
        anchors = [np.array(a, dtype=np.float32) for a in anchors_raw]
        if not anchors:
            raise ValueError("fixed_anchors must have at least one anchor")

        cumulative = 0
        stages: List[CurriculumStage] = []
        for s in cur.get("stages", []):
            cumulative += s.get("timesteps", 0)
            stages.append(CurriculumStage(
                name=s["name"],
                description=s.get("description", ""),
                mode=s["mode"],
                timesteps=cumulative,
                jitter_radius=s.get("jitter_radius", 0.0),
                inner_margin=s.get("inner_margin", 0.15),
                anchor_weight_uniform=s.get("anchor_weight_uniform", True),
            ))

        config = CurriculumConfig(
            enabled=True,
            anchor_box_min=anchor_box_min,
            anchor_box_max=anchor_box_max,
            fixed_anchors=anchors,
            stages=stages,
            log_interval_episodes=cur.get("log_interval_episodes", 50),
            initial_stage_index=cur.get("initial_stage_index", 0),
        )

        return cls(
            config=config,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            seed=seed,
        )

    @classmethod
    def from_gymnasium_config(
        cls,
        gymnasium_curriculum_cfg,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        seed: int = 42,
    ) -> "CurriculumTargetSampler":
        """
        Build a CurriculumTargetSampler from the unified gymnasium.config.CurriculumConfig.

        Parameters
        ----------
        gymnasium_curriculum_cfg
            The ``CurriculumConfig`` dataclass from ``drl_pathplanning.gymnasium.config``.
        workspace_min, workspace_max
            Workspace bounds.
        seed
            Random seed.

        Returns
        -------
        CurriculumTargetSampler
        """
        import numpy as _np

        if not gymnasium_curriculum_cfg.enabled:
            raise ValueError("curriculum.enabled must be True")
        if not gymnasium_curriculum_cfg.stages:
            raise ValueError("curriculum.stages must not be empty")

        anchor_box_min = _np.array(gymnasium_curriculum_cfg.anchor_box_min, dtype=_np.float32)
        anchor_box_max = _np.array(gymnasium_curriculum_cfg.anchor_box_max, dtype=_np.float32)

        anchors = [
            _np.array(a, dtype=_np.float32)
            for a in gymnasium_curriculum_cfg.fixed_anchors
        ]
        if not anchors:
            raise ValueError("fixed_anchors must have at least one anchor")

        cumulative = 0
        stages: List[CurriculumStage] = []
        for s in gymnasium_curriculum_cfg.stages:
            cumulative += s.timesteps
            stages.append(CurriculumStage(
                name=s.name,
                description=s.description,
                mode=s.mode,
                timesteps=cumulative,
                jitter_radius=s.jitter_radius,
                inner_margin=s.inner_margin,
                anchor_weight_uniform=s.anchor_weight_uniform,
            ))

        config = CurriculumConfig(
            enabled=True,
            anchor_box_min=anchor_box_min,
            anchor_box_max=anchor_box_max,
            fixed_anchors=anchors,
            stages=stages,
            log_interval_episodes=gymnasium_curriculum_cfg.log_interval_episodes,
        )

        return cls(
            config=config,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            seed=seed,
        )

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self._current_stage_idx]

    @property
    def stage_changed(self) -> bool:
        return self._stage_changed

    @property
    def total_timesteps(self) -> int:
        return self.config.stages[-1].timesteps

    def update(self, global_step: int) -> CurriculumStage:
        """Advance stage tracking to the given global step."""
        self._stage_changed = False

        new_idx = self._find_stage_idx(global_step)
        if new_idx != self._current_stage_idx:
            self._current_stage_idx = new_idx
            self._stage_changed = True
            self._episodes_in_stage = 0
            self._next_anchor_idx = 0

        return self.current_stage

    def on_episode_end(self) -> None:
        """Call this at the end of each episode to update episode counters."""
        self._episode_count += 1
        self._episodes_in_stage += 1

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a target appropriate for the current stage."""
        mode = self.current_stage.mode

        if mode == "fixed_anchors":
            return self._sample_fixed_anchors(rng)
        elif mode == "anchor_neighborhood":
            return self._sample_anchor_neighborhood(rng)
        elif mode == "random_inner_workspace":
            return self._sample_random_inner_workspace(rng)
        elif mode == "random_full_workspace":
            return self._sample_random_full_workspace(rng)
        else:
            raise ValueError(f"Unknown curriculum mode: {mode}")

    def diagnostics(self) -> dict:
        """Return current stage diagnostics for logging."""
        stage = self.current_stage
        return {
            "stage_name": stage.name,
            "stage_mode": stage.mode,
            "stage_index": self._current_stage_idx,
            "stage_description": stage.description,
            "jitter_radius": stage.jitter_radius,
            "total_timesteps": self.total_timesteps,
            "episodes_in_stage": self._episodes_in_stage,
            "total_episodes": self._episode_count,
        }

    def _find_stage_idx(self, global_step: int) -> int:
        stages = self.config.stages
        idx = 0
        for i, s in enumerate(stages):
            if global_step < s.timesteps:
                break
            idx = i + 1
        return min(idx, len(stages) - 1)

    def _sample_fixed_anchors(self, rng: np.random.Generator) -> np.ndarray:
        anchors = self.config.fixed_anchors
        anchor = anchors[self._next_anchor_idx].copy()
        self._next_anchor_idx = (self._next_anchor_idx + 1) % len(anchors)
        return anchor

    def _sample_anchor_neighborhood(self, rng: np.random.Generator) -> np.ndarray:
        jitter = self.current_stage.jitter_radius
        anchors = self.config.fixed_anchors
        anchor = anchors[self._next_anchor_idx].copy()
        self._next_anchor_idx = (self._next_anchor_idx + 1) % len(anchors)

        if jitter > 0.0:
            direction = rng.normal(size=3).astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            r = float(rng.uniform(0.0, jitter))
            jitter_vec = direction * r
            target = anchor + jitter_vec
        else:
            target = anchor.copy()

        return self._clip_to_workspace(target)

    def _sample_random_inner_workspace(self, rng: np.random.Generator) -> np.ndarray:
        margin = self.current_stage.inner_margin
        ws_range = self._ws_max - self._ws_min
        inner_min = self._ws_min + ws_range * margin
        inner_max = self._ws_max - ws_range * margin
        return rng.uniform(inner_min, inner_max).astype(np.float32)

    def _sample_random_full_workspace(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self._ws_min, self._ws_max).astype(np.float32)

    def _clip_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        clipped = np.clip(pos, self._ws_min, self._ws_max)
        return clipped.astype(np.float32)

    def clone(self, seed: int | None = None, worker_rank: int = 0) -> "CurriculumTargetSampler":
        """Return a new sampler with independent RNG state for per-worker use."""
        new_sampler = CurriculumTargetSampler(
            config=self.config,
            workspace_min=self._ws_min.copy(),
            workspace_max=self._ws_max.copy(),
            seed=seed if seed is not None else self._seed,
        )
        new_sampler._current_stage_idx = self._current_stage_idx
        new_sampler._episode_count = self._episode_count
        new_sampler._episodes_in_stage = self._episodes_in_stage
        n_anchors = len(self.config.fixed_anchors)
        new_sampler._next_anchor_idx = (self._next_anchor_idx + worker_rank) % n_anchors
        new_sampler._worker_rank = worker_rank
        return new_sampler

    def sync_from(self, other: "CurriculumTargetSampler") -> None:
        """Copy stage and episode state from another sampler."""
        self._current_stage_idx = other._current_stage_idx
        self._episode_count = other._episode_count
        self._episodes_in_stage = other._episodes_in_stage
