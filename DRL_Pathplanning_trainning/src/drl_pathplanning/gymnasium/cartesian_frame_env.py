"""
Cartesian Frame Path Planning Gymnasium Environment.

A frame-only DRL environment where the agent is a virtual Cartesian point navigating
in a 3-D workspace. The agent learns a policy::

    observation (15-D)  ->  delta_xyz action  ->  waypoint list

Architecture
------------
The Gymnasium env is the source of truth for all numerical state.
PyBullet visualisation (if used) is driven by EXTERNAL code using the info dict
returned by reset() and step().  The env itself contains NO PyBullet logic.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional

from drl_pathplanning import geometry as _geo
from . import config as _cfg
from . import spaces as _spaces
from .config import load_config
from .reward import RewardCalculator, REWARD_COMPONENT_KEYS
from drl_pathplanning.geometry.collision_geometry import GeometryCollisionChecker
from drl_pathplanning.training.target_provider import TargetProvider
from drl_pathplanning.training.curriculum import CurriculumTargetSampler


class CartesianPathPlanningEnv(gym.Env):
    """
    Gymnasium environment for frame-only Cartesian path planning.

    The agent controls a virtual point by issuing normalised 3-D delta actions
    (dx, dy, dz).  The environment updates the point directly::

        next_pos = current_pos + action * action_step

    NO robot model, NO URDF, NO FK, NO IK, NO MoveIt, NO PyBullet.
    """

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        env_cfg: Optional["drl_pathplanning.gymnasium.config.Config"] = None,
        start_mode: str = "config",
        start_pos: Optional[Tuple[float, float, float]] = None,
        curriculum_sampler: Optional["CurriculumTargetSampler"] = None,
    ) -> None:
        super().__init__()

        if env_cfg is not None:
            self._cfg = env_cfg
        else:
            self._cfg = load_config()

        # Resolve effective start mode:
        #   "config"  -> read from cfg.start.resolved_mode
        #   "fixed"   -> always use fixed start (optionally from start_pos override)
        #   "random"  -> always sample randomly
        if start_mode == "config":
            self._start_mode = self._cfg.start.resolved_mode
        elif start_mode in ("fixed", "random"):
            self._start_mode = start_mode
        else:
            raise ValueError(
                f"start_mode must be 'config', 'fixed', or 'random', got '{start_mode}'"
            )

        self._fixed_start = (
            np.array(start_pos, dtype=np.float32) if start_pos is not None else None
        )

        self._curriculum_sampler = curriculum_sampler
        self._global_step = 0
        self._on_reset_callbacks: list = []

        self.action_space = _spaces.make_action_space()
        self.observation_space = _spaces.make_observation_space()

        self._ws_min = self._cfg.workspace.min_np
        self._ws_max = self._cfg.workspace.max_np
        self._ws_range = self._cfg.workspace.range_np

        self._tr_min = self._cfg.target_region.min_np
        self._tr_max = self._cfg.target_region.max_np

        # Table is always collidable (fixed environment object).
        self._table_center = self._cfg.table.center_np
        self._table_half_extent = self._cfg.table.half_extent_np

        # Obstacle — enabled when cfg.obstacle.enabled is True.
        # Used for both collision checking and observation.
        if self._cfg.obstacle.enabled:
            self._obstacle_center = self._cfg.obstacle.center_np
            self._obstacle_half_extent = self._cfg.obstacle.half_extent_np
            self._collision_half_extent = self._cfg.obstacle.collision_half_extent_np
        else:
            self._obstacle_center = None
            self._obstacle_half_extent = None
            self._collision_half_extent = None

        # Per-episode obstacle geometry (set in reset() when size_random is enabled)
        self._current_obstacle_size: np.ndarray | None = None
        self._current_obstacle_half_extent: np.ndarray | None = None
        self._current_collision_half_extent: np.ndarray | None = None

        # Start region bounds (for random start).
        self._start_region_min = self._cfg.start.random_bounds_min_np
        self._start_region_max = self._cfg.start.random_bounds_max_np
        self._random_start = self._cfg.start.random_start

        # Collision checking: enabled when cfg.collision.enabled is True AND obstacle is present.
        # Table is always checked. Workspace bounds are always checked.
        _collision_enabled = self._cfg.collision.enabled and self._cfg.obstacle.enabled

        self._collision_checker = GeometryCollisionChecker(
            check_box=_collision_enabled,
            obstacle_center=(
                self._obstacle_center.copy() if self._obstacle_center is not None else None
            ),
            obstacle_half_extent=(
                self._obstacle_half_extent.copy() if self._obstacle_half_extent is not None else None
            ),
            collision_half_extent=(
                self._collision_half_extent.copy() if self._collision_half_extent is not None else None
            ),
            obstacle_name=self._cfg.obstacle.name,
            ws_min=self._ws_min.copy(),
            ws_max=self._ws_max.copy(),
            table_center=self._table_center.copy(),
            table_half_extent=self._table_half_extent.copy(),
        )

        self._target_provider = TargetProvider(
            target_region_min=self._tr_min.copy(),
            target_region_max=self._tr_max.copy(),
            workspace_min=self._ws_min.copy(),
            workspace_max=self._ws_max.copy(),
            avoid_box_center=(
                self._obstacle_center.copy() if self._obstacle_center is not None else None
            ),
            avoid_box_half_extent=(
                self._collision_half_extent.copy() if self._collision_half_extent is not None else None
            ),
        )

        # Centralised reward computation — all reward logic lives in reward.py
        self._reward_calculator = RewardCalculator(
            reward_cfg=self._cfg.reward,
        )

        self._current_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._start_pos_episode: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_dist: float = 0.0
        self._step_count: int = 0

        self._prev_action: np.ndarray = np.zeros(3, dtype=np.float32)
        self._episode_path_length: float = 0.0
        self._expected_path_length: float = 0.0
        self._episode_path_points: list = []

        self._current_target_mode: str = "random"
        self._current_corner_index: Optional[int] = None

        self._next_reset_options: Optional[dict] = None

    # ------------------------------------------------------------------ #
    #   Curriculum integration
    # ------------------------------------------------------------------ #
    def add_reset_callback(self, callback: "callable") -> None:
        """Register a callable to be invoked at the end of every reset()."""
        self._on_reset_callbacks.append(callback)

    # ------------------------------------------------------------------ #
    #   Target resolution
    # ------------------------------------------------------------------ #
    def _resolve_target(
        self,
        options: Optional[dict],
    ) -> Tuple[np.ndarray, str, Optional[int]]:
        """Resolve the target position from reset options or curriculum sampler."""
        if self._curriculum_sampler is not None:
            target_pos = self._curriculum_sampler.sample(self.np_random)
            return target_pos, "curriculum", None

        target_mode = "random"
        if options is not None:
            target_mode = options.get("target_mode", "random")

        fixed_target: Optional[np.ndarray] = None
        if target_mode == "fixed":
            raw_target = options.get("target") if options else None
            if raw_target is not None:
                fixed_target = np.asarray(raw_target, dtype=np.float32)
            else:
                raise ValueError(
                    "target_mode='fixed' requires 'target' to be provided in options"
                )

        corner_index: Optional[int] = None
        if target_mode == "static":
            corner_index = options.get("static_corner_index", 0) if options else 0

        target_pos = self._target_provider.get_target(
            mode=target_mode,
            rng=self.np_random,
            fixed_target=fixed_target,
            corner_index=corner_index,
        )

        return target_pos, target_mode, corner_index

    # ------------------------------------------------------------------ #
    #   Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if options is None and self._next_reset_options is not None:
            options = self._next_reset_options
            self._next_reset_options = None

        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._episode_path_length = 0.0

        self._current_pos = self._sample_start()
        self._start_pos_episode = self._current_pos.copy()
        self._prev_pos = self._current_pos.copy()
        self._episode_path_points = [self._current_pos.copy()]

        self._target_pos, self._current_target_mode, self._current_corner_index = \
            self._resolve_target(options)

        self._prev_dist = float(np.linalg.norm(self._target_pos - self._current_pos))
        self._expected_path_length = self._prev_dist

        # ------------------------------------------------------------------ #
        # Build current obstacle geometry per episode
        # ------------------------------------------------------------------ #
        # Log: record center BEFORE z override so we can verify sampling order
        _center_before_z_override = None

        self._current_obstacle_size = self._sample_obstacle_size()
        if self._current_obstacle_size is not None:
            self._current_obstacle_half_extent = self._current_obstacle_size / 2.0
            self._current_collision_half_extent = (
                self._current_obstacle_half_extent
                + self._cfg.obstacle.safety_margin
            )
        else:
            self._current_obstacle_half_extent = None
            self._current_collision_half_extent = None

        # Sample obstacle center x,y per episode (fixed or random depending on resolved_mode)
        self._obstacle_center = self._sample_obstacle_center()

        # Record center z before override (shows what random_bounds contributed)
        if self._obstacle_center is not None:
            _center_before_z_override = self._obstacle_center.copy()

        # Override center_z based on place_on_table setting
        if self._obstacle_center is not None and self._current_obstacle_size is not None:
            _table_top_z = float(
                self._cfg.table.center_np[2] + self._cfg.table.size_np[2] / 2.0
            )
            _computed_center_z = float(_table_top_z + float(self._current_obstacle_size[2]) / 2.0)
            self._obstacle_center[2] = _computed_center_z
            _final_center_z = self._obstacle_center[2]
            _size_z = float(self._current_obstacle_size[2])
            _final_bottom_z = _final_center_z - _size_z / 2.0

        # Check and resample to avoid target inside inflated obstacle AABB
        sr = self._cfg.obstacle.size_random
        _resample_attempt = 0
        _resampled_z_overridden = False
        if (
            self._cfg.obstacle.enabled
            and sr.enabled
            and sr.avoid_target_overlap
            and self._obstacle_center is not None
            and self._current_obstacle_size is not None
        ):
            margin = sr.overlap_margin
            max_attempts = sr.max_resample_attempts
            for _attempt in range(max_attempts):
                if not self._is_target_inside_obstacle_inflated_aabb(
                    self._obstacle_center,
                    self._current_obstacle_size,
                    self._target_pos,
                    margin,
                ):
                    break
                _resample_attempt += 1
                # Resample obstacle center x,y
                self._obstacle_center = self._sample_obstacle_center()
                # Override z again after resample
                self._obstacle_center[2] = _computed_center_z
                _resampled_z_overridden = True
            else:
                print(
                    "[ENV] Warning: target within obstacle inflated AABB "
                    "after " + str(max_attempts) + " resample attempts"
                )

        # NOTE: debug logging removed — training must not be spammed per episode.
        # If needed, re-add under a config flag such as cfg.environment.debug_obstacle = true.

        # Update collision checker with current obstacle center + collision half extent
        if self._collision_checker is not None:
            self._collision_checker.update_obstacle(
                self._obstacle_center,
                self._current_collision_half_extent,
            )

        # Reset per-episode state in the reward calculator (simple reward needs no geometry)
        self._reward_calculator.reset()

        obs = self._build_observation()

        # Compute box_top_z and required_clearance_z from current obstacle geometry
        _box_top_z = 0.0
        _req_clear_z = 0.0
        if self._cfg.obstacle.enabled and self._obstacle_center is not None:
            _box_top_z = float(
                self._obstacle_center[2]
                + (self._current_obstacle_size[2] / 2.0 if self._current_obstacle_size is not None else 0.0)
            )
            _req_clear_z = float(
                _box_top_z + 0.05  # obstacle_clearance removed from RewardConfig
            )

        info: Dict[str, Any] = {
            "is_success": False,
            "distance": self._prev_dist,
            "prev_distance": self._prev_dist,
            "is_collision": False,
            "collision_object": "none",
            "collision_margin": (
                float(self._cfg.obstacle.safety_margin)
                if self._cfg.collision.enabled and self._cfg.obstacle.enabled
                else 0.0
            ),
            "obstacle_collision": False,
            "table_collision": False,
            "box_collision": False,
            "out_of_workspace": False,
            "termination_reason": "none",
            "step_count": 0,
            "current_pos": self._current_pos.copy(),
            "prev_pos": self._prev_pos.copy(),
            "target_pos": self._target_pos.copy(),
            "start_pos": self._start_pos_episode.copy(),
            "path_length": 0.0,
            "path_length_so_far": 0.0,
            "expected_path_length": self._expected_path_length,
            "actual_path_length": 0.0,
            "path_efficiency": float("nan"),
            "path_efficiency_percent": float("nan"),
            "path_efficiency_valid": False,
            "box_top_z": _box_top_z,
            "required_clearance_z": _req_clear_z,
            "phase": "approach_target",
            "inside_obstacle_corridor": False,
            "obstacle_cleared": False,
            "waypoint_reached": False,
            "distance_to_target": self._prev_dist,
            "distance_to_waypoint": 0.0,
            "agent_s": 0.0,
            "forward_delta": 0.0,
            "target_mode": self._current_target_mode,
            "static_corner_index": self._current_corner_index,
            "_curriculum": (
                self._curriculum_sampler.diagnostics()
                if self._curriculum_sampler else None
            ),
            # Obstacle info per episode
            "obstacle_enabled": self._cfg.obstacle.enabled,
            "obstacle_mode": self._cfg.obstacle.resolved_mode,
            "obstacle_center": (
                self._obstacle_center.copy() if self._obstacle_center is not None else None
            ),
            "obstacle_size": (
                self._current_obstacle_size.copy()
                if self._current_obstacle_size is not None else None
            ),
        }

        for cb in self._on_reset_callbacks:
            cb(self)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta = action * self._cfg.environment.action_step
        next_pos = self._current_pos + delta

        prev_dist = self._prev_dist
        new_dist = float(np.linalg.norm(self._target_pos - next_pos))
        path_segment_length = float(np.linalg.norm(delta))

        collision_result = self._collision_checker.check(
            self._current_pos, next_pos
        )
        box_or_table_collision = (
            collision_result.collides
            and collision_result.collision_type.startswith("box")
        )
        table_collision = (
            collision_result.collides
            and collision_result.collision_type.startswith("table")
        )
        out_of_workspace = collision_result.collision_type == "workspace"

        success = new_dist < self._cfg.termination.goal_threshold
        truncated = self._step_count >= self._cfg.environment.max_episode_steps

        self._prev_pos = self._current_pos.copy()
        self._current_pos = next_pos.astype(np.float32)
        self._prev_dist = new_dist
        self._step_count += 1

        self._episode_path_length += path_segment_length
        self._episode_path_points.append(self._current_pos.copy())

        # Collision with obstacle: check only when collision.enabled is True AND obstacle exists.
        box_or_table_collision = (
            collision_result.collides
            and collision_result.collision_type.startswith("box")
        )
        table_collision = (
            collision_result.collides
            and collision_result.collision_type.startswith("table")
        )
        # Workspace termination: controlled by termination.workspace_terminate config.
        out_of_workspace = (
            collision_result.collision_type == "workspace"
            and self._cfg.termination.workspace_terminate
        )

        # Terminate on collision only when termination.collision_terminate is True.
        collision_terminates = (
            box_or_table_collision
            and self._cfg.termination.collision_terminate
        )
        terminated = bool(success or collision_terminates or out_of_workspace)

        # ---- Reward computation (delegated to RewardCalculator) ----
        reward, reward_info = self._reward_calculator.compute(
            prev_pos=self._prev_pos,
            current_pos=self._current_pos,
            target_pos=self._target_pos,
            action=action,
            is_success=success,
            is_collision=box_or_table_collision,
            is_out_of_workspace=out_of_workspace,
            is_timeout=truncated,
            step_count=self._step_count,
            max_steps=self._cfg.environment.max_episode_steps,
        )

        # ---- Termination reason ----
        if success:
            termination_reason = "success"
        elif box_or_table_collision:
            termination_reason = "collision"
        elif out_of_workspace:
            termination_reason = "workspace_limit"
        elif truncated:
            termination_reason = "max_steps"
        else:
            termination_reason = "none"

        # ---- Path efficiency (only computed at episode end) ----
        if terminated:
            if success and self._episode_path_length > 1e-8:
                eff = self._expected_path_length / self._episode_path_length
                eff = float(np.clip(eff, 0.0, 1.0))
                eff_pct = eff * 100.0
                eff_valid = True
            else:
                eff = float("nan")
                eff_pct = float("nan")
                eff_valid = False
        else:
            eff = float("nan")
            eff_pct = float("nan")
            eff_valid = False

        obs = self._build_observation()

        _reward_components = {k: reward_info[k] for k in REWARD_COMPONENT_KEYS if k in reward_info}

        # Flatten reward components into info for easy logging
        reward_flat = {f"reward/{k}": v for k, v in _reward_components.items()}

        info: Dict[str, Any] = {
            "is_success": bool(success),
            "distance": new_dist,
            "prev_distance": prev_dist,
            "is_collision": bool(box_or_table_collision),
            "collision_object": (
                collision_result.obstacle_name
                if (collision_result.collides and collision_result.obstacle_name)
                else "none"
            ),
            "collision_margin": float(
                self._cfg.obstacle.safety_margin
                if (self._cfg.collision.enabled and self._cfg.obstacle.enabled)
                else 0.0
            ),
            "obstacle_collision": bool(box_or_table_collision),
            "table_collision": bool(table_collision),
            "box_collision": bool(box_or_table_collision),
            "out_of_workspace": bool(out_of_workspace),
            "termination_reason": termination_reason,
            "step_count": self._step_count,
            "current_pos": self._current_pos.copy(),
            "prev_pos": self._prev_pos.copy(),
            "target_pos": self._target_pos.copy(),
            "action": np.asarray(action, dtype=np.float32),
            "delta": np.asarray(delta, dtype=np.float32),
            "start_pos": self._start_pos_episode.copy(),
            "expected_path_length": self._expected_path_length,
            "actual_path_length": self._episode_path_length,
            "path_efficiency": eff,
            "path_efficiency_percent": eff_pct,
            "path_efficiency_valid": eff_valid,
            "success_path_efficiency_percent": eff_pct if eff_valid else float("nan"),
            "target_mode": self._current_target_mode,
            "static_corner_index": self._current_corner_index,
            "path_length_so_far": self._episode_path_length,
            "_curriculum": (
                self._curriculum_sampler.diagnostics()
                if self._curriculum_sampler else None
            ),
            "distance_to_target": float(np.linalg.norm(self._current_pos - self._target_pos)),
            "reward_components": _reward_components,
            **reward_flat,
            # Obstacle info per episode
            "obstacle_enabled": self._cfg.obstacle.enabled,
            "obstacle_mode": self._cfg.obstacle.resolved_mode,
            "obstacle_center": (
                self._obstacle_center.copy() if self._obstacle_center is not None else None
            ),
            "obstacle_size": (
                self._current_obstacle_size.copy()
                if self._current_obstacle_size is not None else None
            ),
        }

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Clean up resources (no-op — no external resources)."""
        pass

    def render(self) -> None:
        """No built-in renderer — use an external viewer on the info dict."""
        pass

    # ------------------------------------------------------------------ #
    #   Internal helpers
    # ------------------------------------------------------------------ #
    def _build_observation(self) -> np.ndarray:
        """Assemble the 15-D observation vector.

        When obstacle.enabled is False, the obstacle part is [0,0,0,0,0,0].
        When obstacle.enabled is True, the obstacle part correctly reflects
        [box_center_x, box_center_y, box_center_z, box_size_x, box_size_y, box_size_z].
        """
        err = self._target_pos - self._current_pos

        if self._cfg.obstacle.enabled and self._obstacle_center is not None:
            rel_obs, obs_size = _geo.normalize_obstacle_info(
                self._obstacle_center,
                self._current_pos,
                self._ws_range,
                self._current_obstacle_half_extent,
            )
        else:
            rel_obs = np.zeros(3, dtype=np.float32)
            obs_size = np.zeros(3, dtype=np.float32)

        return np.concatenate(
            [
                self._current_pos,
                self._target_pos,
                err,
                rel_obs,
                obs_size,
            ],
            dtype=np.float32,
        )

    def _sample_obstacle_center(self) -> Optional[np.ndarray]:
        """Sample obstacle center based on cfg.obstacle.resolved_mode.

        z is intentionally NOT sampled here — reset() always overrides it to
        ``table_top_z + size_z/2`` so the box sits on the table surface.

        - fixed  : return cfg.obstacle.center_np.copy()
        - random : sample x,y from random_bounds; z = cfg center (overridden later)
        - disabled: return None
        """
        if not self._cfg.obstacle.enabled:
            return None

        if self._cfg.obstacle.resolved_mode == "fixed":
            return self._cfg.obstacle.center_np.copy()

        # random mode: sample x,y only; z is a placeholder — reset() overrides it.
        rb_min = self._cfg.obstacle.random_bounds.min_np
        rb_max = self._cfg.obstacle.random_bounds.max_np
        cx = self.np_random.uniform(float(rb_min[0]), float(rb_max[0]))
        cy = self.np_random.uniform(float(rb_min[1]), float(rb_max[1]))
        cz_placeholder = float(self._cfg.obstacle.center_np[2])  # will be overridden
        return np.array([cx, cy, cz_placeholder], dtype=np.float32)

    def _sample_obstacle_size(self) -> Optional[np.ndarray]:
        """Sample obstacle size based on cfg.obstacle.size_random.

        - disabled : return cfg.obstacle.size_np.copy()
        - enabled  : sample uniformly in each dimension independently
        - disabled : return None
        """
        if not self._cfg.obstacle.enabled:
            return None

        sr = self._cfg.obstacle.size_random
        if not sr.enabled:
            return self._cfg.obstacle.size_np.copy()

        size_x = self.np_random.uniform(sr.length_min, sr.length_max)
        size_y = self.np_random.uniform(sr.width_min, sr.width_max)
        size_z = self.np_random.uniform(sr.height_min, sr.height_max)
        return np.array([size_x, size_y, size_z], dtype=np.float32)

    def _compute_obstacle_center_z(self, size_z: float) -> float:
        """Compute center_z so the box sits on the table surface.

        box_bottom_z = center_z - size_z/2 = table_top_z
        center_z = table_top_z + size_z/2
        """
        table_top_z = float(
            self._cfg.table.center_np[2] + self._cfg.table.size_np[2] / 2.0
        )
        return float(table_top_z + size_z / 2.0)

    @staticmethod
    def _is_target_inside_obstacle_inflated_aabb(
        obstacle_center: np.ndarray,
        obstacle_size: np.ndarray,
        target_pos: np.ndarray,
        margin: float,
    ) -> bool:
        """Check if target_pos is inside the inflated obstacle AABB.

        inflated AABB = obstacle_center +/- (obstacle_size/2 + margin)
        """
        half_extent = obstacle_size / 2.0
        inflate = half_extent + margin
        obs_min = obstacle_center - inflate
        obs_max = obstacle_center + inflate
        return (
            obs_min[0] <= target_pos[0] <= obs_max[0]
            and obs_min[1] <= target_pos[1] <= obs_max[1]
            and obs_min[2] <= target_pos[2] <= obs_max[2]
        )

    def _sample_start(self) -> np.ndarray:
        """Sample a start position for the episode.

        Uses _start_mode to decide:
        - "fixed": use _fixed_start if overridden, else cfg.start.fixed_position.
        - "random": sample from start.random_bounds (reject on collision).

        In random mode, rejects candidates that collide with the table or enabled obstacle.
        """
        if self._start_mode == "fixed":
            if self._fixed_start is not None:
                return self._fixed_start.copy()
            return np.array(self._cfg.start.fixed_position, dtype=np.float32).copy()

        from drl_pathplanning import geometry as _geo

        for _ in range(100):
            candidate = _geo.sample_point_in_workspace(
                self._start_region_min, self._start_region_max, self.np_random
            )

            # Reject if collides with table (always collidable).
            if _geo.check_point_in_box(
                candidate, self._table_center, self._table_half_extent
            ):
                continue

            # Reject if collides with obstacle (only when obstacle.enabled).
            if self._collision_half_extent is not None and self._obstacle_center is not None:
                if _geo.check_point_in_box(
                    candidate, self._obstacle_center, self._collision_half_extent
                ):
                    continue

            return candidate

        # Fallback: return region centre.
        return ((self._start_region_min + self._start_region_max) / 2.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    #   Public state properties
    # ------------------------------------------------------------------ #
    @property
    def current_pos(self) -> np.ndarray:
        """Current agent position [x, y, z]."""
        return self._current_pos.copy()

    @property
    def target_pos(self) -> np.ndarray:
        """Current target position [x, y, z]."""
        return self._target_pos.copy()

    @property
    def start_pos_episode(self) -> np.ndarray:
        """Start position for the current episode [x, y, z]."""
        return self._start_pos_episode.copy()

    @property
    def prev_pos(self) -> np.ndarray:
        """Previous agent position [x, y, z]."""
        return self._prev_pos.copy()

    @property
    def episode_path_length(self) -> float:
        """Total path length travelled in the current episode."""
        return float(self._episode_path_length)

    @property
    def action_step_size(self) -> float:
        """Action step size in metres."""
        return float(self._cfg.environment.action_step)

    @property
    def target_threshold(self) -> float:
        """Target success threshold in metres. Returns termination.goal_threshold."""
        return float(self._cfg.termination.goal_threshold)

    @property
    def workspace_min(self) -> np.ndarray:
        """Workspace minimum bounds [x, y, z]."""
        return self._ws_min.copy()

    @property
    def workspace_max(self) -> np.ndarray:
        """Workspace maximum bounds [x, y, z]."""
        return self._ws_max.copy()

    @property
    def env_mode(self) -> str:
        """Deprecated: environment mode is now driven by obstacle.enabled and collision.enabled config flags."""
        return "Collision-Free" if self._cfg.collision.enabled and self._cfg.obstacle.enabled else "Default"

    # ------------------------------------------------------------------ #
    #   Target setter
    # ------------------------------------------------------------------ #
    def set_target(self, target_pos: np.ndarray) -> None:
        """Override the current target position (useful for fixed-target evaluation).

        This also re-computes the reward calculator's safe zone because the
        start->target line determines which obstacle regions are relevant.
        """
        self._target_pos = np.asarray(target_pos, dtype=np.float32)
        self._current_target_mode = "fixed"
        self._current_corner_index = None
        self._prev_dist = float(np.linalg.norm(self._target_pos - self._current_pos))
        self._prev_pos = self._current_pos.copy()

        # Re-compute reward calculator for the new target (simple reward needs no geometry).
        self._reward_calculator.reset()

    def set_next_reset_options(self, options: Optional[dict]) -> None:
        """Queue options to be consumed by the next reset() (useful for VecNormalize)."""
        self._next_reset_options = options

    # ------------------------------------------------------------------ #
    #   Box and table collision properties (for external viewer / backward compat)
    # ------------------------------------------------------------------ #
    @property
    def box_center(self) -> Optional[np.ndarray]:
        """Obstacle center, or None when obstacle is disabled."""
        if self._obstacle_center is not None:
            return self._obstacle_center.copy()
        return None

    @property
    def box_half_extent(self) -> Optional[np.ndarray]:
        """Current obstacle visual half-extent, or None when obstacle is disabled."""
        if self._current_obstacle_half_extent is not None:
            return self._current_obstacle_half_extent.copy()
        return None

    @property
    def obstacle_size(self) -> Optional[np.ndarray]:
        """Current obstacle full size (3-D), or None when obstacle is disabled."""
        return (
            self._current_obstacle_size.copy()
            if self._current_obstacle_size is not None else None
        )

    @property
    def table_center(self) -> Optional[np.ndarray]:
        """Table center (always available)."""
        return self._table_center.copy()

    @property
    def table_half_extent(self) -> Optional[np.ndarray]:
        """Table half-extent (always available)."""
        return self._table_half_extent.copy()

    @property
    def collision_half_extent(self) -> Optional[np.ndarray]:
        """Current box collision half-extent (with safety margin), or None when disabled."""
        if self._current_collision_half_extent is not None:
            return self._current_collision_half_extent.copy()
        return None

    # Backward-compatible aliases (obstacle → box)
    @property
    def obstacle_center(self) -> Optional[np.ndarray]:
        """Alias for box_center."""
        return self.box_center

    @property
    def obstacle_half_extent(self) -> Optional[np.ndarray]:
        """Alias for box_half_extent."""
        return self.box_half_extent

    # ------------------------------------------------------------------ #
    #   Backward-compatible no-op viewer passthrough methods
    #   (visualisation is handled by external code; these are no-ops)
    # ------------------------------------------------------------------ #
    def reset_agent(self, pos: np.ndarray) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def update_agent(self, current_pos: np.ndarray) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def clear_path(self) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def draw_path_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        color: Optional[list] = None,
        line_width: Optional[int] = None,
    ) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def set_path_color(self, color: list) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def draw_expected_path(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        color: Optional[list] = None,
        line_width: Optional[int] = None,
    ) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def clear_expected_path(self) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def draw_target(
        self,
        target_pos: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass

    def draw_start(self, start_pos: np.ndarray) -> None:
        """No-op. Visualisation is handled by external FrameViewer code."""
        pass
