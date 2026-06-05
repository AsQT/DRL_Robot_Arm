"""
Environment factory for training.

Provides ``make_env_fn()`` and ``create_training_env()`` which instantiate the
Gymnasium environment from a Config object, optionally create a vectorized
environment with DummyVecEnv or SubprocVecEnv, and wrap it with Monitor.

VecNormalize is intentionally disabled — the raw-observation pipeline does not
normalize observations or rewards.

Supports curriculum training via ``CurriculumTargetSampler``:
- When ``curriculum_config_path`` is provided, each worker gets its own sampler
  cloned from the master sampler.
- The master sampler's stage is updated by ``CurriculumGlobalStepCallback``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    import gymnasium as gym
    import stable_baselines3.common.vec_env as VecEnv
    from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
    from drl_pathplanning.training.curriculum import CurriculumTargetSampler


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class EnvOutput:
    """
    Return type of ``create_training_env()``.

    Attributes
    ----------
    env
        The (possibly wrapped) environment passed to the SB3 model.
    raw_env
        The bare ``CartesianPathPlanningEnv`` without any SB3 wrappers.
        Use this when you need to access env-specific attributes.
        For multi-env runs this is a list of the raw envs.
    training_env
        The ``DummyVecEnv`` or ``SubprocVecEnv`` that the SB3 model actually
        trains on.  This is the same as ``env`` (VecNormalize is disabled).
        Pass this to the ``TrainingViewerCallback`` so it can read fresh state
        from the env that SB3 is actually using.
    master_curriculum_sampler
        The master ``CurriculumTargetSampler`` (one per run), or ``None`` if
        curriculum is not active.  Pass this to ``CurriculumGlobalStepCallback``.
    worker_samplers
        List of per-worker curriculum samplers (one per env).  Used internally
        to sync stage state back into the workers.
    """
    env: "VecEnv.VecEnv | gym.Env"
    raw_env: "CartesianPathPlanningEnv | list[CartesianPathPlanningEnv]"
    training_env: "VecEnv.VecEnv | gym.Env"
    master_curriculum_sampler: "CurriculumTargetSampler | None"
    worker_samplers: list


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def make_env_fn(
    env_cfg: "drl_pathplanning.gymnasium.config.Config",
    seed: int,
    rank: int,
    curriculum_sampler: "CurriculumTargetSampler | None" = None,
) -> Callable[[], "gym.Env"]:
    """
    Build a single environment factory function for use with DummyVecEnv or
    SubprocVecEnv.

    Each call produces a fresh ``CartesianPathPlanningEnv``, wrapped in a
    ``Monitor`` with no output filename to prevent file conflicts across
    parallel workers.

    Environment behavior is driven by config flags (obstacle.enabled, collision.enabled, etc.)
    rather than by a mode string parameter.

    Parameters
    ----------
    env_cfg
        Populated ``Config`` object.
    seed
        Base random seed.  The per-env seed is ``seed + rank`` so that
        environments do not share RNG state.
    rank
        Unique integer index for this environment (0, 1, 2, ...).
    curriculum_sampler
        Per-worker curriculum sampler clone.  If ``None``, curriculum is disabled.

    Returns
    -------
    Callable
        A zero-argument callable that returns a ready-to-use Gymnasium env.
    """

    def _init() -> "gym.Env":
        import stable_baselines3.common.monitor as _monitor
        from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv

        env = CartesianPathPlanningEnv(
            env_cfg=env_cfg,
            start_mode=env_cfg.start.resolved_mode,
            curriculum_sampler=curriculum_sampler,
        )
        env.reset(seed=seed + rank)
        gym_env: gym.Env = _monitor.Monitor(env)
        return gym_env

    return _init


def create_training_env(
    env_cfg: "drl_pathplanning.gymnasium.config.Config",
    seed: int,
    n_envs: int = 1,
    vec_env_type: str = "auto",
    log_dir: "Path | None" = None,
    curriculum_config_path: "Path | None" = None,
    curriculum_enabled: bool | None = None,
) -> EnvOutput:
    """
    Create and wrap the training environment.

    1. Instantiates ``n_envs`` copies of ``CartesianPathPlanningEnv`` from the
       provided Config, each with a unique seed.
    2. Wraps them with ``DummyVecEnv`` (n_envs == 1) or ``SubprocVecEnv``
       (n_envs > 1).
    3. VecNormalize is intentionally disabled (raw-observation pipeline).
    4. Seeds Python, NumPy, PyTorch.
    5. When ``curriculum_config_path`` is given, each worker receives its own
       curriculum sampler cloned from the master sampler.

    Parameters
    ----------
    env_cfg
        Populated ``Config`` object.
    seed
        Base random seed.  Environment ``i`` is seeded with ``seed + i``.
    n_envs
        Number of parallel environments (default: 1).
    vec_env_type
        Vectorized environment type: ``"dummy"``, ``"subproc"``, or ``"auto"``
        (auto selects ``dummy`` for n_envs==1 and ``subproc`` for n_envs>1).
    log_dir
        Directory for Monitor output.  If ``None`` (default), a temporary
        directory is created.  Pass the run's ``log_dir`` to write monitor.csv
        to the correct output location.
    curriculum_config_path
        [Deprecated — kept for backward compat.] Path to a curriculum YAML file.
        If ``None``, curriculum is enabled/disabled based on ``curriculum_enabled``.
    curriculum_enabled
        If ``True``, curriculum is enabled using the curriculum section of
        ``env_cfg``. If ``None``, falls back to checking ``curriculum_config_path``.

    Returns
    -------
    EnvOutput
        Contains the (possibly wrapped) env, the raw env(s), and the master
        curriculum sampler.

    Example
    -------
    ::

        from drl_pathplanning.gymnasium.config import Config
        from drl_pathplanning.training import create_training_env

        cfg = Config.from_yaml(Path("config/environment.yaml"))
        result = create_training_env(cfg, seed=42, n_envs=4)
        model = DDPG(result.env, ...)
        # cleanup
        result.env.close()
    """
    import stable_baselines3.common.vec_env as _vec_env

    # ── Resolve vec_env_type ──────────────────────────────────────────────
    if vec_env_type == "auto":
        vec_env_type = "dummy" if n_envs == 1 else "subproc"
    elif vec_env_type not in ("dummy", "subproc"):
        raise ValueError(
            f"vec_env_type must be 'auto', 'dummy', or 'subproc', got '{vec_env_type}'"
        )

    # ── Seed all random sources ────────────────────────────────────────────
    _seed_all(seed)

    # ── Curriculum sampler setup ───────────────────────────────────────────
    # Support both the legacy YAML path approach and the new unified Config approach
    _use_curriculum = curriculum_enabled if curriculum_enabled is not None else (curriculum_config_path is not None)

    if _use_curriculum and not env_cfg.curriculum.stages:
        print("[WARN] curriculum_enabled=True but cfg.curriculum.stages is empty. Disabling curriculum.")
        _use_curriculum = False

    master_sampler, worker_samplers = _build_curriculum_samplers(
        curriculum_enabled=_use_curriculum,
        curriculum_cfg=env_cfg.curriculum if _use_curriculum else None,
        workspace_min=env_cfg.workspace.min_np,
        workspace_max=env_cfg.workspace.max_np,
        n_envs=n_envs,
        seed=seed,
    )

    if master_sampler is not None:
        print(f"[CURRICULUM] Enabled — {len(master_sampler.config.stages)} stages, "
              f"{master_sampler.total_timesteps:,} total timesteps")

    # ── Create factory functions ───────────────────────────────────────────
    env_fns = [
        make_env_fn(
            env_cfg,
            seed,
            rank,
            curriculum_sampler=worker_samplers[rank] if worker_samplers else None,
        )
        for rank in range(n_envs)
    ]

    # ── Vectorized environment ─────────────────────────────────────────────
    if vec_env_type == "dummy" or n_envs == 1:
        vec_env = _vec_env.DummyVecEnv(env_fns)
    else:
        import sys as _sys

        start_method = "fork" if _sys.platform != "win32" else "spawn"
        vec_env = _vec_env.SubprocVecEnv(env_fns, start_method=start_method)

    # ── VecNormalize is intentionally disabled (raw-observation pipeline) ──
    # No normalization is applied to observations or rewards.

    # ── Collect raw env references for cleanup ────────────────────────────
    from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv

    if n_envs == 1:
        raw_env = CartesianPathPlanningEnv(
            env_cfg=env_cfg,
            start_mode=env_cfg.start.resolved_mode,
            curriculum_sampler=worker_samplers[0] if worker_samplers else None,
        )
    else:
        # For SubprocVecEnv, the raw envs live in the subprocess workers.
        # We return the VecEnv itself so the caller can call env.close().
        raw_env = [_vec_env.DummyVecEnv([fn]).envs[0] for fn in env_fns]

    return EnvOutput(
        env=vec_env,
        raw_env=raw_env,
        training_env=vec_env,
        master_curriculum_sampler=master_sampler,
        worker_samplers=worker_samplers,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_curriculum_samplers(
    curriculum_enabled: bool,
    curriculum_cfg,
    workspace_min,
    workspace_max,
    n_envs: int,
    seed: int,
    curriculum_config_path: "Path | None" = None,
) -> tuple:
    """
    Build the master and per-worker curriculum samplers.

    Supports both:
      - Unified Config approach: curriculum_enabled=True, curriculum_cfg = cfg.curriculum
      - Legacy YAML approach: curriculum_config_path = Path(...)

    Returns
    -------
    tuple
        (master_sampler_or_None, list_of_worker_samplers_or_None)
    """
    from drl_pathplanning.training.curriculum import CurriculumTargetSampler

    if not curriculum_enabled:
        return None, []

    if curriculum_cfg is not None:
        # New unified config approach
        master = CurriculumTargetSampler.from_gymnasium_config(
            gymnasium_curriculum_cfg=curriculum_cfg,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            seed=seed,
        )
    elif curriculum_config_path is not None:
        # Legacy YAML approach (backward compat)
        master = CurriculumTargetSampler.from_yaml(
            curriculum_yaml_path=curriculum_config_path,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            seed=seed,
        )
    else:
        return None, []

    workers = [
        master.clone(seed=seed + rank, worker_rank=rank) for rank in range(n_envs)
    ]
    return master, workers


def _seed_all(seed: int) -> None:
    """Seed all random sources used by the training pipeline."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
