"""
Configuration loader for the frame-only Cartesian DRL path planning project.

This module loads environment parameters from YAML files and exposes them through
typed dataclasses.  No numeric constant is hardcoded in any other module.

Scope
-----
This is a FRAME-ONLY project.  There is no robot, URDF, FK, IK, joints, or
MoveIt.  All geometry is expressed in Cartesian (x, y, z) space relative to a
workspace bounding box.

What the config controls
------------------------
- **start**:      fixed or random start position within the workspace.
- **workspace**:  axis-aligned Cartesian bounding box (search region).
- **target**:     target sampling region and fixed targets.
- **table**:      fixed black table block (always visible, always collidable).
- **box**:        optional box obstacle (hidden by default, collidable when enabled).
- **reward**:     reward weights and mode ("default" or "simple_distance").
- **environment**: action_step, max_steps, target_threshold, env mode.
- **visualization**: PyBullet scene settings (labels, paths, frames, camera).
- **training**:   algorithm defaults (hyperparams stay in config/experiments/).
- **evaluation**: episode defaults.

Usage::

    from drl_pathplanning.gymnasium.config import load_config

    # Load from a specific YAML file
    cfg = load_config("config/my_setup.yaml")

    # Load the default config/environment.yaml (auto-detects project root)
    cfg = load_config()

The ``load_config()`` function walks upward from this module to find the project
root (the directory containing ``config/``, ``Training/``, or a Python package
marker), then resolves ``config/environment.yaml`` relative to it.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import warnings

import numpy as np


# --------------------------------------------------------------------------- #
# Project root discovery
# --------------------------------------------------------------------------- #

def _find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward from ``start`` (default: this module) until a directory
    containing ``config/`` and (``Training/`` or ``src/``) is found.

    This resolves the default config path correctly regardless of whether
    the package is installed or run as a script from a subdirectory.
    """
    if start is None:
        candidate = Path(__file__).resolve().parent
    else:
        candidate = Path(start).resolve()

    if candidate.is_file():
        candidate = candidate.parent

    # Limit search to ~10 levels to avoid infinite loops
    for _ in range(10):
        has_config = (candidate / "config").is_dir()
        has_training = (candidate / "Training").is_dir()
        has_src = (candidate / "src").is_dir()
        if has_config and (has_training or has_src):
            return candidate

        parent = candidate.parent
        if parent == candidate:
            # Reached filesystem root
            break
        candidate = parent

    # Fallback: assume src/drl_pathplanning/gymnasium/config.py -> project root is 3 levels up
    fallback = Path(__file__).resolve().parent.parent.parent.parent
    return fallback


def _default_config_path() -> Path:
    """Resolve the path to ``config/environment.yaml`` relative to the project root."""
    root = _find_project_root()
    return root / "config" / "environment.yaml"


# --------------------------------------------------------------------------- #
# YAML loader
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"[CONFIG] File not found: {path}. "
            "Please pass --config to specify the config file path."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Individual dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ProjectConfig:
    """Project metadata (frame-only — no robot)."""
    name: str
    project_type: str  # e.g. "FRAME_ONLY"
    unit: str          # always "meter"


@dataclass
class RandomBoundsConfig:
    """Bounding box for random sampling, stored as [x, y, z] lists."""
    min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @property
    def min_np(self) -> np.ndarray:
        return np.array(self.min, dtype=np.float32)

    @property
    def max_np(self) -> np.ndarray:
        return np.array(self.max, dtype=np.float32)


@dataclass
class ObstacleSizeRandomConfig:
    """Configuration for random obstacle size sampling."""
    enabled: bool = False
    length_min: float = 0.03
    length_max: float = 0.15
    width_min: float = 0.03
    width_max: float = 0.15
    height_min: float = 0.03
    height_max: float = 0.20
    place_on_table: bool = True
    avoid_target_overlap: bool = True
    overlap_margin: float = 0.02
    max_resample_attempts: int = 50


@dataclass
class ObstacleVisualConfig:
    """Visual appearance for an obstacle."""
    enabled: bool = True
    color: List[float] = field(default_factory=lambda: [1.0, 0.85, 0.0, 1.0])


@dataclass
class ObstacleCollisionVisualConfig:
    """Visual appearance for obstacle collision/bounding box."""
    enabled: bool = True
    color: List[float] = field(default_factory=lambda: [1.0, 0.65, 0.0, 0.25])
    padding: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wireframe: bool = True


@dataclass
class StartConfig:
    """Start position configuration."""
    mode: str = "fixed"
    fixed_position: List[float] = field(default_factory=lambda: [0.35, -0.33, 0.10])
    random_bounds: RandomBoundsConfig = field(default_factory=RandomBoundsConfig)
    # Legacy fields (backward compat — parsed from YAML, converted internally)
    random: bool = False
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    random_start: bool = False
    random_space_enabled: bool = False
    _legacy_region_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _legacy_region_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @property
    def resolved_mode(self) -> str:
        """Effective start mode, resolving legacy fields."""
        if self.mode in ("fixed", "random"):
            return self.mode
        if self.random:
            return "random"
        if self.random_start:
            return "random"
        return "fixed"

    @property
    def random_bounds_min_np(self) -> np.ndarray:
        return self.random_bounds.min_np

    @property
    def random_bounds_max_np(self) -> np.ndarray:
        return self.random_bounds.max_np


@dataclass
class WorkspaceConfig:
    """Axis-aligned Cartesian workspace / search region."""
    name: str = "search_region"
    x_min: float = -0.200
    x_max: float = 0.500
    y_min: float = -0.800
    y_max: float = 0.000
    z_min: float = 0.020
    z_max: float = 0.320
    # Legacy fields (backward compat)
    min: List[float] = field(default_factory=list)
    max: List[float] = field(default_factory=list)

    @property
    def min_np(self) -> np.ndarray:
        if self.min:
            return np.array(self.min, dtype=np.float32)
        return np.array([self.x_min, self.y_min, self.z_min], dtype=np.float32)

    @property
    def max_np(self) -> np.ndarray:
        if self.max:
            return np.array(self.max, dtype=np.float32)
        return np.array([self.x_max, self.y_max, self.z_max], dtype=np.float32)

    @property
    def range_np(self) -> np.ndarray:
        return self.max_np - self.min_np


@dataclass
class TargetRegionConfig:
    """Target sampling region configuration."""
    mode: str = "random"
    enabled: bool = True
    fixed_position: List[float] = field(default_factory=lambda: [0.030, -0.535, 0.110])
    random_bounds: RandomBoundsConfig = field(default_factory=lambda: RandomBoundsConfig(
        min=[-0.135, -0.715, 0.10], max=[0.195, -0.385, 0.10]
    ))
    # Legacy fields (backward compat)
    random: bool = True
    min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fixed_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    random_target: bool = True
    random_space_enabled: bool = False

    @property
    def resolved_mode(self) -> str:
        """Effective target mode, resolving legacy fields."""
        if self.mode in ("fixed", "random"):
            return self.mode
        if self.random or self.random_target:
            return "random"
        return "fixed"

    @property
    def min_np(self) -> np.ndarray:
        return self.random_bounds.min_np

    @property
    def max_np(self) -> np.ndarray:
        return self.random_bounds.max_np


@dataclass
class FixedObjectConfig:
    """Fixed visual object (e.g. solid black block representing table base)."""
    enabled: bool
    name: str
    type: str
    center: List[float]
    size: List[float]
    color: List[float]
    collision: bool

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)

    @property
    def size_np(self) -> np.ndarray:
        return np.array(self.size, dtype=np.float32)

    @property
    def half_extent_np(self) -> np.ndarray:
        return self.size_np / 2.0


@dataclass
class PlaneConfig:
    """Plane / ground surface (visual only)."""
    enabled: bool
    z: float
    center: List[float]
    size: List[float]
    color: List[float]
    collision: bool

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)


@dataclass
class TableConfig:
    """
    Fixed black table / environment base.

    The table is part of the static scene — always visible and always collidable
    by default.  It provides environmental context (a physical work surface) and
    is independent of the optional box obstacle.
    """
    enabled: bool
    name: str
    type: str
    center: List[float]
    size: List[float]
    color: List[float]
    collision: bool

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)

    @property
    def size_np(self) -> np.ndarray:
        return np.array(self.size, dtype=np.float32)

    @property
    def half_extent_np(self) -> np.ndarray:
        return self.size_np / 2.0


@dataclass
class BoxConfig:
    """
    Optional box obstacle.

    Hidden and non-collidable by default.  Only visible and collidable when
    ``enabled`` is True.  This is distinct from the fixed table — the box is
    used for training scenarios that require an obstacle to be navigated around.
    """
    enabled: bool
    name: str
    type: str
    center: List[float]
    size: List[float]
    color: List[float]
    collision: bool
    safety_margin: float

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)

    @property
    def size_np(self) -> np.ndarray:
        return np.array(self.size, dtype=np.float32)

    @property
    def half_extent_np(self) -> np.ndarray:
        return self.size_np / 2.0

    @property
    def collision_half_extent_np(self) -> np.ndarray:
        return self.half_extent_np + self.safety_margin


@dataclass
class RandomRegionConfig:
    """Randomisation region for obstacles."""
    enabled: bool
    min: List[float]
    max: List[float]

    @property
    def min_np(self) -> np.ndarray:
        return np.array(self.min, dtype=np.float32)

    @property
    def max_np(self) -> np.ndarray:
        return np.array(self.max, dtype=np.float32)


@dataclass
class ObstacleConfig:
    """
    Box obstacle definition.

    Visibility: controlled by obstacle.visual.enabled (source of truth).
    Collision:  always active when obstacle.enabled is True.
    Randomisation: mode='random' samples centre from random_bounds.

    Schema::

        obstacle:
          enabled: true
          mode: fixed           # fixed | random — source of truth
          name: small_obstacle
          type: box
          center: [0.145, -0.550, 0.080]
          size: [0.100, 0.100, 0.100]
          safety_margin: 0.01

          random_bounds:
            min: [-0.100, -0.650, 0.060]
            max: [0.400, -0.400, 0.160]

          visual:
            enabled: true
            color: [1.0, 0.85, 0.0, 1.0]

          collision_visual:
            enabled: true
            color: [1.0, 0.65, 0.0, 0.25]
            padding: [0.0, 0.0, 0.0]
            wireframe: true
    """
    enabled: bool
    mode: str = "fixed"
    name: str = "small_obstacle"
    type: str = "box"
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: List[float] = field(default_factory=lambda: [0.10, 0.10, 0.10])
    safety_margin: float = 0.01
    random_bounds: RandomBoundsConfig = field(default_factory=RandomBoundsConfig)
    size_random: ObstacleSizeRandomConfig = field(default_factory=ObstacleSizeRandomConfig)
    # Visual configs
    visual: ObstacleVisualConfig = field(default_factory=ObstacleVisualConfig)
    collision_visual: ObstacleCollisionVisualConfig = field(
        default_factory=ObstacleCollisionVisualConfig
    )
    # Legacy fields (backward compat — parsed but not used in new flow)
    random: bool = False
    random_region: RandomRegionConfig = field(default_factory=RandomRegionConfig)

    @property
    def resolved_mode(self) -> str:
        """Effective obstacle mode, resolving legacy field."""
        if self.mode in ("fixed", "random"):
            return self.mode
        if self.random:
            return "random"
        return "fixed"

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)

    @property
    def size_np(self) -> np.ndarray:
        return np.array(self.size, dtype=np.float32)

    @property
    def half_extent_np(self) -> np.ndarray:
        return self.size_np / 2.0

    @property
    def collision_half_extent_np(self) -> np.ndarray:
        return self.half_extent_np + self.safety_margin

    @property
    def collision_aabb_min_np(self) -> np.ndarray:
        return self.center_np - self.collision_half_extent_np

    @property
    def collision_aabb_max_np(self) -> np.ndarray:
        return self.center_np + self.collision_half_extent_np

    @property
    def collision_visual_padding_np(self) -> np.ndarray:
        return np.array(self.collision_visual.padding, dtype=np.float32)

    @property
    def collision_visual_half_extent_np(self) -> np.ndarray:
        return self.half_extent_np + self.collision_visual_padding_np


@dataclass
class TerminationConfig:
    """Termination conditions for an episode."""
    goal_threshold: float = 0.03
    collision_terminate: bool = True
    workspace_terminate: bool = True


@dataclass
class CollisionConfig:
    """Collision behavior for the environment."""
    enabled: bool = True


@dataclass
class EnvironmentStepConfig:
    """Environment step / episode parameters."""
    observation_type: str = "frame_only"  # only "frame_only" is supported
    action_step: float = 0.01
    max_episode_steps: int = 500

    @property
    def max_steps(self) -> int:
        """Alias for max_episode_steps (backward compat)."""
        return self.max_episode_steps


@dataclass
class RewardConfig:
    """
    Simple distance-based reward for Cartesian path planning.

    r_t = r_success + r_collision + r_distance + r_workspace + r_episode + r_time + r_shake

    General for random targets, random obstacles, and random obstacle sizes.
    No geometric waypoints, no phase logic, no safe-zone strategies.
    """
    # Terminal rewards
    success_bonus: float = 10.0
    collision_penalty: float = 300.0
    workspace_penalty: float = 300.0
    timeout_penalty: float = 50.0

    # Distance shaping
    distance_scale: float = 1.0

    # Time penalty per step
    time_penalty: float = 0.01

    # Shake penalty
    shake_penalty_scale: float = 0.005
    shake_window: int = 10
    shake_dot_threshold: float = 0.0
    shake_min_movement: float = 1e-6





@dataclass
class CameraConfig:
    """PyBullet orbit camera settings."""
    distance: float
    yaw: float
    pitch: float
    target: List[float]

    @property
    def target_np(self) -> np.ndarray:
        return np.array(self.target, dtype=np.float32)


@dataclass
class GroundConfig:
    """Ground / table reference surface settings."""
    z: float
    center: List[float]
    size: List[float]

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float32)


@dataclass
class StyleConfig:
    """Visual style for PyBullet scene elements."""
    frame_axis_length: float
    agent_radius: float
    path_line_width: int


@dataclass
class VisualizationConfig:
    """
    PyBullet visualization settings.

    Global on/off: ``enabled``
    Obstacle visibility: driven by ``obstacle.visual.enabled`` and
    ``obstacle.collision_visual.enabled`` inside the obstacle section —
    NOT by show_box / show_obstacle_safety_zone.
    """
    enabled: bool
    gui: bool
    hide_debug_ui: bool
    show_workspace: bool
    show_target_region: bool
    show_start_frame: bool
    show_target_frame: bool
    show_agent_frame: bool
    show_table: bool
    show_path: bool
    show_ground_plane: bool
    show_labels: bool
    show_plane: bool
    camera: CameraConfig
    style: StyleConfig


@dataclass
class TrainingConfig:
    """Training configuration (unified — replaces config/experiments/*.yaml)."""
    algorithm: str
    total_timesteps: int
    seed: int
    device: str
    # Parallel environments
    n_envs: int = 1
    vec_env_type: str = "auto"
    # Progress & logging
    progress_bar: bool = True
    log_interval: int = 50
    episode_log_interval: int = 1000
    # Checkpointing
    eval_freq: int = 50000
    save_freq: int = 50000


@dataclass
class EvaluationConfig:
    """Evaluation defaults."""
    seed: int
    num_episodes: int
    export_waypoints: bool


# --------------------------------------------------------------------------- #
# Training / Algorithm config (unified — replaces config/experiments/*.yaml)
# --------------------------------------------------------------------------- #

@dataclass
class TD3Config:
    """TD3 algorithm hyperparameters. Also used as defaults for DDPG."""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    buffer_size: int = 1000000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})


@dataclass
class DDPGConfig:
    """DDPG algorithm hyperparameters (same base params as TD3, without the TD3-specific twin Q update)."""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0001
    buffer_size: int = 1000000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})


@dataclass
class SACConfig:
    """SAC algorithm hyperparameters."""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0001
    buffer_size: int = 1000000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_entropy: str = "auto"
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters (on-policy, no replay buffer)."""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})


@dataclass
class ActionNoiseConfig:
    """Action noise / exploration settings."""
    enabled: bool = True
    type: str = "NormalActionNoise"
    mean: float = 0.0
    sigma: float = 0.1


@dataclass
class LoggingConfig:
    """Logging and checkpointing settings."""
    tensorboard: bool = True
    log_dir: str = "Data/Training"
    save_model: bool = True
    save_replay_buffer: bool = True


@dataclass
class CurriculumStageConfig:
    """A single curriculum stage."""
    name: str
    description: str
    mode: str
    jitter_radius: float = 0.0
    timesteps: int = 500000
    anchor_weight_uniform: bool = True
    inner_margin: float = 0.15


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = False
    anchor_box_min: List[float] = field(default_factory=lambda: [-0.06, -0.64, 0.08])
    anchor_box_max: List[float] = field(default_factory=lambda: [0.36, -0.16, 0.26])
    fixed_anchors: List[List[float]] = field(default_factory=list)
    stages: List[CurriculumStageConfig] = field(default_factory=list)
    log_interval_episodes: int = 50


# --------------------------------------------------------------------------- #
# Top-level Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """
    Centralised configuration object for the frame-only Cartesian DRL environment.

    All environment parameters used across the project are read from this object.
    No module should contain a hardcoded environment geometry constant.
    """

    project: ProjectConfig
    start: StartConfig
    workspace: WorkspaceConfig
    target_region: TargetRegionConfig
    obstacle: ObstacleConfig
    collision: CollisionConfig
    termination: TerminationConfig
    table: TableConfig
    plane: PlaneConfig
    environment: EnvironmentStepConfig
    reward: RewardConfig
    visualization: VisualizationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    # Unified training / algorithm config (replaces config/experiments/*.yaml)
    td3: TD3Config = field(default_factory=TD3Config)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    action_noise: ActionNoiseConfig = field(default_factory=ActionNoiseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path
            Absolute or relative path to the YAML config file.

        Returns
        -------
        Config
            Populated configuration object.
        """
        raw = _load_yaml(Path(path))
        return cls._from_dict(raw)

    @classmethod
    def from_default(cls) -> "Config":
        """
        Load the default ``config/environment.yaml`` from the project root.

        The project root is auto-detected by walking upward until a directory
        containing ``config/`` and (``Training/`` or ``src/``) is found.
        """
        default_path = _default_config_path()
        return cls.from_yaml(default_path)

    @classmethod
    def _from_dict(cls, d: dict) -> "Config":
        """Build a Config from a parsed YAML dict."""
        project_raw = d.get("project", {})

        # Support both "project_type" (new) and "robot_name" (backward compat).
        # "robot_name" is accepted but mapped to project_type internally.
        project_type = project_raw.get("project_type", project_raw.get("robot_name", "FRAME_ONLY"))
        project = ProjectConfig(
            name=project_raw.get("name", "DRL_Pathplanning_trainning"),
            project_type=project_type,
            unit=project_raw.get("unit", "meter"),
        )

        # ---- Start config ----
        # New schema: mode, fixed_position, random_bounds.min/max
        # Legacy schema: random, random_space, region_min, region_max (backward compat)
        start_raw = d.get("start", {})
        _raw_mode = start_raw.get("mode")

        # Determine effective mode from new or legacy schema
        if _raw_mode in ("fixed", "random"):
            _effective_mode = _raw_mode
        elif start_raw.get("random"):
            _effective_mode = "random"
        elif start_raw.get("random_start"):
            _effective_mode = "random"
        elif start_raw.get("random_space", {}).get("enabled"):
            _effective_mode = "random"
        else:
            _effective_mode = "fixed"

        # Parse fixed_position (new) or legacy alternatives
        _fixed_pos = start_raw.get("fixed_position") or start_raw.get("position", [0.35, -0.33, 0.10])

        # Parse random_bounds (new) or convert legacy random_space/region_min/max
        _rb_raw = start_raw.get("random_bounds", {})
        if _rb_raw:
            # New schema: random_bounds.min / random_bounds.max
            _rb_min = _rb_raw.get("min", [0.30, -0.75, 0.10])
            _rb_max = _rb_raw.get("max", [0.45, -0.30, 0.20])
        else:
            # Legacy: random_space or region_min/region_max
            _rs = start_raw.get("random_space", {})
            if _rs.get("enabled"):
                _rb_min = [_rs.get("x_min", 0.30), _rs.get("y_min", -0.75), _rs.get("z_min", 0.10)]
                _rb_max = [_rs.get("x_max", 0.45), _rs.get("y_max", -0.30), _rs.get("z_max", 0.20)]
            elif start_raw.get("region_min") and start_raw.get("region_max"):
                _rb_min = start_raw.get("region_min")
                _rb_max = start_raw.get("region_max")
            else:
                _rb_min = [0.30, -0.75, 0.10]
                _rb_max = [0.45, -0.30, 0.20]

        random_bounds = RandomBoundsConfig(min=_rb_min, max=_rb_max)
        start = StartConfig(
            mode=_effective_mode,
            fixed_position=_fixed_pos,
            random_bounds=random_bounds,
            # Legacy fields (for backward compat inspection)
            random=bool(start_raw.get("random", False)),
            position=start_raw.get("position", [0.0, 0.0, 0.0]),
            random_start=bool(start_raw.get("random_start", False)),
            random_space_enabled=start_raw.get("random_space", {}).get("enabled", False),
            _legacy_region_min=start_raw.get("region_min", [0.0, 0.0, 0.0]),
            _legacy_region_max=start_raw.get("region_max", [1.0, 1.0, 1.0]),
        )

        ws_raw = d.get("workspace", {})
        workspace = WorkspaceConfig(
            name=ws_raw.get("name", "search_region"),
            x_min=ws_raw.get("x_min", -0.200),
            x_max=ws_raw.get("x_max", 0.500),
            y_min=ws_raw.get("y_min", -0.800),
            y_max=ws_raw.get("y_max", 0.000),
            z_min=ws_raw.get("z_min", 0.020),
            z_max=ws_raw.get("z_max", 0.320),
            # Legacy
            min=ws_raw.get("min", []),
            max=ws_raw.get("max", []),
        )

        # ---- Target region config ----
        # New schema: mode, fixed_position, random_bounds.min/max
        # Legacy schema: random, random_space, min/max (backward compat)
        target_raw = d.get("target_region", d.get("target", {}))
        _t_raw_mode = target_raw.get("mode")

        if _t_raw_mode in ("fixed", "random"):
            _t_mode = _t_raw_mode
        elif target_raw.get("random") or target_raw.get("random_target"):
            _t_mode = "random"
        elif target_raw.get("random_space", {}).get("enabled"):
            _t_mode = "random"
        else:
            _t_mode = "random"

        _t_fixed_pos = target_raw.get("fixed_position") or target_raw.get("fixed_target", [0.030, -0.535, 0.110])

        _t_rb_raw = target_raw.get("random_bounds", {})
        if _t_rb_raw:
            _t_rb_min = _t_rb_raw.get("min", [-0.135, -0.715, 0.10])
            _t_rb_max = _t_rb_raw.get("max", [0.195, -0.385, 0.10])
        else:
            _t_rs = target_raw.get("random_space", {})
            if _t_rs.get("enabled"):
                _t_rb_min = [_t_rs.get("x_min", -0.135), _t_rs.get("y_min", -0.715), _t_rs.get("z_min", 0.10)]
                _t_rb_max = [_t_rs.get("x_max", 0.195), _t_rs.get("y_max", -0.385), _t_rs.get("z_max", 0.10)]
            elif target_raw.get("min") and target_raw.get("max"):
                _t_rb_min = target_raw.get("min")
                _t_rb_max = target_raw.get("max")
            else:
                _t_rb_min = [-0.135, -0.715, 0.10]
                _t_rb_max = [0.195, -0.385, 0.10]

        t_random_bounds = RandomBoundsConfig(min=_t_rb_min, max=_t_rb_max)
        target_region = TargetRegionConfig(
            mode=_t_mode,
            enabled=target_raw.get("enabled", True),
            fixed_position=_t_fixed_pos,
            random_bounds=t_random_bounds,
            # Legacy fields
            random=bool(target_raw.get("random", target_raw.get("random_target", True))),
            min=target_raw.get("min", []),
            max=target_raw.get("max", []),
            fixed_target=target_raw.get("fixed_target", [0.0, 0.0, 0.0]),
            random_target=bool(target_raw.get("random_target", True)),
            random_space_enabled=target_raw.get("random_space", {}).get("enabled", False),
        )

        # ---- Obstacle config ----
        # New schema: enabled, mode, center, size, random_bounds
        # Legacy schema: enabled, random, random_space/random_region (backward compat)
        obs_raw = d.get("obstacle", d.get("box", {}))
        _obs_mode = obs_raw.get("mode")
        _obs_random = obs_raw.get("random", False)

        if _obs_mode in ("fixed", "random"):
            _eff_mode = _obs_mode
        elif _obs_random:
            _eff_mode = "random"
        elif obs_raw.get("random_space", {}).get("enabled"):
            _eff_mode = "random"
        elif obs_raw.get("random_region", {}).get("enabled"):
            _eff_mode = "random"
        else:
            _eff_mode = "fixed"

        # Parse random_bounds (new) or convert legacy random_space/random_region
        _rb_raw = obs_raw.get("random_bounds", {})
        if _rb_raw:
            _rb_min = _rb_raw.get("min", [-0.100, -0.650, 0.060])
            _rb_max = _rb_raw.get("max", [0.400, -0.400, 0.160])
        else:
            # Legacy: random_space (x_min/y_min/z_min/x_max/y_max/z_max)
            _rs = obs_raw.get("random_space", obs_raw.get("random_region", {}))
            if _rs.get("enabled") or ("x_min" in _rs):
                _rb_min = [_rs.get("x_min", -0.100), _rs.get("y_min", -0.650), _rs.get("z_min", 0.060)]
                _rb_max = [_rs.get("x_max", 0.400), _rs.get("y_max", -0.400), _rs.get("z_max", 0.160)]
            elif _rs.get("min") and _rs.get("max"):
                _rb_min = _rs.get("min")
                _rb_max = _rs.get("max")
            else:
                _rb_min = [-0.100, -0.650, 0.060]
                _rb_max = [0.400, -0.400, 0.160]

        _obs_rb = RandomBoundsConfig(min=_rb_min, max=_rb_max)

        # Legacy random_region (for backward compat inspection)
        _lr_raw = obs_raw.get("random_region", obs_raw.get("random_space", {}))
        if "x_min" in _lr_raw or "min" in _lr_raw:
            _lr = RandomRegionConfig(
                enabled=_lr_raw.get("enabled", False),
                min=_lr_raw.get("min", [_lr_raw.get("x_min", 0.0), _lr_raw.get("y_min", 0.0), _lr_raw.get("z_min", 0.0)]),
                max=_lr_raw.get("max", [_lr_raw.get("x_max", 1.0), _lr_raw.get("y_max", 1.0), _lr_raw.get("z_max", 1.0)]),
            )
        else:
            _lr = RandomRegionConfig(
                enabled=_lr_raw.get("enabled", False),
                min=_lr_raw.get("min", [0.0, 0.0, 0.0]),
                max=_lr_raw.get("max", [1.0, 1.0, 1.0]),
            )

        # Visual config
        _vis_raw = obs_raw.get("visual", {})
        obstacle_visual = ObstacleVisualConfig(
            enabled=bool(_vis_raw.get("enabled", True)),
            color=_vis_raw.get("color", [1.0, 0.85, 0.0, 1.0]),
        )

        # Collision visual config
        _cv_raw = obs_raw.get("collision_visual", {})
        obstacle_collision_visual = ObstacleCollisionVisualConfig(
            enabled=bool(_cv_raw.get("enabled", True)),
            color=_cv_raw.get("color", [1.0, 0.65, 0.0, 0.25]),
            padding=_cv_raw.get("padding", [0.0, 0.0, 0.0]),
            wireframe=bool(_cv_raw.get("wireframe", True)),
        )

        # Size random config
        _sr_raw = obs_raw.get("size_random", {})
        obstacle_size_random = ObstacleSizeRandomConfig(
            enabled=bool(_sr_raw.get("enabled", False)),
            length_min=float(_sr_raw.get("length_min", 0.03)),
            length_max=float(_sr_raw.get("length_max", 0.15)),
            width_min=float(_sr_raw.get("width_min", 0.03)),
            width_max=float(_sr_raw.get("width_max", 0.15)),
            height_min=float(_sr_raw.get("height_min", 0.03)),
            height_max=float(_sr_raw.get("height_max", 0.20)),
            place_on_table=bool(_sr_raw.get("place_on_table", True)),
            avoid_target_overlap=bool(_sr_raw.get("avoid_target_overlap", True)),
            overlap_margin=float(_sr_raw.get("overlap_margin", 0.02)),
            max_resample_attempts=int(_sr_raw.get("max_resample_attempts", 50)),
        )

        obstacle = ObstacleConfig(
            enabled=bool(obs_raw.get("enabled", False)),
            mode=_eff_mode,
            name=obs_raw.get("name", "small_obstacle"),
            type=obs_raw.get("type", "box"),
            center=obs_raw.get("center", [0.0, 0.0, 0.0]),
            size=obs_raw.get("size", [0.10, 0.10, 0.10]),
            safety_margin=obs_raw.get("safety_margin", 0.01),
            random_bounds=_obs_rb,
            size_random=obstacle_size_random,
            random_region=_lr,
            random=bool(_obs_random),
            visual=obstacle_visual,
            collision_visual=obstacle_collision_visual,
        )

        collision_raw = d.get("collision", {})
        collision = CollisionConfig(
            enabled=bool(collision_raw.get("enabled", True)),
        )

        term_raw = d.get("termination", {})
        termination = TerminationConfig(
            goal_threshold=term_raw.get("goal_threshold", 0.01),
            collision_terminate=bool(term_raw.get("collision_terminate", True)),
            workspace_terminate=bool(term_raw.get("workspace_terminate", True)),
        )

        env_raw = d.get("environment", {})
        environment = EnvironmentStepConfig(
            observation_type=env_raw.get("observation_type", "frame_only"),
            action_step=env_raw.get("action_step", 0.01),
            max_episode_steps=env_raw.get("max_episode_steps", env_raw.get("max_steps", 500)),
        )

        rw_raw = d.get("reward", {})
        reward = RewardConfig(
            success_bonus=rw_raw.get("success_bonus", 10.0),
            collision_penalty=rw_raw.get("collision_penalty", 300.0),
            workspace_penalty=rw_raw.get("workspace_penalty", 300.0),
            timeout_penalty=rw_raw.get("timeout_penalty", 50.0),
            distance_scale=rw_raw.get("distance_scale", 1.0),
            time_penalty=rw_raw.get("time_penalty", 0.01),
            shake_penalty_scale=rw_raw.get("shake_penalty_scale", 0.005),
            shake_window=rw_raw.get("shake_window", 10),
            shake_dot_threshold=rw_raw.get("shake_dot_threshold", 0.0),
            shake_min_movement=rw_raw.get("shake_min_movement", 1e-6),
        )

        plane_raw = d.get("plane", {})
        plane = PlaneConfig(
            enabled=plane_raw.get("enabled", True),
            z=plane_raw.get("z", -0.330),
            center=plane_raw.get("center", [0.150, -0.350, -0.330]),
            size=plane_raw.get("size", [1.0, 1.0]),
            color=plane_raw.get("color", [0.82, 0.82, 0.82, 0.35]),
            collision=plane_raw.get("collision", False),
        )

        table_raw = d.get("table", {})
        table = TableConfig(
            enabled=table_raw.get("enabled", True),
            name=table_raw.get("name", "table"),
            type=table_raw.get("type", "box"),
            center=table_raw.get("center", [0.030, -0.550, -0.150]),
            size=table_raw.get("size", [0.330, 0.330, 0.360]),
            color=table_raw.get("color", [0.0, 0.0, 0.0, 1.0]),
            collision=table_raw.get("collision", True),
        )

        vis_raw = d.get("visualization", {})
        cam_raw = vis_raw.get("camera", {})
        camera = CameraConfig(
            distance=cam_raw.get("distance", 1.2),
            yaw=cam_raw.get("yaw", -60.0),
            pitch=cam_raw.get("pitch", -35.0),
            target=cam_raw.get("target", [0.150, -0.350, 0.170]),
        )
        style_raw = vis_raw.get("style", {})
        style = StyleConfig(
            frame_axis_length=style_raw.get("frame_axis_length", 0.05),
            agent_radius=style_raw.get("agent_radius", 0.015),
            path_line_width=style_raw.get("path_line_width", 4),
        )
        visualization = VisualizationConfig(
            enabled=vis_raw.get("enabled", False),
            gui=vis_raw.get("gui", True),
            hide_debug_ui=vis_raw.get("hide_debug_ui", True),
            show_workspace=vis_raw.get("show_workspace", True),
            show_target_region=vis_raw.get("show_target_region", True),
            show_start_frame=vis_raw.get("show_start_frame", True),
            show_target_frame=vis_raw.get("show_target_frame", True),
            show_agent_frame=vis_raw.get("show_agent_frame", True),
            show_table=vis_raw.get("show_table", vis_raw.get("show_fixed_objects", True)),
            show_path=vis_raw.get("show_path", True),
            show_ground_plane=vis_raw.get("show_ground_plane", True),
            show_labels=vis_raw.get("show_labels", True),
            show_plane=vis_raw.get("show_plane", True),
            camera=camera,
            style=style,
        )

        train_raw = d.get("training", {})
        training = TrainingConfig(
            algorithm=train_raw.get("algorithm", "DDPG"),
            total_timesteps=train_raw.get("total_timesteps", 500_000),
            seed=train_raw.get("seed", 42),
            device=train_raw.get("device", "auto"),
            n_envs=train_raw.get("n_envs", 1),
            vec_env_type=train_raw.get("vec_env_type", "auto"),
            progress_bar=train_raw.get("progress_bar", True),
            log_interval=train_raw.get("log_interval", 50),
            episode_log_interval=train_raw.get("episode_log_interval", 1000),
            eval_freq=train_raw.get("eval_freq", 50000),
            save_freq=train_raw.get("save_freq", 50000),
        )

        eval_raw = d.get("evaluation", {})
        evaluation = EvaluationConfig(
            seed=eval_raw.get("seed", 42),
            num_episodes=eval_raw.get("num_episodes", 100),
            export_waypoints=eval_raw.get("export_waypoints", True),
        )

        # ---- Unified training / algorithm config ----
        td3_raw = d.get("td3", {})
        td3_pk_raw = td3_raw.get("policy_kwargs", {})
        td3 = TD3Config(
            policy=td3_raw.get("policy", "MlpPolicy"),
            learning_rate=td3_raw.get("learning_rate", 0.0003),
            buffer_size=td3_raw.get("buffer_size", 1000000),
            learning_starts=td3_raw.get("learning_starts", 10000),
            batch_size=td3_raw.get("batch_size", 256),
            tau=td3_raw.get("tau", 0.005),
            gamma=td3_raw.get("gamma", 0.99),
            train_freq=td3_raw.get("train_freq", 1),
            gradient_steps=td3_raw.get("gradient_steps", 1),
            policy_delay=td3_raw.get("policy_delay", 2),
            target_policy_noise=td3_raw.get("target_policy_noise", 0.2),
            target_noise_clip=td3_raw.get("target_noise_clip", 0.5),
            policy_kwargs=td3_pk_raw if td3_pk_raw else {"net_arch": [256, 256]},
        )

        ddpg_raw = d.get("ddpg", {})
        ddpg_pk_raw = ddpg_raw.get("policy_kwargs", {})
        ddpg_cfg = DDPGConfig(
            policy=ddpg_raw.get("policy", "MlpPolicy"),
            learning_rate=ddpg_raw.get("learning_rate", 0.0001),
            buffer_size=ddpg_raw.get("buffer_size", 1000000),
            learning_starts=ddpg_raw.get("learning_starts", 10000),
            batch_size=ddpg_raw.get("batch_size", 256),
            tau=ddpg_raw.get("tau", 0.005),
            gamma=ddpg_raw.get("gamma", 0.99),
            train_freq=ddpg_raw.get("train_freq", 1),
            gradient_steps=ddpg_raw.get("gradient_steps", 1),
            policy_kwargs=ddpg_pk_raw if ddpg_pk_raw else {"net_arch": [256, 256]},
        )

        sac_raw = d.get("sac", {})
        sac_pk_raw = sac_raw.get("policy_kwargs", {})
        sac_cfg = SACConfig(
            policy=sac_raw.get("policy", "MlpPolicy"),
            learning_rate=sac_raw.get("learning_rate", 0.0001),
            buffer_size=sac_raw.get("buffer_size", 1000000),
            learning_starts=sac_raw.get("learning_starts", 10000),
            batch_size=sac_raw.get("batch_size", 256),
            tau=sac_raw.get("tau", 0.005),
            gamma=sac_raw.get("gamma", 0.99),
            train_freq=sac_raw.get("train_freq", 1),
            gradient_steps=sac_raw.get("gradient_steps", 1),
            ent_coef=sac_raw.get("ent_coef", "auto"),
            target_entropy=sac_raw.get("target_entropy", "auto"),
            policy_kwargs=sac_pk_raw if sac_pk_raw else {"net_arch": [256, 256]},
        )

        ppo_raw = d.get("ppo", {})
        ppo_pk_raw = ppo_raw.get("policy_kwargs", {})
        ppo_cfg = PPOConfig(
            policy=ppo_raw.get("policy", "MlpPolicy"),
            learning_rate=ppo_raw.get("learning_rate", 0.0003),
            n_steps=ppo_raw.get("n_steps", 2048),
            batch_size=ppo_raw.get("batch_size", 256),
            n_epochs=ppo_raw.get("n_epochs", 10),
            gamma=ppo_raw.get("gamma", 0.99),
            gae_lambda=ppo_raw.get("gae_lambda", 0.95),
            clip_range=ppo_raw.get("clip_range", 0.2),
            ent_coef=ppo_raw.get("ent_coef", 0.0),
            vf_coef=ppo_raw.get("vf_coef", 0.5),
            max_grad_norm=ppo_raw.get("max_grad_norm", 0.5),
            policy_kwargs=ppo_pk_raw if ppo_pk_raw else {"net_arch": [256, 256]},
        )

        noise_raw = d.get("action_noise", {})
        action_noise = ActionNoiseConfig(
            enabled=noise_raw.get("enabled", True),
            type=noise_raw.get("type", "NormalActionNoise"),
            mean=float(noise_raw.get("mean", 0.0)),
            sigma=float(noise_raw.get("sigma", 0.1)),
        )

        log_raw = d.get("logging", {})
        logging = LoggingConfig(
            tensorboard=log_raw.get("tensorboard", True),
            log_dir=log_raw.get("log_dir", "Data/Training"),
            save_model=log_raw.get("save_model", True),
            save_replay_buffer=log_raw.get("save_replay_buffer", True),
        )

        # ---- Curriculum config ----
        curr_raw = d.get("curriculum", {})
        curr_stages_raw = curr_raw.get("stages", [])
        curr_stages = [
            CurriculumStageConfig(
                name=s.get("name", f"stage_{i}"),
                description=s.get("description", ""),
                mode=s.get("mode", "fixed_anchors"),
                jitter_radius=s.get("jitter_radius", 0.0),
                timesteps=s.get("timesteps", 500000),
                anchor_weight_uniform=s.get("anchor_weight_uniform", True),
                inner_margin=s.get("inner_margin", 0.15),
            )
            for i, s in enumerate(curr_stages_raw)
        ]
        anchor_box = curr_raw.get("anchor_box", {})
        curriculum = CurriculumConfig(
            enabled=curr_raw.get("enabled", False),
            anchor_box_min=anchor_box.get("min", [-0.06, -0.64, 0.08]),
            anchor_box_max=anchor_box.get("max", [0.36, -0.16, 0.26]),
            fixed_anchors=curr_raw.get("fixed_anchors", []),
            stages=curr_stages,
            log_interval_episodes=curr_raw.get("log_interval_episodes", 50),
        )

        cfg = cls(
            project=project,
            start=start,
            workspace=workspace,
            target_region=target_region,
            obstacle=obstacle,
            collision=collision,
            termination=termination,
            table=table,
            plane=plane,
            environment=environment,
            reward=reward,
            visualization=visualization,
            training=training,
            evaluation=evaluation,
            td3=td3,
            ddpg=ddpg_cfg,
            sac=sac_cfg,
            ppo=ppo_cfg,
            action_noise=action_noise,
            logging=logging,
            curriculum=curriculum,
        )

        _validate(cfg)
        return cfg


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate(cfg: Config) -> None:
    """
    Validate the loaded configuration.

    Raises
    ------
    ValueError
        If any invariant is violated.
    """
    ws = cfg.workspace
    if not all(a < b for a, b in zip(ws.min_np, ws.max_np)):
        raise ValueError(
            f"workspace.min must be less than workspace.max in all axes. "
            f"Got min={ws.min_np.tolist()}, max={ws.max_np.tolist()}"
        )

    if cfg.start.resolved_mode == "fixed":
        for i, v in enumerate(cfg.start.fixed_position):
            if not (ws.min_np[i] <= v <= ws.max_np[i]):
                raise ValueError(
                    f"start.fixed_position[{i}]={v} is outside workspace bounds "
                    f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                )

    if cfg.start.resolved_mode == "random":
        _srb_min = cfg.start.random_bounds.min_np
        _srb_max = cfg.start.random_bounds.max_np
        for i in range(3):
            if not (ws.min_np[i] <= _srb_min[i] <= ws.max_np[i]):
                raise ValueError(
                    f"start.random_bounds.min[{i}]={_srb_min[i]} is outside workspace bounds "
                    f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                )
            if not (ws.min_np[i] <= _srb_max[i] <= ws.max_np[i]):
                raise ValueError(
                    f"start.random_bounds.max[{i}]={_srb_max[i]} is outside workspace bounds "
                    f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                )
            if not (_srb_min[i] <= _srb_max[i]):
                raise ValueError(
                    f"start.random_bounds.min[{i}]={_srb_min[i]} must be <= "
                    f"start.random_bounds.max[{i}]={_srb_max[i]}"
                )

    # Target bounds validation
    _trb_min = cfg.target_region.random_bounds.min_np
    _trb_max = cfg.target_region.random_bounds.max_np
    for i in range(3):
        if not (ws.min_np[i] <= _trb_min[i] <= ws.max_np[i]):
            raise ValueError(
                f"target.random_bounds.min[{i}]={_trb_min[i]} is outside workspace bounds "
                f"[{ws.min_np[i]}, {ws.max_np[i]}]"
            )
        if not (ws.min_np[i] <= _trb_max[i] <= ws.max_np[i]):
            raise ValueError(
                f"target.random_bounds.max[{i}]={_trb_max[i]} is outside workspace bounds "
                f"[{ws.min_np[i]}, {ws.max_np[i]}]"
            )
        if not (_trb_min[i] <= _trb_max[i]):
            raise ValueError(
                f"target.random_bounds.min[{i}]={_trb_min[i]} must be <= "
                f"target.random_bounds.max[{i}]={_trb_max[i]}"
            )

    for i, v in enumerate(cfg.target_region.fixed_position):
        if not (ws.min_np[i] <= v <= ws.max_np[i]):
            raise ValueError(
                f"target_region.fixed_position[{i}]={v} is outside workspace bounds "
                f"[{ws.min_np[i]}, {ws.max_np[i]}]"
            )

    if not all(s > 0 for s in cfg.plane.size):
        raise ValueError(f"plane.size must be positive. Got {cfg.plane.size}")

    if cfg.table.enabled:
        if not all(s > 0 for s in cfg.table.size):
            raise ValueError(f"table.size must be positive. Got {cfg.table.size}")

    if cfg.obstacle.enabled:
        if not all(s > 0 for s in cfg.obstacle.size):
            raise ValueError(f"obstacle.size must be positive. Got {cfg.obstacle.size}")
        if cfg.obstacle.safety_margin < 0:
            raise ValueError(
                f"obstacle.safety_margin must be non-negative. Got {cfg.obstacle.safety_margin}"
            )
        if cfg.obstacle.resolved_mode == "fixed":
            for i, v in enumerate(cfg.obstacle.center):
                if not (ws.min_np[i] <= v <= ws.max_np[i]):
                    raise ValueError(
                        f"obstacle.center[{i}]={v} is outside workspace bounds "
                        f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                    )
        if cfg.obstacle.resolved_mode == "random":
            _orb_min = cfg.obstacle.random_bounds.min_np
            _orb_max = cfg.obstacle.random_bounds.max_np
            for i in range(3):
                if not (ws.min_np[i] <= _orb_min[i] <= ws.max_np[i]):
                    raise ValueError(
                        f"obstacle.random_bounds.min[{i}]={_orb_min[i]} is outside workspace bounds "
                        f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                    )
                if not (ws.min_np[i] <= _orb_max[i] <= ws.max_np[i]):
                    raise ValueError(
                        f"obstacle.random_bounds.max[{i}]={_orb_max[i]} is outside workspace bounds "
                        f"[{ws.min_np[i]}, {ws.max_np[i]}]"
                    )
                if not (_orb_min[i] <= _orb_max[i]):
                    raise ValueError(
                        f"obstacle.random_bounds.min[{i}]={_orb_min[i]} must be <= "
                        f"obstacle.random_bounds.max[{i}]={_orb_max[i]}"
                    )

        sr = cfg.obstacle.size_random
        if sr.enabled:
            if not (sr.length_min > 0):
                raise ValueError(
                    f"obstacle.size_random.length_min must be > 0. Got {sr.length_min}"
                )
            if not (sr.length_max >= sr.length_min):
                raise ValueError(
                    f"obstacle.size_random.length_max={sr.length_max} "
                    f"must be >= length_min={sr.length_min}"
                )
            if not (sr.width_min > 0):
                raise ValueError(
                    f"obstacle.size_random.width_min must be > 0. Got {sr.width_min}"
                )
            if not (sr.width_max >= sr.width_min):
                raise ValueError(
                    f"obstacle.size_random.width_max={sr.width_max} "
                    f"must be >= width_min={sr.width_min}"
                )
            if not (sr.height_min > 0):
                raise ValueError(
                    f"obstacle.size_random.height_min must be > 0. Got {sr.height_min}"
                )
            if not (sr.height_max >= sr.height_min):
                raise ValueError(
                    f"obstacle.size_random.height_max={sr.height_max} "
                    f"must be >= height_min={sr.height_min}"
                )
            if not (sr.max_resample_attempts >= 1):
                raise ValueError(
                    f"obstacle.size_random.max_resample_attempts must be >= 1. "
                    f"Got {sr.max_resample_attempts}"
                )
            if not (sr.overlap_margin >= 0):
                raise ValueError(
                    f"obstacle.size_random.overlap_margin must be >= 0. "
                    f"Got {sr.overlap_margin}"
                )

    if cfg.environment.action_step <= 0:
        raise ValueError(
            f"environment.action_step must be positive. Got {cfg.environment.action_step}"
        )

    if cfg.environment.max_episode_steps <= 0:
        raise ValueError(
            f"environment.max_episode_steps must be positive. Got {cfg.environment.max_episode_steps}"
        )

    if cfg.termination.goal_threshold <= 0:
        raise ValueError(
            f"termination.goal_threshold must be positive. Got {cfg.termination.goal_threshold}"
        )
    train_raw_key = "training"
    if cfg.training.algorithm not in ("TD3", "DDPG", "SAC", "PPO"):
        raise ValueError(
            f"training.algorithm must be 'TD3', 'DDPG', 'SAC', or 'PPO'. "
            f"Got '{cfg.training.algorithm}'"
        )
    if cfg.training.total_timesteps <= 0:
        raise ValueError(
            f"training.total_timesteps must be positive. Got {cfg.training.total_timesteps}"
        )
    if cfg.training.n_envs <= 0:
        raise ValueError(
            f"training.n_envs must be positive. Got {cfg.training.n_envs}"
        )

    # ---- TD3 config validation ----
    algo = cfg.training.algorithm.upper()
    if algo in ("TD3", "DDPG"):
        td3 = cfg.td3
        if td3.learning_rate <= 0:
            raise ValueError(f"td3.learning_rate must be positive. Got {td3.learning_rate}")
        if td3.batch_size <= 0:
            raise ValueError(f"td3.batch_size must be positive. Got {td3.batch_size}")
        if td3.buffer_size <= 0:
            raise ValueError(f"td3.buffer_size must be positive. Got {td3.buffer_size}")
        if td3.tau <= 0 or td3.tau > 1:
            raise ValueError(f"td3.tau must be in (0, 1]. Got {td3.tau}")
        if td3.gamma <= 0 or td3.gamma > 1:
            raise ValueError(f"td3.gamma must be in (0, 1]. Got {td3.gamma}")
        if algo == "TD3":
            if td3.policy_delay <= 0:
                raise ValueError(f"td3.policy_delay must be positive. Got {td3.policy_delay}")

    # ---- PPO config validation ----
    if algo == "PPO":
        ppo = cfg.ppo
        if ppo.learning_rate <= 0:
            raise ValueError(f"ppo.learning_rate must be positive. Got {ppo.learning_rate}")
        if ppo.n_steps <= 0:
            raise ValueError(f"ppo.n_steps must be positive. Got {ppo.n_steps}")
        if ppo.batch_size <= 0:
            raise ValueError(f"ppo.batch_size must be positive. Got {ppo.batch_size}")
        if ppo.n_epochs <= 0:
            raise ValueError(f"ppo.n_epochs must be positive. Got {ppo.n_epochs}")
        if ppo.gamma <= 0 or ppo.gamma > 1:
            raise ValueError(f"ppo.gamma must be in (0, 1]. Got {ppo.gamma}")
        if ppo.gae_lambda <= 0 or ppo.gae_lambda > 1:
            raise ValueError(f"ppo.gae_lambda must be in (0, 1]. Got {ppo.gae_lambda}")
        if ppo.clip_range <= 0:
            raise ValueError(f"ppo.clip_range must be positive. Got {ppo.clip_range}")

    # ---- Curriculum validation ----
    if cfg.curriculum.enabled and not cfg.curriculum.stages:
        raise ValueError(
            "curriculum.enabled=true but no curriculum.stages are defined"
        )


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def load_config(path: Union[str, Path, None] = None) -> Config:
    """
    Load a config from YAML.

    Parameters
    ----------
    path
        Path to the YAML config file.  If ``None``, loads the default
        ``config/environment.yaml`` from the project root (auto-detected).

    Returns
    -------
    Config
        Populated configuration object.

    Example::

        from drl_pathplanning.gymnasium.config import load_config

        # Explicit path
        cfg = load_config("config/my_setup.yaml")

        # Default (project_root/config/environment.yaml)
        cfg = load_config()
    """
    if path is None:
        return Config.from_default()
    return Config.from_yaml(Path(path))


# --------------------------------------------------------------------------- #
# Deprecated module-level singleton
# --------------------------------------------------------------------------- #
# Old code may access ``from drl_pathplanning.gymnasium import config``
# then ``config.cfg``.  This is deprecated — use ``load_config()`` instead.
# We keep the singleton so existing code doesn't break immediately, but
# emit a deprecation warning on first access.
# --------------------------------------------------------------------------- #

class _LazyConfig:
    """Lazy-load a singleton Config on first access, then cache it."""

    _instance: Optional[Config] = None

    def __getattr__(self, name: str):
        if _LazyConfig._instance is None:
            try:
                _LazyConfig._instance = Config.from_default()
            except FileNotFoundError:
                # Provide a minimal placeholder so static analysis / type-check
                # doesn't crash.
                warnings.warn(
                    "config.cfg: default config not found. "
                    "This is a deprecated API. Use load_config() instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                _LazyConfig._instance = _minimal_placeholder()
        return getattr(_LazyConfig._instance, name)

    def __dir__(self):
        if _LazyConfig._instance is None:
            try:
                _LazyConfig._instance = Config.from_default()
            except FileNotFoundError:
                _LazyConfig._instance = _minimal_placeholder()
        return dir(_LazyConfig._instance)


def _minimal_placeholder() -> Config:
    """Build a minimal Config when the YAML file cannot be found."""
    _def_rb = RandomBoundsConfig(min=[0.30, -0.75, 0.10], max=[0.45, -0.30, 0.20])
    _def_trb = RandomBoundsConfig(min=[-0.135, -0.715, 0.10], max=[0.195, -0.385, 0.10])
    return Config(
        project=ProjectConfig(name="DRL_Pathplanning_trainning", project_type="FRAME_ONLY", unit="meter"),
        start=StartConfig(
            mode="fixed",
            fixed_position=[0.35, -0.33, 0.10],
            random_bounds=_def_rb,
            random=False,
            position=[0.35, -0.33, 0.10],
            random_start=False,
            random_space_enabled=False,
            _legacy_region_min=[0.0, 0.0, 0.0],
            _legacy_region_max=[1.0, 1.0, 1.0],
        ),
        workspace=WorkspaceConfig(
            name="search_region",
            x_min=-0.200, x_max=0.500, y_min=-0.800, y_max=0.000, z_min=0.020, z_max=0.320,
            min=[], max=[],
        ),
        target_region=TargetRegionConfig(
            mode="random",
            enabled=True,
            fixed_position=[0.030, -0.535, 0.110],
            random_bounds=_def_trb,
            random=True,
            min=[],
            max=[],
            fixed_target=[0.030, -0.535, 0.110],
            random_target=True,
            random_space_enabled=True,
        ),
        obstacle=ObstacleConfig(
            enabled=False, mode="fixed", name="small_obstacle", type="box",
            center=[0.0, 0.0, 0.0], size=[0.10, 0.10, 0.10],
            safety_margin=0.01,
            random_bounds=RandomBoundsConfig(min=[-0.100, -0.650, 0.060], max=[0.400, -0.400, 0.160]),
            random_region=RandomRegionConfig(),
            random=False,
            visual=ObstacleVisualConfig(),
            collision_visual=ObstacleCollisionVisualConfig(),
        ),
        collision=CollisionConfig(enabled=True),
        termination=TerminationConfig(goal_threshold=0.03, collision_terminate=True, workspace_terminate=True),
        table=TableConfig(
            enabled=True, name="table", type="box",
            center=[0.030, -0.550, -0.150], size=[0.330, 0.330, 0.360],
            color=[0.0, 0.0, 0.0, 1.0], collision=True,
        ),
        plane=PlaneConfig(
            enabled=True, z=-0.330,
            center=[0.150, -0.350, -0.330], size=[1.0, 1.0],
            color=[0.82, 0.82, 0.82, 0.35], collision=False,
        ),
        environment=EnvironmentStepConfig(
            observation_type="frame_only", action_step=0.01, max_episode_steps=500,
        ),
        reward=RewardConfig(
            success_bonus=10.0,
            collision_penalty=300.0,
            workspace_penalty=300.0,
            timeout_penalty=50.0,
            distance_scale=1.0,
            time_penalty=0.01,
            shake_penalty_scale=0.005,
            shake_window=10,
            shake_dot_threshold=0.0,
            shake_min_movement=1e-6,
        ),
        visualization=VisualizationConfig(
            enabled=True, gui=True, hide_debug_ui=True,
            show_workspace=True, show_target_region=True,
            show_start_frame=True, show_target_frame=True,
            show_agent_frame=True,
            show_table=True,
            show_path=True, show_ground_plane=True, show_labels=True,
            show_plane=True,
            camera=CameraConfig(distance=1.2, yaw=-60.0, pitch=-35.0, target=[0.150, -0.350, 0.170]),
            style=StyleConfig(frame_axis_length=0.05, agent_radius=0.015, path_line_width=4),
        ),
        training=TrainingConfig(
            algorithm="TD3", total_timesteps=500_000, seed=42, device="auto",
            n_envs=1, vec_env_type="auto",
            progress_bar=True, log_interval=50, episode_log_interval=1000,
            eval_freq=50000, save_freq=50000,
        ),
        evaluation=EvaluationConfig(seed=42, num_episodes=100, export_waypoints=True),
        td3=TD3Config(),
        action_noise=ActionNoiseConfig(),
        logging=LoggingConfig(),
        curriculum=CurriculumConfig(),
    )


#: Deprecated singleton accessor.  Use ``load_config()`` instead.
#: Accessed as ``from drl_pathplanning.gymnasium import config; config.cfg.environment.action_step``
#: This emits a ``DeprecationWarning`` on first access.
cfg: object = _LazyConfig()
