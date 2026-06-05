"""
Training utilities for SB3 algorithms.

Modules
-------
trainer          : Main training entry-point (train_sb3_model).
env_factory      : Vectorized environment factory with curriculum support.
sb3_factory      : Stable-Baselines3 model factory.
run_config       : Run directory management and config loading.
io               : I/O helpers for saving/loading training artifacts.
callbacks        : SB3 training callbacks (episode logging, safety, curriculum).
curriculum       : Curriculum target sampler for staged training.
target_provider  : Target position selection helpers.
viewer_callback  : SB3 callback for live training GUI debug.
logger           : Training and evaluation logger utilities.
paths            : Project root and directory resolution helpers.
seed             : Reproducibility seed utilities.
trajectory        : Waypoint trajectory serialization.
"""

from drl_pathplanning.training.trainer import train_sb3_model, TimestepProgressCallback

from drl_pathplanning.training.env_factory import create_training_env, EnvOutput

from drl_pathplanning.training.sb3_factory import create_sb3_model

from drl_pathplanning.training.run_config import (
    TrainingRun,
    create_training_run,
    load_algo_config,
)

from drl_pathplanning.training.io import (
    load_replay_buffer,
    save_training_snapshot,
    save_final_model,
    save_replay_buffer,
    save_training_time,
)

from drl_pathplanning.training.callbacks import (
    EpisodeCallback,
    TerminationStatsCallback,
    EarlyStopSafetyCallback,
    CurriculumGlobalStepCallback,
)

from drl_pathplanning.training.curriculum import (
    CurriculumTargetSampler,
    CurriculumStage,
    CurriculumConfig,
)

from drl_pathplanning.training.target_provider import (
    TargetProvider,
    build_target_sequence,
    get_target_region_corners,
)

from drl_pathplanning.training.logger import (
    save_config_json,
    save_time_log,
    print_training_summary,
    print_prediction_summary,
)

from drl_pathplanning.training.paths import (
    find_project_root,
    project_root,
    src_dir,
    config_dir,
    training_data_dir,
    make_run_dir,
    resolve_model,
)

from drl_pathplanning.training.seed import (
    seed_everything,
    get_numpy_generator,
)

from drl_pathplanning.training.trajectory import (
    Trajectory,
    Waypoint,
    load_waypoints_from_csv,
)

from drl_pathplanning.training.viewer_callback import create_training_viewer_callback

__all__ = [
    "train_sb3_model",
    "TimestepProgressCallback",
    "create_training_env",
    "EnvOutput",
    "create_sb3_model",
    "TrainingRun",
    "create_training_run",
    "load_algo_config",
    "load_replay_buffer",
    "save_training_snapshot",
    "save_final_model",
    "save_replay_buffer",
    "save_training_time",
    "print_training_summary",
    "EpisodeCallback",
    "TerminationStatsCallback",
    "EarlyStopSafetyCallback",
    "CurriculumGlobalStepCallback",
    "CurriculumTargetSampler",
    "CurriculumStage",
    "CurriculumConfig",
    "TargetProvider",
    "build_target_sequence",
    "get_target_region_corners",
    "save_config_json",
    "save_time_log",
    "print_prediction_summary",
    "find_project_root",
    "project_root",
    "src_dir",
    "config_dir",
    "training_data_dir",
    "make_run_dir",
    "resolve_model",
    "seed_everything",
    "get_numpy_generator",
    "Trajectory",
    "Waypoint",
    "load_waypoints_from_csv",
    "create_training_viewer_callback",
]
