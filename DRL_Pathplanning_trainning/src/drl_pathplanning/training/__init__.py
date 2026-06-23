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
trajectory       : Waypoint trajectory serialization.
"""

from importlib import import_module


_EXPORTS = {
    "train_sb3_model": ("drl_pathplanning.training.trainer", "train_sb3_model"),
    "TimestepProgressCallback": (
        "drl_pathplanning.training.trainer",
        "TimestepProgressCallback",
    ),
    "create_training_env": ("drl_pathplanning.training.env_factory", "create_training_env"),
    "EnvOutput": ("drl_pathplanning.training.env_factory", "EnvOutput"),
    "create_sb3_model": ("drl_pathplanning.training.sb3_factory", "create_sb3_model"),
    "TrainingRun": ("drl_pathplanning.training.run_config", "TrainingRun"),
    "create_training_run": ("drl_pathplanning.training.run_config", "create_training_run"),
    "load_algo_config": ("drl_pathplanning.training.run_config", "load_algo_config"),
    "load_replay_buffer": ("drl_pathplanning.training.io", "load_replay_buffer"),
    "save_training_snapshot": ("drl_pathplanning.training.io", "save_training_snapshot"),
    "save_final_model": ("drl_pathplanning.training.io", "save_final_model"),
    "save_replay_buffer": ("drl_pathplanning.training.io", "save_replay_buffer"),
    "save_training_time": ("drl_pathplanning.training.io", "save_training_time"),
    "print_training_summary": ("drl_pathplanning.training.logger", "print_training_summary"),
    "EpisodeCallback": ("drl_pathplanning.training.callbacks", "EpisodeCallback"),
    "TerminationStatsCallback": (
        "drl_pathplanning.training.callbacks",
        "TerminationStatsCallback",
    ),
    "EarlyStopSafetyCallback": (
        "drl_pathplanning.training.callbacks",
        "EarlyStopSafetyCallback",
    ),
    "CurriculumGlobalStepCallback": (
        "drl_pathplanning.training.callbacks",
        "CurriculumGlobalStepCallback",
    ),
    "CurriculumTargetSampler": (
        "drl_pathplanning.training.curriculum",
        "CurriculumTargetSampler",
    ),
    "CurriculumStage": ("drl_pathplanning.training.curriculum", "CurriculumStage"),
    "CurriculumConfig": ("drl_pathplanning.training.curriculum", "CurriculumConfig"),
    "TargetProvider": ("drl_pathplanning.training.target_provider", "TargetProvider"),
    "build_target_sequence": (
        "drl_pathplanning.training.target_provider",
        "build_target_sequence",
    ),
    "get_target_region_corners": (
        "drl_pathplanning.training.target_provider",
        "get_target_region_corners",
    ),
    "save_config_json": ("drl_pathplanning.training.logger", "save_config_json"),
    "save_time_log": ("drl_pathplanning.training.logger", "save_time_log"),
    "print_prediction_summary": (
        "drl_pathplanning.training.logger",
        "print_prediction_summary",
    ),
    "find_project_root": ("drl_pathplanning.training.paths", "find_project_root"),
    "project_root": ("drl_pathplanning.training.paths", "project_root"),
    "src_dir": ("drl_pathplanning.training.paths", "src_dir"),
    "config_dir": ("drl_pathplanning.training.paths", "config_dir"),
    "training_data_dir": ("drl_pathplanning.training.paths", "training_data_dir"),
    "make_run_dir": ("drl_pathplanning.training.paths", "make_run_dir"),
    "resolve_model": ("drl_pathplanning.training.paths", "resolve_model"),
    "seed_everything": ("drl_pathplanning.training.seed", "seed_everything"),
    "get_numpy_generator": ("drl_pathplanning.training.seed", "get_numpy_generator"),
    "Trajectory": ("drl_pathplanning.training.trajectory", "Trajectory"),
    "Waypoint": ("drl_pathplanning.training.trajectory", "Waypoint"),
    "load_waypoints_from_csv": (
        "drl_pathplanning.training.trajectory",
        "load_waypoints_from_csv",
    ),
    "create_training_viewer_callback": (
        "drl_pathplanning.training.viewer_callback",
        "create_training_viewer_callback",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import public training utilities only when they are requested."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
