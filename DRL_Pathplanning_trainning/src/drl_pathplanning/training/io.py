"""
I/O helpers for saving training artifacts.

Provides functions for saving:
  - Experiment config snapshots (JSON + YAML + run_info.json)
  - Final model checkpoints
  - Replay buffers
  - Training wall-clock time
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def save_training_snapshot(
    run_dir: Path,
    env_cfg: "drl_pathplanning.gymnasium.config.Config",
    algo_cfg: dict[str, Any],
    algo_name: str,
    device: str,
    total_timesteps: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a full experiment configuration snapshot as ``run_dir/config.json``.

    Parameters
    ----------
    run_dir
        Root run directory (created by ``create_training_run``).
    env_cfg
        Populated environment Config object.
    algo_cfg
        Parsed algorithm YAML config dict.
    algo_name
        Algorithm name (e.g. ``"DDPG"``).
    device
        Computed device string (``"cuda"`` or ``"cpu"``).
    total_timesteps
        Total training timesteps used.
    extra
        Optional additional fields to merge into the snapshot.

    Returns
    -------
    Path
        Path to the saved ``config.json`` file.
    """
    snapshot = {
        "env_config_file": None,  # filled by caller if needed
        "obstacle_enabled": env_cfg.obstacle.enabled,
        "collision_enabled": env_cfg.collision.enabled,
        "algorithm": algo_name.upper(),
        "total_timesteps": total_timesteps,
        "seed": env_cfg.training.seed,
        "run_id": None,  # filled by caller
        "device": device,
        "hyperparameters": _extract_hyperparameters(algo_cfg),
        "reward_settings": {
            "success_bonus": env_cfg.reward.success_bonus,
            "collision_penalty": env_cfg.reward.collision_penalty,
            "workspace_penalty": env_cfg.reward.workspace_penalty,
            "timeout_penalty": env_cfg.reward.timeout_penalty,
            "distance_scale": env_cfg.reward.distance_scale,
            "time_penalty": env_cfg.reward.time_penalty,
            "shake_penalty_scale": env_cfg.reward.shake_penalty_scale,
            "shake_window": env_cfg.reward.shake_window,
            "shake_dot_threshold": env_cfg.reward.shake_dot_threshold,
        },
        "max_steps": env_cfg.environment.max_episode_steps,
        "goal_threshold": env_cfg.termination.goal_threshold,
        "workspace": {
            "x_min": env_cfg.workspace.x_min,
            "x_max": env_cfg.workspace.x_max,
            "y_min": env_cfg.workspace.y_min,
            "y_max": env_cfg.workspace.y_max,
            "z_min": env_cfg.workspace.z_min,
            "z_max": env_cfg.workspace.z_max,
        },
        "table": {
            "center": env_cfg.table.center,
            "size": env_cfg.table.size,
        },
        "obstacle": {
            "enabled": env_cfg.obstacle.enabled,
            "center": env_cfg.obstacle.center,
            "size": env_cfg.obstacle.size,
        },
        "collision": {
            "enabled": env_cfg.collision.enabled,
        },
        "start": {
            "mode": env_cfg.start.mode,
            "fixed_position": env_cfg.start.fixed_position,
            "random_bounds": {
                "min": env_cfg.start.random_bounds.min,
                "max": env_cfg.start.random_bounds.max,
            },
        },
        "target_region": {
            "enabled": env_cfg.target_region.enabled,
            "mode": env_cfg.target_region.mode,
            "fixed_position": env_cfg.target_region.fixed_position,
            "random_bounds": {
                "min": env_cfg.target_region.random_bounds.min,
                "max": env_cfg.target_region.random_bounds.max,
            },
        },
    }
    if extra:
        snapshot.update(extra)

    path = run_dir / "config.json"
    with open(path, "w") as fh:
        json.dump(snapshot, fh, indent=4)
    return path


def save_final_model(
    model: "stable_baselines3.BaseAlgorithm",
    model_dir: Path,
) -> Path:
    """
    Save the final model checkpoint to ``model_dir / "final_model.zip"``.

    Parameters
    ----------
    model
        Trained SB3 model.
    model_dir
        ``run_dir / "model"`` directory.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    path = model_dir / "final_model"
    model.save(str(path))
    return Path(str(path) + ".zip")


def save_replay_buffer(
    model: "stable_baselines3.BaseAlgorithm",
    run_dir: Path,
) -> Path | None:
    """
    Save the model's replay buffer to ``run_dir / "replay_buffer.pkl"``.

    Returns ``None`` if saving fails (e.g. algorithm has no replay buffer).

    Parameters
    ----------
    model
        Trained SB3 model.
    run_dir
        Root run directory.

    Returns
    -------
    Path | None
        Path to the saved buffer, or ``None`` if saving failed.
    """
    path = run_dir / "replay_buffer.pkl"
    try:
        model.save_replay_buffer(str(path))
        return path
    except Exception:
        return None


def load_replay_buffer(
    model: "stable_baselines3.BaseAlgorithm",
    replay_buffer_path: Path,
) -> bool:
    """
    Load a saved replay buffer into the model.

    Parameters
    ----------
    model
        SB3 model (must have a replay buffer, e.g. TD3, DDPG, SAC).
    replay_buffer_path
        Path to the ``replay_buffer.pkl`` file saved by ``save_replay_buffer``.

    Returns
    -------
    bool
        ``True`` if the buffer was loaded successfully, ``False`` otherwise.
    """
    if not replay_buffer_path.exists():
        print(f"[WARN] Replay buffer not found: {replay_buffer_path}")
        return False
    try:
        model.load_replay_buffer(str(replay_buffer_path))
        return True
    except Exception as e:
        print(f"[WARN] Failed to load replay buffer: {e}")
        return False


def save_training_time(
    log_dir: Path,
    elapsed_seconds: float,
) -> Path:
    """
    Write the wall-clock training time to ``log_dir / "time.txt"``.

    Parameters
    ----------
    log_dir
        ``run_dir / "logs"`` directory.
    elapsed_seconds
        Elapsed time in seconds.

    Returns
    -------
    Path
        Path to the saved time file.
    """
    path = log_dir / "time.txt"
    path.write_text(f"Training time: {elapsed_seconds:.5f}s\n")
    return path


def print_training_summary(
    episode_rewards: list[float],
    episode_lengths: list[int],
    episode_successes: list[bool],
    episode_distances: list[float],
    elapsed: float,
    episode_expected_path_lengths: list[float] | None = None,
    episode_actual_path_lengths: list[float] | None = None,
) -> None:
    """
    Print a formatted summary of training results.

    Mirrors ``drl_pathplanning.training.logger.print_training_summary``.
    """
    import numpy as np

    n = len(episode_rewards)
    if n < 1:
        return

    early = np.mean(episode_rewards[: max(1, n // 3)])
    late = np.mean(episode_rewards[-max(1, n // 3) :])
    success_count = sum(1 for s in episode_successes if s)
    finite_dists = [d for d in episode_distances if np.isfinite(d)]

    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Episodes           : {n}")
    print(f"  Early avg reward   : {early:.3f}  (first 1/3)")
    print(f"  Late  avg reward   : {late:.3f}  (last  1/3)")
    print(f"  Improvement        : {late - early:+.3f}")
    print(f"  Success rate       : {success_count}/{n} = {success_count / n * 100:.1f}%")
    if finite_dists:
        print(
            f"  Final distance     : mean={np.mean(finite_dists):.4f}  "
            f"min={np.min(finite_dists):.4f}  max={np.max(finite_dists):.4f}"
        )

    if episode_expected_path_lengths and episode_actual_path_lengths:
        finite_s_exp = [
            x for x, s in zip(episode_expected_path_lengths, episode_successes)
            if s and np.isfinite(x)
        ]
        finite_s_act = [
            x for x, s in zip(episode_actual_path_lengths, episode_successes)
            if s and np.isfinite(x)
        ]
        if finite_s_exp and finite_s_act:
            effs = [
                exp / act * 100.0
                for exp, act in zip(finite_s_exp, finite_s_act)
                if act > 1e-8
            ]
            if effs:
                print(
                    f"  Success path eff %: mean={np.mean(effs):.2f}%  "
                    f"n={len(effs)}"
                )
            print(
                f"  Success exp path  : mean={np.mean(finite_s_exp):.4f} m  "
                f"n={len(finite_s_exp)}"
            )
            print(
                f"  Success act path  : mean={np.mean(finite_s_act):.4f} m  "
                f"n={len(finite_s_act)}"
            )

    print(f"  Training time      : {elapsed:.1f}s")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _extract_hyperparameters(algo_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat dict of hyperparameters from the algo config."""
    return {
        "gamma": algo_cfg.get("gamma"),
        "learning_rate": algo_cfg.get("learning_rate"),
        "batch_size": algo_cfg.get("batch_size"),
        "net_arch": algo_cfg.get("policy_kwargs", {}).get("net_arch"),
        "action_noise": algo_cfg.get("action_noise"),
        "total_timesteps": algo_cfg.get("total_timesteps"),
        "log_interval": algo_cfg.get("log_interval"),
    }


# --------------------------------------------------------------------------- #
# Config snapshot helpers (YAML copy + run_info.json)
# --------------------------------------------------------------------------- #

def save_config_yaml_snapshot(config_path: Path, run_dir: Path) -> Path:
    """
    Copy the raw YAML config file into the run directory as ``config.yaml``.

    Parameters
    ----------
    config_path
        Path to the YAML file used for this run.
    run_dir
        Root run directory.

    Returns
    -------
    Path
        Path to the saved ``config.yaml`` in the run directory.
    """
    import shutil
    dest = run_dir / "config.yaml"
    shutil.copy2(config_path, dest)
    return dest


def save_run_info(
    run_dir: Path,
    algorithm: str,
    mode: str,
    config_path: Path,
    checkpoint_path: Path | None = None,
    load_replay_buffer: bool = False,
    reset_num_timesteps: bool = False,
    continue_timesteps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a machine-readable ``run_info.json`` file into the run directory.

    Parameters
    ----------
    run_dir
        Root run directory.
    algorithm
        Algorithm name (TD3, DDPG, SAC, PPO).
    mode
        "from_scratch" or "continue".
    config_path
        Path to the YAML config file used for this run.
    checkpoint_path
        Path to the checkpoint loaded (only for continue mode).
    load_replay_buffer
        Whether the old replay buffer was loaded.
    reset_num_timesteps
        Whether SB3 episode counter was reset.
    continue_timesteps
        Number of additional timesteps trained (for continue mode).
    seed
        Random seed used.
    device
        Torch device used.
    extra
        Optional additional fields.

    Returns
    -------
    Path
        Path to the saved ``run_info.json`` file.
    """
    from datetime import datetime, timezone

    info: dict[str, Any] = {
        "algorithm": algorithm.upper(),
        "mode": mode,
        "config_path": str(config_path.resolve()),
        "config_yaml_snapshot": str((run_dir / "config.yaml").resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "load_replay_buffer": load_replay_buffer,
        "replay_buffer_source": None,
        "reset_num_timesteps": reset_num_timesteps,
        "continue_timesteps": continue_timesteps,
        "seed": seed,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if mode == "continue":
        info["replay_buffer_source"] = (
            str(checkpoint_path) if load_replay_buffer else "new_empty"
        )

    if extra:
        info.update(extra)

    path = run_dir / "run_info.json"
    with open(path, "w") as fh:
        json.dump(info, fh, indent=4)
    return path


def save_full_config_snapshots(
    run_dir: Path,
    env_cfg: "drl_pathplanning.gymnasium.config.Config",
    algo_cfg: dict[str, Any],
    algo_name: str,
    device: str,
    total_timesteps: int,
    config_path: Path,
    mode: str,
    checkpoint_path: Path | None = None,
    load_replay_buffer: bool = False,
    reset_num_timesteps: bool = False,
    continue_timesteps: int | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """
    Save all three config snapshots for a training run.

    Writes:
      - ``run_dir/config.yaml`` — raw YAML copy
      - ``run_dir/config.json`` — parsed config snapshot (via ``save_training_snapshot``)
      - ``run_dir/run_info.json`` — run metadata

    Parameters
    ----------
    run_dir
        Root run directory.
    env_cfg
        Populated environment Config object.
    algo_cfg
        Parsed algorithm YAML config dict.
    algo_name
        Algorithm name.
    device
        Torch device string.
    total_timesteps
        Total training timesteps.
    config_path
        Path to the YAML file used for this run.
    mode
        "from_scratch" or "continue".
    checkpoint_path
        Checkpoint path (continue mode only).
    load_replay_buffer
        Whether old replay buffer was loaded.
    reset_num_timesteps
        Whether SB3 episode counter was reset.
    continue_timesteps
        Number of additional timesteps (continue mode).
    seed
        Random seed.
    extra
        Additional fields for run_info.json.

    Returns
    -------
    tuple[Path, Path, Path]
        Paths to (config.yaml, config.json, run_info.json).
    """
    # 1. Raw YAML copy
    yaml_path = save_config_yaml_snapshot(config_path, run_dir)

    # 2. Parsed JSON snapshot (uses existing helper)
    json_path = save_training_snapshot(
        run_dir=run_dir,
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        algo_name=algo_name,
        device=device,
        total_timesteps=total_timesteps,
        extra=extra,
    )

    # 3. Run info metadata
    info_path = save_run_info(
        run_dir=run_dir,
        algorithm=algo_name,
        mode=mode,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        load_replay_buffer=load_replay_buffer,
        reset_num_timesteps=reset_num_timesteps,
        continue_timesteps=continue_timesteps,
        seed=seed if seed is not None else getattr(env_cfg.training, "seed", None),
        device=device,
        extra=extra,
    )

    return yaml_path, json_path, info_path
