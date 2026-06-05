"""
Run configuration and directory management for training experiments.

Provides:
  - TrainingRun dataclass holding all paths for a single training run
  - create_training_run() factory that creates timestamped run directories
  - load_algo_config() for loading algorithm YAML configs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingRun:
    """
    Holds all directory paths for a single training run.

    Attributes
    ----------
    run_id
        Human-readable timestamp string (e.g. ``20260517_161800``).
    run_dir
        Root run directory.
    model_dir
        ``run_dir / "model"`` — saved policy checkpoints.
    log_dir
        ``run_dir / "logs"`` — SB3 logs, monitor CSV, time log.
    tensorboard_dir
        ``run_dir / "tensorboard"`` — TensorBoard event files.
    trajectory_dir
        ``run_dir / "trajectory"`` — exported waypoint trajectories.
    algorithm
        Algorithm name (e.g. ``DDPG``, ``SAC``, ``TD3``).
    algo_name
        Lower-case algorithm name for directory naming.
    """
    run_id: str
    run_dir: Path
    model_dir: Path
    log_dir: Path
    tensorboard_dir: Path
    trajectory_dir: Path
    algorithm: str = "DDPG"

    @property
    def algo_name(self) -> str:
        """Lower-case algorithm name for directory paths."""
        return self.algorithm.upper()


def create_training_run(
    algorithm: str,
    base_dir: Path | None = None,
) -> TrainingRun:
    """
    Create and return a timestamped training run directory.

    The directory structure follows the existing convention::

        Data/Training/{algorithm}/FRAME_ONLY/run_{run_id}/

    Environment behavior is controlled by config flags, not by a mode string.
    The directory name embeds obstacle/collision settings derived from the
    config (e.g. ``obs_on_col_on`` or ``obs_off_col_off``).

    Parameters
    ----------
    algorithm
        Algorithm name (e.g. ``"DDPG"``, ``"SAC"``, ``"TD3"``).
    base_dir
        Override base directory.  Defaults to ``Data/Training`` relative to
        the project root.

    Returns
    -------
    TrainingRun
        Populated run object with all paths created as empty directories.
    """
    if base_dir is None:
        base_dir = _find_project_root() / "Data" / "Training"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        base_dir
        / algorithm.upper()
        / "FRAME_ONLY"
        / f"run_{run_id}"
    )

    model_dir = run_dir / "model"
    log_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    trajectory_dir = run_dir / "trajectory"

    for d in [model_dir, log_dir, tensorboard_dir, trajectory_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return TrainingRun(
        run_id=run_id,
        run_dir=run_dir,
        model_dir=model_dir,
        log_dir=log_dir,
        tensorboard_dir=tensorboard_dir,
        trajectory_dir=trajectory_dir,
        algorithm=algorithm.upper(),
    )


def load_algo_config(path: Path | str) -> dict[str, Any]:
    """
    Load an algorithm experiment YAML config.

    Parameters
    ----------
    path
        Path to the algorithm YAML file (e.g. ``config/experiments/ddpg_default.yaml``).

    Returns
    -------
    dict
        Parsed YAML dict of algorithm hyperparameters.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[CONFIG] Algorithm config not found: {p}")
    with open(p) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _find_project_root() -> Path:
    """Walk upward from this file to find the project root."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "src").exists() and (current / "Training").exists():
            return current
        current = current.parent
    # Fallback: use a known path relative to this file
    # src/drl_pathplanning/training/run_config.py -> project root
    return Path(__file__).resolve().parents[3]
