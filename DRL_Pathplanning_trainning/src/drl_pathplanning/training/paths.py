"""
Path utilities for DRL_Pathplanning_trainning.

Provides helpers for resolving the project root, output directories, and
model checkpoints regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward from ``start`` (defaults to this file's directory) until a
    directory containing both ``src/`` and ``Training/`` is found.

    Parameters
    ----------
    start
        Starting directory for the search.  Defaults to this file's parent.

    Returns
    -------
    Path
        The project root directory.

    Raises
    ------
    RuntimeError
        If no project root can be located.
    """
    if start is None:
        start = Path(__file__).resolve().parent.parent.parent
    else:
        start = Path(start).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "Training").exists():
            return candidate

    raise RuntimeError(
        f"[PATHS] Could not locate project root from {start}.\n"
        "        Expected to find 'src/' and 'Training/' directories."
    )


# --------------------------------------------------------------------------- #
# Directory helpers
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Return the project root directory (singleton, cached on first call)."""
    return find_project_root()


def src_dir() -> Path:
    """Return the src/ directory."""
    return project_root() / "src"


def config_dir() -> Path:
    """Return the config/ directory."""
    return project_root() / "config"


def training_data_dir(env_mode: str, algorithm: str) -> Path:
    """
    Return the base training data directory for a given environment mode
    and algorithm.

    Data/Training/Environment_{env_mode}/{algorithm}/FRAME_ONLY/
    """
    return (
        project_root()
        / "Data"
        / "Training"
        / f"Environment_{env_mode}"
        / algorithm
        / "FRAME_ONLY"
    )


def make_run_dir(env_mode: str, algorithm: str, run_id: str) -> Path:
    """
    Create and return a timestamped run directory.

    Data/Training/Environment_{env_mode}/{algorithm}/FRAME_ONLY/run_{run_id}/
    """
    run_dir = training_data_dir(env_mode, algorithm) / f"run_{run_id}"
    for sub in ("model", "logs", "tensorboard", "trajectory"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_model(model_path: Optional[Path | str], run_dir: Optional[Path | str]) -> Path:
    """
    Resolve a model .zip path from either a direct path or a run directory.

    Search order within a run directory:
        1. run_dir / "model" / "final_model.zip"
        2. run_dir / "final_model.zip"
        3. run_dir / "model" / "best_model.zip"
        4. run_dir / "best_model.zip"

    Parameters
    ----------
    model_path
        Direct path to a .zip file.
    run_dir
        Path to a run directory (model will be auto-detected inside it).

    Returns
    -------
    Path
        Resolved path to the model .zip file.

    Raises
    ------
    FileNotFoundError
        If neither source is valid.
    """
    root = project_root()

    if model_path is not None:
        p = Path(model_path)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"[PATHS] Model file not found: {p}")

    if run_dir is not None:
        rd = Path(run_dir)
        if not rd.is_absolute():
            rd = root / rd
        if not rd.exists():
            raise FileNotFoundError(f"[PATHS] Run directory not found: {rd}")

        candidates = [
            rd / "model" / "final_model.zip",
            rd / "final_model.zip",
            rd / "model" / "best_model.zip",
            rd / "best_model.zip",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError(
            f"[PATHS] No model found in {rd}.\n"
            "        Expected: final_model.zip or best_model.zip "
            "in run_dir/ or run_dir/model/."
        )

    raise FileNotFoundError(
        "[PATHS] No model source configured. "
        "Pass model_path or run_dir."
    )
