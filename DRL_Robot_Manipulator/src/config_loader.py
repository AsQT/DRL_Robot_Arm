"""
Project Root and Configuration Loader
===================================
Locates the project root directory by searching for ``config/config.yaml`` upward
from this file, then loads and exposes configuration values as module-level variables.

Exports:
    PROJECT_ROOT:  Path to the project root directory.
    CONFIG_FILE:   Path to the ``config.yaml`` file.
    PROJECT_FOLDER_NAME: Name string from the config (e.g. ``'DRL_Robot_Manipulator'``).
"""

from pathlib import Path
import yaml


def find_project_root() -> Path:
    """
    Walk upward from this file until ``config/config.yaml`` is found.

    The project root is identified as the directory two levels above ``config_loader.py``
    (i.e. ``config_loader.py`` is at ``<root>/src/config_loader.py``).

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If ``config/config.yaml`` is not found in any ancestor.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        check_path = parent.parent
        if (check_path / "config" / "config.yaml").exists():
            return check_path
    raise RuntimeError("config.yaml not found in any check_path directory.")


PROJECT_ROOT = find_project_root()
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Expose config values globally
PROJECT_FOLDER_NAME = config["PROJECT_FOLDER_NAME"]
