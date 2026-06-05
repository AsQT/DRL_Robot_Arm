"""
Shared visualization configuration builder for the Cartesian frame environment.

This module provides a single function — :func:`build_viz_config` — that
produces the ``viz_config`` dict consumed by :class:`FrameViewer`.  It is
used by evaluation scripts, test scripts, and future live-training viewers.

All visual defaults (colours, sizes, line widths, camera, display toggles)
live here so they need to be maintained in only one place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
#   Fallback defaults
# --------------------------------------------------------------------------- #

_DEFAULT_CAMERA_DISTANCE: float = 1.2
_DEFAULT_CAMERA_YAW: float = 45.0
_DEFAULT_CAMERA_PITCH: float = -35.0
_DEFAULT_CAMERA_TARGET: list = [0.0, -0.45, 0.1]

_DEFAULT_AGENT_RADIUS: float = 0.015
_DEFAULT_START_RADIUS: float = 0.015
_DEFAULT_TARGET_RADIUS: float = 0.015

_DEFAULT_EXPECTED_PATH_WIDTH: int = 3
_DEFAULT_ACTUAL_PATH_WIDTH: int = 5
_DEFAULT_EXPECTED_PATH_COLOR: list = [1.0, 0.0, 0.0]   # red
_DEFAULT_ACTUAL_PATH_COLOR: list = [0.0, 0.8, 1.0]     # cyan


# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #

def build_viz_config(
    cfg: Any,
    gui: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Build a visualization config dict from an environment Config object.

    The returned dict is passed to :class:`FrameViewer` or used to construct
    a :class:`FrameViewerSceneSpec`.

    Parameters
    ----------
    cfg
        An environment :class:`Config` object.  May have any of its optional
        sub-objects (visualization, obstacle, camera, style) absent.
    gui
        Open the PyBullet GUI window.  Default ``True``.
    show
        Print startup / episode results to stdout.  Default ``True``.
        (This is a hint for the caller; the dict stores it so that downstream
        code can inspect it without re-parsing CLI flags.)

    Returns
    -------
    dict
        Keys: ``gui``, all display toggles, style values (radius, colour,
        line width), and camera parameters.  Safe to pass to
        :class:`FrameViewer` or :class:`FrameViewerSceneSpec`.

    Example::

        from drl_pathplanning.gymnasium.config import Config
        from drl_pathplanning.pybullet import build_viz_config

        cfg = Config.from_yaml(Path("config/environment.yaml"))
        viz_cfg = build_viz_config(cfg, gui=True)
        viewer = FrameViewer.from_scene(FrameViewerSceneSpec(**viz_cfg))
    """
    viz_cfg: Dict[str, Any] = {
        "gui": gui,
        # Display toggles
        "show_workspace": True,
        "show_target_region": True,
        "show_path": True,
        "show_start_frame": True,
        "show_target_frame": True,
        "show_agent_frame": False,
        "show_ground_plane": False,
        "show_labels": True,
        "show_plane": False,
        "show_fixed_objects": False,
        "show_expected_path": True,
        # Style
        "expected_path_color": _DEFAULT_EXPECTED_PATH_COLOR,
        "expected_path_line_width": _DEFAULT_EXPECTED_PATH_WIDTH,
        "path_line_width": _DEFAULT_ACTUAL_PATH_WIDTH,
        "actual_path_color": _DEFAULT_ACTUAL_PATH_COLOR,
        "hide_debug_ui": True,
        "agent_radius": _DEFAULT_AGENT_RADIUS,
        "start_sphere_radius": _DEFAULT_START_RADIUS,
        "target_sphere_radius": _DEFAULT_TARGET_RADIUS,
        # Camera
        "camera_distance": _DEFAULT_CAMERA_DISTANCE,
        "camera_yaw": _DEFAULT_CAMERA_YAW,
        "camera_pitch": _DEFAULT_CAMERA_PITCH,
        "camera_target": _DEFAULT_CAMERA_TARGET,
    }

    # Obstacle visualization is controlled by obstacle.visual.enabled in the
    # obstacle config section, not by a show_box flag.
    # Obstacle box is drawn in draw_static_scene from obstacle cfg.
    try:
        viz_cfg["show_table"] = bool(cfg.table.enabled)
    except AttributeError:
        viz_cfg["show_table"] = True

    # --- Visualization section (optional in YAML) ---
    try:
        v = cfg.visualization
        if v is None:
            return viz_cfg
    except AttributeError:
        return viz_cfg

    # --- Style overrides from config ---
    try:
        style = v.style
        viz_cfg["agent_radius"] = float(style.agent_radius)
    except AttributeError:
        pass

    try:
        viz_cfg["start_sphere_radius"] = float(
            getattr(v.style, "start_sphere_radius", _DEFAULT_START_RADIUS)
        )
    except AttributeError:
        pass

    try:
        viz_cfg["target_sphere_radius"] = float(
            getattr(v.style, "target_sphere_radius", _DEFAULT_TARGET_RADIUS)
        )
    except AttributeError:
        pass

    try:
        viz_cfg["path_line_width"] = int(v.style.path_line_width)
    except AttributeError:
        pass

    # --- Display toggle overrides from config ---
    for key in (
        "show_workspace",
        "show_target_region",
        "show_path",
        "show_start_frame",
        "show_target_frame",
        "show_agent_frame",
        "show_ground_plane",
        "show_labels",
        "show_plane",
        "show_fixed_objects",
        "show_expected_path",
    ):
        try:
            viz_cfg[key] = bool(getattr(v, key, viz_cfg.get(key)))
        except AttributeError:
            pass

    # --- Camera overrides from config ---
    try:
        cam = v.camera
        viz_cfg["camera_distance"] = float(cam.distance)
    except AttributeError:
        pass

    try:
        viz_cfg["camera_yaw"] = float(v.camera.yaw)
    except AttributeError:
        pass

    try:
        viz_cfg["camera_pitch"] = float(v.camera.pitch)
    except AttributeError:
        pass

    try:
        ct = v.camera.target
        if ct is not None:
            viz_cfg["camera_target"] = (
                list(ct) if isinstance(ct, (list, np.ndarray, tuple))
                else _DEFAULT_CAMERA_TARGET
            )
    except AttributeError:
        pass

    # --- Explicit CLI overrides ---
    viz_cfg["gui"] = gui

    return viz_cfg
