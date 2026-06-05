"""
pybullet — Optional PyBullet visualisation backend for the Cartesian frame environment.

This module provides a clean visualisation layer that mirrors the abstract
Gymnasium environment state (positions, targets, path history) into PyBullet.

Design principles
-----------------
- The Gymnasium env is the numerical source of truth.  All state, physics
  (frame-level), reward, and done-condition logic lives in the env.
- This module ONLY mirrors state into PyBullet for visualisation.
- No URDF, no IK/FK, no physics simulation, no robot model.
- No joint logic, no URDF loading, no MoveIt, no ROS2.

Modules
--------
frame_viewer
    PyBulletPathPlanningViewer (low-level viewer) and FrameViewer (high-level API),
    plus FrameViewerSceneSpec dataclass.
viewer_config
    Shared build_viz_config() function for constructing a viz config dict.
viewer_sync
    Shared sync_obstacle_to_viewer() and unwrap_cartesian_env() helpers for
    synchronising the PyBullet viewer with the current env state per episode.
primitives
    Colour constants and bare PyBullet drawing helpers (sphere, box, line,
    frame, text).  Consumed by frame_viewer; not needed by application code.
"""

from drl_pathplanning.pybullet.frame_viewer import (
    FixedObjectSpec,
    FrameViewer,
    FrameViewerSceneSpec,
    HAVE_PYBULLET,
    PyBulletPathPlanningViewer,
)
from drl_pathplanning.pybullet.viewer_config import build_viz_config
from drl_pathplanning.pybullet.viewer_sync import (
    sync_obstacle_to_viewer,
    unwrap_cartesian_env,
    get_current_obstacle_geometry,
    ObstacleGeometry,
)

__all__ = [
    "FixedObjectSpec",
    "FrameViewer",
    "FrameViewerSceneSpec",
    "HAVE_PYBULLET",
    "PyBulletPathPlanningViewer",
    "build_viz_config",
    "sync_obstacle_to_viewer",
    "unwrap_cartesian_env",
    "get_current_obstacle_geometry",
    "ObstacleGeometry",
]
