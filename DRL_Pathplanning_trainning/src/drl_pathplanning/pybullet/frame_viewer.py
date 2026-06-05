"""
PyBullet visualisation backend for the Cartesian frame environment.

Provides two classes:

``PyBulletPathPlanningViewer``
    Low-level viewer that owns the PyBullet client, manages all debug shapes,
    and maps scene state to PyBullet calls.

``FrameViewer``
    High-level public API owned by the Gymnasium env.  Maps env lifecycle
    events to PyBulletPathPlanningViewer calls.

PyBullet is used ONLY for visualising abstract scene elements:
workspace box, target region, frames, obstacle, path, agent point.
No robot URDF, no IK/FK, no joint control, no robot dynamics.

Design principles
-----------------
- Static scene (workspace, plane, obstacle, target_region) is drawn ONCE
  after connect() and kept for the entire session.
- Agent sphere body is created ONCE and repositioned every step with
  resetBasePositionAndOrientation (no remove/recreate).
- Target and start markers are drawn per episode.
- All addUserDebugLine/addUserDebugText calls are guarded by a failure counter.
  After 20 failures, drawing is silently disabled and a single warning is printed.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from drl_pathplanning.pybullet.primitives import (
    HAVE_PYBULLET,
    _DEBUG_DRAW_FAILURE_LIMIT,
    _COL_WORKSPACE_WIRE,
    _COL_TARGET_WIRE,
    _COL_YELLOW,
    _COL_AXIS_X,
    _COL_AXIS_Y,
    _COL_AXIS_Z,
    _COL_WHITE,
    _COL_LABEL_WORKSPACE,
    _COL_LABEL_TARGET,
    _COL_LABEL_OBSTACLE,
    _COL_LABEL_AGENT,
    _COL_LABEL_START,
    _COL_LABEL_END,
    _COL_LABEL_PLANE,
    _COL_LABEL_BLACK_BLOCK,
    _DEF_WORKSPACE_COLOR,
    _DEF_OBSTACLE_COLOR,
    _DEF_AGENT_COLOR,
    _DEF_TARGET_COLOR,
    _DEF_PATH_COLOR,
    _DEF_PLANE_COLOR,
    _DEF_BLACK_BLOCK_COLOR,
    _DEF_START_COLOR,
    _DEF_TARGET_SPHERE_COLOR,
    draw_box_wireframe,
    draw_solid_box,
    draw_transparent_box,
    draw_frame,
    draw_sphere,
    move_sphere,
    draw_line,
    draw_polyline,
    draw_ground_plane,
    draw_text,
    remove_debug_items,
    hide_debug_ui,
)

if TYPE_CHECKING:
    import pybullet


class PyBulletPathPlanningViewer:
    """
    Optional PyBullet visualization layer for the Cartesian path planning environment.

    This viewer mirrors the abstract environment state (positions, frames, obstacle,
    path) in a PyBullet scene. It does NOT run any physics simulation that affects
    the environment — it is purely a visualisation tool.

    The viewer is created lazily (on first enable) to support headless training
    without requiring PyBullet to be installed or connected.

    Performance note: call draw_static_scene() once after connect(), then use
    clear_episode_items() + draw_target() + reset_agent() + update_agent() for
    each episode. Do NOT call reset_scene() every episode.
    """

    def __init__(
        self,
        gui: bool = True,
        cfg: Optional[object] = None,
        time_step: float = 0.01,
        show_workspace: bool = True,
        show_target_region: bool = True,
        show_start_frame: bool = True,
        show_target_frame: bool = True,
        show_agent_frame: bool = True,
        show_table: bool = True,
        show_path: bool = True,
        show_ground_plane: bool = True,
        show_labels: bool = True,
        show_plane: bool = True,
        show_fixed_objects: bool = True,
        show_start_sphere: bool = True,
        show_target_sphere: bool = True,
        show_base_link: bool = True,
        hide_debug_ui: bool = False,
        path_line_width: int = 4,
        expected_path_line_width: int = 4,
        expected_path_color: Optional[List[float]] = None,
        show_expected_path: bool = True,
        frame_axis_length: float = 0.05,
        agent_radius: float = 0.015,
        start_sphere_radius: float = 0.010,
        target_sphere_radius: float = 0.010,
        base_link_axis_length: float = 0.10,
        agent_color: Optional[List[float]] = None,
        path_color: Optional[List[float]] = None,
        camera_distance: float = 1.2,
        camera_yaw: float = -60.0,
        camera_pitch: float = -35.0,
        camera_target: Optional[List[float]] = None,
        ground_z: float = 0.0,
        ground_size: float = 1.0,
        table_center: Optional[List[float]] = None,
        table_half_extent: Optional[List[float]] = None,
        table_color: Optional[List[float]] = None,
        box_center: Optional[List[float]] = None,
        box_half_extent: Optional[List[float]] = None,
        box_color: Optional[List[float]] = None,
        debug: bool = False,
    ) -> None:
        self._gui = gui
        self._cfg = cfg
        self._time_step = float(time_step)
        self._debug = debug

        # Scene element toggles.
        self._show_workspace = show_workspace
        self._show_target_region = show_target_region
        self._show_start_frame = show_start_frame
        self._show_target_frame = show_target_frame
        self._show_agent_frame = show_agent_frame
        self._show_table = show_table
        self._show_path = show_path
        self._show_ground_plane = show_ground_plane
        self._show_labels = show_labels
        self._show_plane = show_plane
        self._show_fixed_objects = show_fixed_objects
        self._show_start_sphere = show_start_sphere
        self._show_target_sphere = show_target_sphere
        self._show_base_link = show_base_link
        self._hide_debug_ui = hide_debug_ui

        # Visual styling.
        self._path_line_width = path_line_width
        self._expected_path_line_width = expected_path_line_width
        self._expected_path_color = expected_path_color or [1.0, 0.0, 0.0]
        self._show_expected_path = show_expected_path
        self._frame_axis_length = frame_axis_length
        self._agent_radius = agent_radius
        self._start_sphere_radius = start_sphere_radius
        self._target_sphere_radius = target_sphere_radius
        self._base_link_axis_length = base_link_axis_length

        # Colours.
        self._workspace_color = _DEF_WORKSPACE_COLOR
        self._box_color = _DEF_OBSTACLE_COLOR
        self._agent_color = agent_color or _DEF_AGENT_COLOR
        self._target_color = _DEF_TARGET_COLOR
        self._path_color = path_color or _DEF_PATH_COLOR
        self._plane_color = _DEF_PLANE_COLOR
        self._black_block_color = _DEF_BLACK_BLOCK_COLOR
        self._start_color = _DEF_START_COLOR
        self._target_sphere_color = _DEF_TARGET_SPHERE_COLOR

        # Camera.
        self._cam_distance = camera_distance
        self._cam_yaw = camera_yaw
        self._cam_pitch = camera_pitch
        self._cam_target = np.array(camera_target or [0.2, -0.25, 0.25], dtype=np.float32)

        # Ground.
        self._ground_z = float(ground_z)
        self._ground_size = float(ground_size)

        # Table and box geometry (from scene spec or config).
        self._table_center = np.asarray(table_center, dtype=np.float32) if table_center is not None else None
        self._table_half_extent = np.asarray(table_half_extent, dtype=np.float32) if table_half_extent is not None else None
        self._table_color = table_color
        self._box_center = np.asarray(box_center, dtype=np.float32) if box_center is not None else None
        self._box_half_extent = np.asarray(box_half_extent, dtype=np.float32) if box_half_extent is not None else None
        self._box_color = box_color

        # Tracks the actual half_extent of the PyBullet body so we can detect size changes.
        self._box_body_half_extent: Optional[np.ndarray] = None

        # PyBullet state.
        self._client_id: int = -1
        self._pb_module: Optional["pybullet"] = None

        # Static scene elements.
        self._workspace_ids: List[int] = []
        self._target_region_ids: List[int] = []
        self._box_body_id: int = -1
        self._obstacle_body_ids: List[int] = []   # source of truth for obstacle cleanup
        self._obstacle_safety_zone_ids: List[int] = []
        self._obstacle_label_ids: List[int] = []
        self._ground_body_id: int = -1
        self._plane_body_id: int = -1
        self._table_body_id: int = -1
        self._base_link_frame_ids: List[int] = []
        self._static_label_ids: List[int] = []

        # Dynamic episode elements.
        self._agent_body_id: int = -1
        self._agent_frame_ids: List[int] = []
        self._start_frame_ids: List[int] = []
        self._start_sphere_id: int = -1
        self._target_frame_ids: List[int] = []
        self._target_sphere_id: int = -1
        self._episode_body_ids: List[int] = []  # track all dynamic body ids for cleanup
        self._path_points: List[np.ndarray] = []
        self._path_debug_ids: List[int] = []
        self._expected_path_debug_ids: List[int] = []
        self._episode_label_ids: List[int] = []

        # State snapshots.
        self._current_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._start_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._table_center: Optional[np.ndarray] = None
        self._table_half_extent: Optional[np.ndarray] = None
        self._box_center: Optional[np.ndarray] = None
        self._box_half_extent: Optional[np.ndarray] = None
        self._ws_min: np.ndarray = np.zeros(3, dtype=np.float32)
        self._ws_max: np.ndarray = np.zeros(3, dtype=np.float32)
        self._tr_min: np.ndarray = np.zeros(3, dtype=np.float32)
        self._tr_max: np.ndarray = np.zeros(3, dtype=np.float32)

        # Debug draw guard.
        self._debug_draw_failures: int = 0
        self._debug_draw_disabled: bool = False
        self._warned_about_disabled: bool = False

        # Static scene guard.
        self._static_scene_drawn: bool = False

        # Config reference.
        self._scene_cfg: Optional[object] = None

    # ------------------------------------------------------------------ #
    #   Debug draw guard
    # ------------------------------------------------------------------ #
    def _record_draw_failure(self) -> bool:
        if self._debug_draw_disabled:
            return False
        self._debug_draw_failures += 1
        if self._debug_draw_failures >= _DEBUG_DRAW_FAILURE_LIMIT:
            self._debug_draw_disabled = True
            if not self._warned_about_disabled:
                print("[WARN] PyBullet debug draw failed too many times; disabling path/frame debug drawing.")
                self._warned_about_disabled = True
            return False
        return True

    def _can_draw(self) -> bool:
        return not self._debug_draw_disabled

    # ------------------------------------------------------------------ #
    #   Connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Connect to the PyBullet client (GUI or DIRECT)."""
        import pybullet as pb

        self._pb_module = pb
        if self._gui:
            self._client_id = pb.connect(pb.GUI)
        else:
            self._client_id = pb.connect(pb.DIRECT)

        print(f"[VIEWER] gui={self._gui}  client_id={self._client_id}  connected={self._client_id >= 0}")

        if self._client_id >= 0:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id)

        if self._gui and self._hide_debug_ui:
            hide_debug_ui(self._client_id)

        pb.setGravity(0, 0, 0, physicsClientId=self._client_id)

    def disconnect(self) -> None:
        if self._client_id >= 0:
            self._pb_module.disconnect(physicsClientId=self._client_id)
            self._client_id = -1

    def _ensure_connected(self) -> "pybullet":
        if self._client_id < 0:
            self.connect()
        return self._pb_module

    # ------------------------------------------------------------------ #
    #   Static scene — drawn once after connect()
    # ------------------------------------------------------------------ #
    def setup_camera(self) -> None:
        """Configure the PyBullet debug camera."""
        pb = self._ensure_connected()
        if self._gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id)
        pb.resetDebugVisualizerCamera(
            cameraDistance=self._cam_distance,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target.tolist(),
            physicsClientId=self._client_id,
        )

    def draw_static_scene(self, cfg: object) -> None:
        """Draw all static scene elements once after connect()."""
        if self._static_scene_drawn:
            return
        self._static_scene_drawn = True
        self._scene_cfg = cfg

        pb = self._ensure_connected()

        if self._show_workspace:
            self._draw_workspace_static(cfg.workspace.min_np, cfg.workspace.max_np)

        if self._show_target_region:
            self._draw_target_region_static(cfg.target_region.min_np, cfg.target_region.max_np)

        if self._show_plane:
            plane = getattr(cfg, "plane", None)
            if plane is not None:
                self._draw_plane_static(plane.center_np, plane.size, plane.color, "PLANE")
            else:
                self._draw_plane_static(None, None, None, "PLANE")

        # Draw table if provided via from_scene (table_* params on the viewer).
        if (
            self._show_table
            and self._table_center is not None
            and self._table_half_extent is not None
        ):
            self._draw_table_static(self._table_center, self._table_half_extent)

        # Draw table from config (env-driven path via annotated_cfg).
        table_center_cfg = getattr(cfg, "_table_center", None)
        table_half_extent_cfg = getattr(cfg, "_table_half_extent", None)
        if self._show_table and table_center_cfg is not None and table_half_extent_cfg is not None:
            self._draw_table_static(table_center_cfg, table_half_extent_cfg)

        # ---- Obstacle box drawing ----
        # Source of truth (env-driven path): obstacle.enabled AND obstacle.visual.enabled
        # Box color comes from obstacle.visual.color.
        # Box geometry from _obstacle_center/_obstacle_half_extent or obstacle attrs.
        _obs_cfg = getattr(cfg, "obstacle", None)

        _obs_enabled = getattr(_obs_cfg, "enabled", False) if _obs_cfg else False
        _vis_cfg = getattr(_obs_cfg, "visual", None) if _obs_cfg else None
        _vis_enabled = getattr(_vis_cfg, "enabled", True) if _vis_cfg else True
        _cv_cfg = getattr(_obs_cfg, "collision_visual", None) if _obs_cfg else None
        _cv_enabled = getattr(_cv_cfg, "enabled", True) if _cv_cfg else True

        if _obs_cfg is not None and _obs_enabled:
            if _vis_cfg is not None and _vis_enabled:
                _box_center = getattr(cfg, "_obstacle_center", None)
                _box_half_extent = getattr(cfg, "_obstacle_half_extent", None)
                if _box_center is None:
                    _box_center = getattr(_obs_cfg, "center_np", None)
                if _box_half_extent is None:
                    _box_half_extent = getattr(_obs_cfg, "half_extent_np", None)
                _box_color = getattr(_vis_cfg, "color", self._box_color)

                if _box_center is not None and _box_half_extent is not None:
                    self._draw_box_static(
                        np.asarray(_box_center, dtype=np.float32),
                        np.asarray(_box_half_extent, dtype=np.float32),
                        color=_box_color,
                        obstacle_cfg=_obs_cfg,
                    )

        if self._show_base_link:
            self._draw_base_link_static()

        if self._gui:
            pb.resetDebugVisualizerCamera(
                cameraDistance=self._cam_distance,
                cameraYaw=self._cam_yaw,
                cameraPitch=self._cam_pitch,
                cameraTargetPosition=self._cam_target.tolist(),
                physicsClientId=self._client_id,
            )

    def _draw_workspace_static(self, ws_min: np.ndarray, ws_max: np.ndarray) -> None:
        self._ws_min = ws_min.astype(np.float32)
        self._ws_max = ws_max.astype(np.float32)
        center = (ws_min.astype(np.float32) + ws_max.astype(np.float32)) / 2.0
        half_extent = (ws_max.astype(np.float32) - ws_min.astype(np.float32)) / 2.0

        if self._show_workspace:
            self._workspace_ids = draw_box_wireframe(
                client_id=self._client_id,
                center=center, half_extent=half_extent,
                color=_COL_WORKSPACE_WIRE, line_width=2, gui=self._gui,
            )

        if self._show_labels:
            label_z = float(ws_max[2]) + 0.04
            label_pos = np.array([float(center[0]), float(center[1]), label_z], dtype=np.float32)
            uid = draw_text(self._client_id, "WORKSPACE", label_pos,
                            color=_COL_LABEL_WORKSPACE, text_size=1.0, gui=self._gui)
            self._static_label_ids.append(uid)

    def _draw_target_region_static(self, region_min: np.ndarray, region_max: np.ndarray) -> None:
        self._tr_min = region_min.astype(np.float32)
        self._tr_max = region_max.astype(np.float32)
        center = (region_min + region_max) / 2.0
        half_extent = (region_max - region_min) / 2.0

        if self._show_target_region:
            self._target_region_ids = draw_box_wireframe(
                client_id=self._client_id,
                center=center.astype(np.float32), half_extent=half_extent.astype(np.float32),
                color=_COL_TARGET_WIRE, line_width=2, gui=self._gui,
            )

        if self._show_labels:
            label_z = float(region_max[2]) + 0.04
            label_pos = np.array([float(center[0]), float(center[1]), label_z], dtype=np.float32)
            uid = draw_text(self._client_id, "TARGET REGION", label_pos,
                            color=_COL_LABEL_TARGET, text_size=1.0, gui=self._gui)
            self._static_label_ids.append(uid)

    def _draw_plane_static(
        self,
        center: Optional[np.ndarray],
        size: Optional[List[float]],
        color: Optional[List[float]],
        label: str = "PLANE",
    ) -> None:
        if center is None:
            if len(self._ws_min) == 3:
                cx = (self._ws_min[0] + self._ws_max[0]) / 2.0
                cy = (self._ws_min[1] + self._ws_max[1]) / 2.0
            else:
                cx, cy = 0.15, -0.35
            center = np.array([cx, cy, self._ground_z], dtype=np.float32)

        size_val = float(size[0]) if size is not None else self._ground_size
        rgba = color or self._plane_color

        self._plane_body_id = draw_ground_plane(
            client_id=self._client_id,
            center=center, size=size_val, color=rgba,
        )

        if self._show_labels:
            label_pos = np.array(
                [float(center[0]), float(center[1]), float(center[2]) + 0.04],
                dtype=np.float32,
            )
            uid = draw_text(self._client_id, label, label_pos,
                            color=_COL_LABEL_PLANE, text_size=1.0, gui=self._gui)
            self._static_label_ids.append(uid)

    def _draw_fixed_object_static(
        self,
        center: np.ndarray,
        half_extent: np.ndarray,
        color: Optional[List[float]],
        name: str = "FIXED_OBJECT",
    ) -> None:
        rgba = color or self._black_block_color
        body_id = draw_solid_box(
            client_id=self._client_id,
            center=center.astype(np.float32),
            half_extent=half_extent.astype(np.float32),
            color=rgba,
        )
        self._fixed_object_ids.append(body_id)

        if self._show_labels:
            label_z = float(center[2]) + float(half_extent[2]) + 0.03
            label_pos = np.array([float(center[0]), float(center[1]), label_z], dtype=np.float32)
            uid = draw_text(self._client_id, name.upper(), label_pos,
                            color=_COL_LABEL_BLACK_BLOCK, text_size=1.0, gui=self._gui)
            self._static_label_ids.append(uid)

    def _draw_box_static(
        self,
        center: np.ndarray,
        half_extent: np.ndarray,
        color: Optional[List[float]] = None,
        line_width: Optional[int] = None,
        obstacle_cfg: Optional[object] = None,
    ) -> None:
        self._box_center = np.asarray(center, dtype=np.float32)
        self._box_half_extent = np.asarray(half_extent, dtype=np.float32)

        # Drawing is controlled exclusively by obstacle_cfg.visual.enabled and
        # obstacle_cfg.collision_visual.enabled. No self._show_box dependency.
        _should_draw_main = False
        _should_draw_cv_bounds = False
        if obstacle_cfg is not None:
            _obs_enabled = getattr(obstacle_cfg, "enabled", False)
            _vis_cfg = getattr(obstacle_cfg, "visual", None)
            _vis_enabled = getattr(_vis_cfg, "enabled", True) if _vis_cfg else True
            _should_draw_main = _obs_enabled and _vis_enabled
            _cv_cfg = getattr(obstacle_cfg, "collision_visual", None)
            _cv_enabled = getattr(_cv_cfg, "enabled", True) if _cv_cfg else True
            _should_draw_cv_bounds = _obs_enabled and _cv_enabled

        if _should_draw_main:
            # Use obstacle.visual.color if available, else the passed color param.
            if obstacle_cfg is not None:
                _vis_cfg = getattr(obstacle_cfg, "visual", None)
                rgba = getattr(_vis_cfg, "color", None) if _vis_cfg else None
                if rgba is None:
                    rgba = color if color is not None else self._box_color
            else:
                rgba = color if color is not None else self._box_color
            body_id = draw_transparent_box(
                client_id=self._client_id,
                center=self._box_center,
                half_extent=self._box_half_extent,
                color=rgba,
            )
            self._box_body_id = body_id
            self._obstacle_body_ids = [body_id] if body_id >= 0 else []
            self._box_body_half_extent = np.asarray(self._box_half_extent, dtype=np.float32)

        if _should_draw_cv_bounds and obstacle_cfg is not None:
            _cv_cfg = getattr(obstacle_cfg, "collision_visual", None)
            if _cv_cfg is not None:
                cv_wireframe = getattr(_cv_cfg, "wireframe", True)

                cv_padding = getattr(_cv_cfg, "padding", [0.0, 0.0, 0.0])
                cv_half_extent = getattr(obstacle_cfg, "half_extent_np", self._box_half_extent) + np.asarray(cv_padding, dtype=np.float32)

                # RED wireframe for collision bounds — easy to see against yellow box
                _cv_rgb = [1.0, 0.3, 0.0]
                if cv_wireframe:
                    _wire_ids = draw_box_wireframe(
                        client_id=self._client_id,
                        center=self._box_center,
                        half_extent=cv_half_extent,
                        color=_cv_rgb,
                        line_width=4,
                        gui=self._gui,
                    )
                    # Save wireframe line ids so update_obstacle() can remove them later
                    for uid in _wire_ids:
                        if uid >= 0:
                            self._obstacle_safety_zone_ids.append(uid)
                else:
                    draw_transparent_box(
                        client_id=self._client_id,
                        center=self._box_center,
                        half_extent=cv_half_extent,
                        color=getattr(_cv_cfg, "color", [1.0, 0.65, 0.0, 0.25]),
                    )

        if self._show_labels and _should_draw_main:
            cz = center[2]
            hz = half_extent[2]
            label_z = float(cz) + float(hz) + 0.03
            label_pos = np.array([float(center[0]), float(center[1]), label_z], dtype=np.float32)
            size = self._box_half_extent * 2.0
            box_text = (
                f"BOX\n"
                f"center: [{float(center[0]):.3f}, {float(center[1]):.3f}, {float(center[2]):.3f}]\n"
                f"size: [{float(size[0]):.3f}, {float(size[1]):.3f}, {float(size[2]):.3f}]"
            )
            uid = draw_text(self._client_id, box_text, label_pos,
                            color=_COL_LABEL_OBSTACLE, text_size=0.8, gui=self._gui)
            # Save box label uid separately so update_obstacle() removes only obstacle labels
            if uid >= 0:
                self._obstacle_label_ids.append(uid)

    def _draw_table_static(
        self,
        center: np.ndarray,
        half_extent: np.ndarray,
        color: Optional[List[float]] = None,
    ) -> None:
        self._table_center = np.asarray(center, dtype=np.float32)
        self._table_half_extent = np.asarray(half_extent, dtype=np.float32)
        rgba = color or self._black_block_color

        body_id = draw_solid_box(
            client_id=self._client_id,
            center=self._table_center,
            half_extent=self._table_half_extent,
            color=rgba,
        )
        self._table_body_id = body_id

        if self._show_labels:
            label_z = float(center[2]) + float(half_extent[2]) + 0.03
            label_pos = np.array([float(center[0]), float(center[1]), label_z], dtype=np.float32)
            uid = draw_text(self._client_id, "TABLE", label_pos,
                            color=_COL_LABEL_BLACK_BLOCK, text_size=1.0, gui=self._gui)
            self._static_label_ids.append(uid)

    def _draw_obstacle_safety_zone_static(
        self,
        center: np.ndarray,
        collision_half_extent: np.ndarray,
    ) -> None:
        if not self._show_obstacle_safety_zone or not self._gui:
            return

        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        hx, hy, hz = float(collision_half_extent[0]), float(collision_half_extent[1]), float(collision_half_extent[2])

        corners = np.array([
            [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
        ], dtype=float)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for a, b in edges:
            uid = draw_line(self._client_id, corners[a], corners[b],
                            color=[1.0, 0.65, 0.0], line_width=2, gui=self._gui)
            if uid < 0:
                self._record_draw_failure()
            self._obstacle_safety_zone_ids.append(uid)

        if self._show_labels:
            label_pos = np.array([cx, cy, cz + hz + 0.03], dtype=np.float32)
            uid = draw_text(self._client_id, "COLLISION BOUNDS", label_pos,
                            color=[1.0, 0.65, 0.0], text_size=0.8, gui=self._gui)
            if uid < 0:
                self._record_draw_failure()
            self._obstacle_safety_zone_ids.append(uid)

    def _draw_base_link_static(self) -> None:
        if not self._show_base_link:
            return
        self._base_link_frame_ids = draw_frame(
            client_id=self._client_id,
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis_length=self._base_link_axis_length,
            line_width=2.5,
            label="base_link",
            label_color=_COL_WHITE[:3],
            gui=self._gui,
        )

    # ------------------------------------------------------------------ #
    #   Per-episode dynamic elements
    # ------------------------------------------------------------------ #
    def clear_episode_items(self) -> None:
        """Clear all dynamic per-episode items. Keeps static scene.

        Removes start/target spheres and path debug items from the previous episode.
        Does NOT touch obstacle bodies — those are managed exclusively by update_obstacle().
        Does NOT reset _box_body_id or _obstacle_body_ids.
        """
        pb = self._ensure_connected()

        remove_debug_items(self._client_id, self._path_debug_ids)
        self._path_debug_ids = []
        self._path_points = []
        remove_debug_items(self._client_id, self._expected_path_debug_ids)
        self._expected_path_debug_ids = []

        # ------------------------------------------------------------------ #
        # Remove start and target spheres only. Obstacle is NOT removed here —
        # it is handled by update_obstacle() which calls _purge_obstacle_bodies().
        # ------------------------------------------------------------------ #
        def _safe_remove(body_id: int, label: str) -> None:
            if body_id < 0:
                return
            try:
                pb.removeBody(body_id, physicsClientId=self._client_id)
            except Exception:
                pass

        _safe_remove(self._start_sphere_id, "start_sphere")
        self._start_sphere_id = -1
        _safe_remove(self._target_sphere_id, "target_sphere")
        self._target_sphere_id = -1

        remove_debug_items(self._client_id, self._obstacle_safety_zone_ids)
        self._obstacle_safety_zone_ids = []
        remove_debug_items(self._client_id, self._obstacle_label_ids)
        self._obstacle_label_ids = []
        remove_debug_items(self._client_id, self._start_frame_ids)
        self._start_frame_ids = []
        remove_debug_items(self._client_id, self._target_frame_ids)
        self._target_frame_ids = []
        remove_debug_items(self._client_id, self._agent_frame_ids)
        self._agent_frame_ids = []
        remove_debug_items(self._client_id, self._episode_label_ids)
        self._episode_label_ids = []

    # ------------------------------------------------------------------ #
    #   Agent
    # ------------------------------------------------------------------ #
    def create_agent(self, current_pos: np.ndarray) -> None:
        """Create the agent sphere body once. Call once at first episode."""
        pb = self._ensure_connected()
        self._current_pos = np.asarray(current_pos, dtype=np.float32)

        if self._agent_body_id >= 0:
            try:
                pb.removeBody(self._agent_body_id, physicsClientId=self._client_id)
            except Exception:
                pass

        self._agent_body_id = draw_sphere(
            client_id=self._client_id,
            position=self._current_pos,
            radius=self._agent_radius,
            color=self._agent_color,
        )

        if self._show_agent_frame and self._can_draw():
            self._agent_frame_ids = draw_frame(
                client_id=self._client_id,
                origin=self._current_pos,
                axis_length=self._frame_axis_length * 0.7,
                line_width=2.0,
                label="AGENT",
                label_color=_COL_LABEL_AGENT,
                gui=self._gui,
            )

    def update_agent(self, current_pos: np.ndarray) -> None:
        """Move the agent sphere to a new position (fast in-place update)."""
        self._ensure_connected()
        self._current_pos = np.asarray(current_pos, dtype=np.float32)

        if self._agent_body_id >= 0:
            move_sphere(self._client_id, self._agent_body_id, self._current_pos)

    def reset_agent(self, pos: np.ndarray) -> None:
        """Reset agent to a starting position and clear the path."""
        pb = self._ensure_connected()
        self._current_pos = np.asarray(pos, dtype=np.float32)

        if self._agent_body_id >= 0:
            try:
                move_sphere(self._client_id, self._agent_body_id, self._current_pos)
            except Exception as exc:
                if self._debug:
                    print("[Viewer] reset_agent: move_sphere failed (agent body gone): " + str(exc)
                          + " — recreating agent")
                self._agent_body_id = -1
                self.create_agent(self._current_pos)
        else:
            self.create_agent(self._current_pos)

        if self._show_agent_frame and self._can_draw():
            remove_debug_items(self._client_id, self._agent_frame_ids)
            self._agent_frame_ids = draw_frame(
                client_id=self._client_id,
                origin=self._current_pos,
                axis_length=self._frame_axis_length * 0.7,
                line_width=2.0,
                label="AGENT",
                label_color=_COL_LABEL_AGENT,
                gui=self._gui,
            )

        self.clear_path()

    # ------------------------------------------------------------------ #
    #   Start marker
    # ------------------------------------------------------------------ #
    def draw_start(self, start_pos: np.ndarray) -> None:
        """Draw the start marker: green sphere + RGB frame + 'START' label."""
        pb = self._ensure_connected()
        self._start_pos = np.asarray(start_pos, dtype=np.float32)

        if self._show_start_sphere:
            if self._start_sphere_id >= 0:
                try:
                    pb.removeBody(self._start_sphere_id, physicsClientId=self._client_id)
                except Exception:
                    pass
            self._start_sphere_id = draw_sphere(
                client_id=self._client_id,
                position=self._start_pos,
                radius=self._start_sphere_radius,
                color=self._start_color,
            )

        if self._show_start_frame and self._can_draw():
            remove_debug_items(self._client_id, self._start_frame_ids)
            self._start_frame_ids = draw_frame(
                client_id=self._client_id,
                origin=self._start_pos,
                axis_length=self._frame_axis_length,
                line_width=2.5,
                label="START",
                label_color=_COL_LABEL_START,
                gui=self._gui,
            )

    def draw_start_frame(self, start_pos: np.ndarray) -> None:
        """Alias for draw_start (backward compatibility)."""
        self.draw_start(start_pos)

    def draw_start_sphere(self, start_pos: np.ndarray) -> None:
        """Draw only the green start sphere (no frame).

        Does NOT append to _episode_body_ids — draw_start() handles that,
        and calling both draw_start and draw_start_sphere would double-create spheres.
        """
        pb = self._ensure_connected()
        self._start_pos = np.asarray(start_pos, dtype=np.float32)

        if self._show_start_sphere:
            if self._start_sphere_id >= 0:
                try:
                    pb.removeBody(self._start_sphere_id, physicsClientId=self._client_id)
                except Exception:
                    pass
            self._start_sphere_id = draw_sphere(
                client_id=self._client_id,
                position=self._start_pos,
                radius=self._start_sphere_radius,
                color=self._start_color,
            )

    # ------------------------------------------------------------------ #
    #   Target marker
    # ------------------------------------------------------------------ #
    def draw_target(
        self,
        target_pos: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        """Draw the target marker: yellow sphere + RGB frame + 'TARGET' label."""
        pb = self._ensure_connected()
        self._target_pos = np.asarray(target_pos, dtype=np.float32)
        frame_label = label if label is not None else "TARGET"

        if self._show_target_sphere:
            if self._target_sphere_id >= 0:
                try:
                    pb.removeBody(self._target_sphere_id, physicsClientId=self._client_id)
                except Exception:
                    pass
            self._target_sphere_id = draw_sphere(
                client_id=self._client_id,
                position=self._target_pos,
                radius=self._target_sphere_radius,
                color=self._target_sphere_color,
            )
            if self._target_sphere_id >= 0:
                self._episode_body_ids.append(self._target_sphere_id)

        if self._show_target_frame and self._can_draw():
            remove_debug_items(self._client_id, self._target_frame_ids)
            self._target_frame_ids = draw_frame(
                client_id=self._client_id,
                origin=self._target_pos,
                axis_length=self._frame_axis_length,
                line_width=2.5,
                label=frame_label,
                label_color=_COL_LABEL_END,
                gui=self._gui,
            )

    def draw_target_frame(self, target_pos: np.ndarray) -> None:
        """Alias for draw_target (backward compatibility)."""
        self.draw_target(target_pos)

    def draw_target_sphere(self, target_pos: np.ndarray) -> None:
        """Draw only the yellow target sphere (no frame)."""
        pb = self._ensure_connected()
        self._target_pos = np.asarray(target_pos, dtype=np.float32)

        if self._show_target_sphere:
            if self._target_sphere_id >= 0:
                try:
                    pb.removeBody(self._target_sphere_id, physicsClientId=self._client_id)
                except Exception:
                    pass
            self._target_sphere_id = draw_sphere(
                client_id=self._client_id,
                position=self._target_pos,
                radius=self._target_sphere_radius,
                color=self._target_sphere_color,
            )
            if self._target_sphere_id >= 0:
                self._episode_body_ids.append(self._target_sphere_id)

    # ------------------------------------------------------------------ #
    #   Path
    # ------------------------------------------------------------------ #
    def append_path_point(self, point: np.ndarray) -> None:
        """Add a point to the recorded path."""
        self._path_points.append(np.asarray(point, dtype=np.float32))

    def draw_path_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        color: Optional[List[float]] = None,
        line_width: Optional[int] = None,
    ) -> int:
        """Draw a single actual-path line segment."""
        if not self._show_path or not self._can_draw():
            return -1

        col = color if color is not None else self._path_color
        lw = line_width if line_width is not None else self._path_line_width

        uid = draw_line(
            client_id=self._client_id,
            p0=np.asarray(p0, dtype=np.float32),
            p1=np.asarray(p1, dtype=np.float32),
            color=col, line_width=lw, gui=self._gui,
        )
        if uid < 0:
            self._record_draw_failure()
            return -1
        self._path_debug_ids.append(uid)
        return uid

    def draw_current_segment(self) -> int:
        """Draw the most recent path segment from recorded path points."""
        if not self._show_path or not self._can_draw() or len(self._path_points) < 2:
            return -1
        p0, p1 = self._path_points[-2], self._path_points[-1]
        uid = draw_line(self._client_id, p0, p1,
                        color=self._path_color, line_width=self._path_line_width, gui=self._gui)
        if uid < 0:
            self._record_draw_failure()
            return -1
        self._path_debug_ids.append(uid)
        return uid

    def draw_full_path(self) -> List[int]:
        """Draw the complete accumulated path as a polyline."""
        if not self._show_path or not self._can_draw() or len(self._path_points) < 2:
            return []
        line_ids = draw_polyline(
            client_id=self._client_id,
            points=self._path_points,
            color=self._path_color,
            line_width=self._path_line_width,
            gui=self._gui,
        )
        for uid in line_ids:
            if uid < 0:
                self._record_draw_failure()
            else:
                self._path_debug_ids.append(uid)
        return line_ids

    def clear_path(self) -> None:
        """Remove all path line segments and reset recorded path points."""
        self._ensure_connected()
        remove_debug_items(self._client_id, self._path_debug_ids)
        self._path_debug_ids = []
        self._path_points = []

    def draw_expected_path(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        color: Optional[List[float]] = None,
        line_width: Optional[int] = None,
    ) -> int:
        """Draw the straight-line reference from start to target.

        Both endpoints are lifted by +0.002 in Z so the red reference line
        remains visible even when the blue actual path overlaps it.
        """
        if not self._show_expected_path or not self._can_draw():
            return -1

        col = color if color is not None else self._expected_path_color
        lw = line_width if line_width is not None else self._expected_path_line_width

        p0 = np.asarray(start_pos, dtype=np.float32)
        p1 = np.asarray(target_pos, dtype=np.float32)
        p0[2] += 0.002
        p1[2] += 0.002

        uid = draw_line(
            client_id=self._client_id,
            p0=p0, p1=p1,
            color=col, line_width=lw, gui=self._gui,
        )
        if uid < 0:
            self._record_draw_failure()
            return -1
        self._expected_path_debug_ids.append(uid)
        return uid

    def clear_expected_path(self) -> None:
        """Remove the expected-path reference line."""
        self._ensure_connected()
        remove_debug_items(self._client_id, self._expected_path_debug_ids)
        self._expected_path_debug_ids = []

    def set_path_color(self, color: list) -> None:
        """Set the default colour for actual path segments."""
        self._path_color = list(color)

    # ------------------------------------------------------------------ #
    #   Backward compatibility stubs
    # ------------------------------------------------------------------ #
    def reset_scene(self) -> None:
        """Backward-compatible stub — static scene drawn once via draw_static_scene()."""
        pass

    def draw_workspace(self, ws_min: np.ndarray, ws_max: np.ndarray) -> None:
        """Backward-compatible stub."""
        self._ws_min = np.asarray(ws_min, dtype=np.float32)
        self._ws_max = np.asarray(ws_max, dtype=np.float32)

    def draw_target_region(
        self,
        region_min: np.ndarray,
        region_max: np.ndarray,
    ) -> None:
        """Backward-compatible stub."""
        self._tr_min = np.asarray(region_min, dtype=np.float32)
        self._tr_max = np.asarray(region_max, dtype=np.float32)

    def draw_obstacle(
        self,
        center: np.ndarray,
        half_extent: np.ndarray,
    ) -> None:
        """Backward-compatible stub — redirects to box."""
        self._box_center = np.asarray(center, dtype=np.float32)
        self._box_half_extent = np.asarray(half_extent, dtype=np.float32)

    def update_obstacle(
        self,
        box_center: Optional[np.ndarray],
        box_half_extent: Optional[np.ndarray],
        box_color: Optional[List[float]] = None,
        obstacle_cfg: Optional[object] = None,
    ) -> None:
        """Update the obstacle box visual.

        Always recreates the obstacle body from scratch to guarantee correct size.
        """
        if box_center is None or box_half_extent is None:
            return

        pb = self._ensure_connected()
        _new_center = np.asarray(box_center, dtype=np.float32)
        _new_half_extent = np.asarray(box_half_extent, dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Step 1: Purge ALL obstacle box bodies (tracked + untracked)
        # ------------------------------------------------------------------ #
        self._purge_obstacle_bodies(reason="update_obstacle")

        # ------------------------------------------------------------------ #
        # Step 2: Remove old wireframe and labels
        # ------------------------------------------------------------------ #
        remove_debug_items(self._client_id, self._obstacle_safety_zone_ids)
        self._obstacle_safety_zone_ids = []
        remove_debug_items(self._client_id, self._obstacle_label_ids)
        self._obstacle_label_ids = []

        # ------------------------------------------------------------------ #
        # Step 3: Determine whether to draw main box and/or wireframe
        # ------------------------------------------------------------------ #
        _should_draw_main = True
        _should_draw_cv_bounds = False
        if obstacle_cfg is not None:
            _obs_enabled = getattr(obstacle_cfg, "enabled", False)
            _vis_cfg = getattr(obstacle_cfg, "visual", None)
            _vis_enabled = getattr(_vis_cfg, "enabled", True) if _vis_cfg else True
            _should_draw_main = _obs_enabled and _vis_enabled
            _cv_cfg = getattr(obstacle_cfg, "collision_visual", None)
            _cv_enabled = getattr(_cv_cfg, "enabled", True) if _cv_cfg else True
            _should_draw_cv_bounds = _obs_enabled and _cv_enabled

        # ------------------------------------------------------------------ #
        # Step 4: Create new obstacle body
        # ------------------------------------------------------------------ #
        if _should_draw_main:
            rgba = box_color or self._box_color
            _new_id = draw_transparent_box(
                client_id=self._client_id,
                center=_new_center,
                half_extent=_new_half_extent,
                color=rgba,
            )
            self._box_body_id = _new_id
            self._obstacle_body_ids = [_new_id] if _new_id >= 0 else []
            self._box_body_half_extent = np.asarray(_new_half_extent, dtype=np.float32)
            self._box_center = _new_center.copy()
            self._box_half_extent = _new_half_extent.copy()
        else:
            self._box_center = _new_center.copy()
            self._box_half_extent = _new_half_extent.copy()

        # ------------------------------------------------------------------ #
        # Step 5: Draw collision wireframe using current box_half_extent
        # ------------------------------------------------------------------ #
        if _should_draw_cv_bounds and obstacle_cfg is not None:
            _cv_cfg = getattr(obstacle_cfg, "collision_visual", None)
            if _cv_cfg is not None:
                cv_wireframe = getattr(_cv_cfg, "wireframe", True)
                cv_padding = getattr(_cv_cfg, "padding", [0.0, 0.0, 0.0])
                # Always use current self._box_half_extent (not cfg.half_extent_np)
                cv_half_extent = (
                    self._box_half_extent
                    + np.asarray(cv_padding, dtype=np.float32)
                )
                if cv_wireframe:
                    _wire_ids = draw_box_wireframe(
                        client_id=self._client_id,
                        center=self._box_center,
                        half_extent=cv_half_extent,
                        color=[1.0, 0.3, 0.0],
                        line_width=4,
                        gui=self._gui,
                    )
                    self._obstacle_safety_zone_ids.extend(
                        [uid for uid in _wire_ids if uid >= 0]
                    )

        # ------------------------------------------------------------------ #
        # Step 6: Draw label
        # ------------------------------------------------------------------ #
        if self._show_labels:
            cz = float(self._box_center[2])
            hz = float(self._box_half_extent[2])
            label_z = cz + hz + 0.03
            size = self._box_half_extent * 2.0
            box_text = (
                f"BOX\n"
                f"center: [{float(self._box_center[0]):.3f}, {float(self._box_center[1]):.3f}, {cz:.3f}]\n"
                f"size: [{float(size[0]):.3f}, {float(size[1]):.3f}, {float(size[2]):.3f}]"
            )
            uid = draw_text(
                self._client_id, box_text,
                np.array([float(self._box_center[0]), float(self._box_center[1]), label_z], dtype=np.float32),
                color=_COL_LABEL_OBSTACLE, text_size=0.8, gui=self._gui,
            )
            if uid >= 0:
                self._obstacle_label_ids.append(uid)

        # ------------------------------------------------------------------ #
        # Step 7: AABB-CHECK
        # ------------------------------------------------------------------ #
        if self._box_body_id >= 0 and self._box_body_half_extent is not None:
            try:
                _exp_half = np.asarray(self._box_half_extent, dtype=np.float32)
                _exp_center = np.asarray(self._box_center, dtype=np.float32)
                _exp_bottom_z = float(_exp_center[2]) - float(_exp_half[2])
                _actual_half_ext = np.asarray(self._box_body_half_extent, dtype=np.float32)
                _actual_bottom_z = float(_exp_center[2]) - float(_actual_half_ext[2])
                _tbl_cz = float(self._table_center[2]) if self._table_center is not None else 0.0
                _tbl_hz = float(self._table_half_extent[2]) if self._table_half_extent is not None else 0.0
                _tbl_top_z = _tbl_cz + _tbl_hz
                _gap = _actual_bottom_z - _tbl_top_z
                _half_match = np.allclose(_exp_half, _actual_half_ext, atol=1e-6)

                print(
                    f"[AABB-CHECK]  box_body_id={self._box_body_id}"
                    f"  body_half_ext=[{float(_actual_half_ext[0]):.4f},"
                    f"{float(_actual_half_ext[1]):.4f},{float(_actual_half_ext[2]):.4f}]"
                )
                print(
                    f"           exp_half=[{float(_exp_half[0]):.4f},{float(_exp_half[1]):.4f},{float(_exp_half[2]):.4f}]"
                    f"  actual_half=[{float(_actual_half_ext[0]):.4f},"
                    f"{float(_actual_half_ext[1]):.4f},{float(_actual_half_ext[2]):.4f}]"
                    f"  half_match={_half_match}"
                )
                print(
                    f"           exp_bottom_z={_exp_bottom_z:+.6f}  actual_bottom_z={_actual_bottom_z:+.6f}"
                    f"  table_top_z={_tbl_top_z:+.6f}  gap={_gap:+.6f}"
                    f"  bottom_match={abs(_actual_bottom_z - _tbl_top_z) < 1e-6}"
                )
            except Exception as _exc:
                print("[AABB-CHECK] failed: " + str(_exc))

        # ------------------------------------------------------------------ #
        # Step 8: Body audit
        # ------------------------------------------------------------------ #
        _n = pb.getNumBodies(physicsClientId=self._client_id)
        _protected_ids = {
            self._table_body_id,
            self._agent_body_id,
            self._start_sphere_id,
            self._target_sphere_id,
        }
        _box_count = 0
        for _i in range(_n):
            _bid = pb.getBodyUniqueId(_i, physicsClientId=self._client_id)
            if _bid not in _protected_ids and self._is_box_body(_bid):
                _box_count += 1

        if self._debug:
            print(f"[BODY-AUDIT] after update_obstacle  num_bodies=" + str(_n)
                  + "  table=" + str(self._table_body_id)
                  + "  agent=" + str(self._agent_body_id)
                  + "  start=" + str(self._start_sphere_id)
              + "  target=" + str(self._target_sphere_id)
              + "  box=" + str(self._box_body_id)
              + "  obstacle_ids=" + str(self._obstacle_body_ids)
              + "  box_count=" + str(_box_count))
        if _box_count > 1:
            print("[Viewer][ERROR] DUPLICATE obstacle boxes detected: " + str(_box_count))


    def draw_agent(self, current_pos: np.ndarray) -> None:
        """Backward-compatible stub."""
        if self._agent_body_id < 0:
            self.create_agent(np.asarray(current_pos, dtype=np.float32))

    # ------------------------------------------------------------------ #
    #   Step control
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """Advance the PyBullet simulation by one time step."""
        pb = self._ensure_connected()
        pb.stepSimulation(physicsClientId=self._client_id)

    def sleep_if_needed(self, real_time: bool = False) -> None:
        if real_time:
            time.sleep(self._time_step)

    @property
    def current_pos(self) -> np.ndarray:
        return self._current_pos.copy()

    def _is_box_body(self, body_id: int) -> bool:
        """Return True if the body has a GEOM_BOX visual shape."""
        pb = self._ensure_connected()
        try:
            data = pb.getVisualShapeData(body_id, physicsClientId=self._client_id)
            return any(shape[2] == pb.GEOM_BOX for shape in data)
        except Exception:
            return False

    def _purge_obstacle_bodies(self, reason: str = "") -> None:
        """Remove ALL obstacle box bodies from PyBullet world.

        Collects IDs from:
        1. Known tracked IDs: self._box_body_id, self._obstacle_body_ids
        2. Untracked box bodies: scans all bodies, removes any GEOM_BOX not in protected set

        Protected bodies (table/agent/start/target) are never removed.
        After removal, resets all obstacle tracking state.
        """
        pb = self._ensure_connected()

        _protected = {
            self._table_body_id,
            self._agent_body_id,
            self._start_sphere_id,
            self._target_sphere_id,
        }

        _ids_to_remove: set = set()

        # Known tracked obstacle IDs
        if self._box_body_id is not None and self._box_body_id >= 0:
            _ids_to_remove.add(int(self._box_body_id))
        for _bid in self._obstacle_body_ids:
            if _bid is not None and int(_bid) >= 0:
                _ids_to_remove.add(int(_bid))

        # Untracked leaked obstacle boxes: scan all bodies
        _n = pb.getNumBodies(physicsClientId=self._client_id)
        for _i in range(_n):
            _bid = pb.getBodyUniqueId(_i, physicsClientId=self._client_id)
            if _bid in _protected:
                continue
            if self._is_box_body(_bid):
                _ids_to_remove.add(int(_bid))

        _removed_count = 0
        _failed_count = 0

        for _bid in sorted(_ids_to_remove):
            if _bid in _protected:
                if self._debug:
                    print("[Viewer][ERROR] refuse to remove protected body: " + str(_bid))
                continue
            try:
                pb.removeBody(_bid, physicsClientId=self._client_id)
                _removed_count += 1
            except Exception as _exc:
                if self._debug:
                    print("[Viewer][WARN] failed to remove obstacle body " + str(_bid) + ": " + str(_exc))
                _failed_count += 1

        self._box_body_id = -1
        self._obstacle_body_ids = []
        self._box_body_half_extent = None

        if self._debug:
            print("[Viewer][_purge_obstacle_bodies] reason='" + reason
                  + "'  removed=" + str(_removed_count) + "  failed=" + str(_failed_count))

    # ------------------------------------------------------------------ #
    #   Debug / Audit
    # ------------------------------------------------------------------ #
    def debug_bodies(self, tag: str = "") -> None:
        """Print PyBullet body audit with AABB and duplicate detection."""
        if not self._debug:
            return
        try:
            pb = self._ensure_connected()
            n = pb.getNumBodies(physicsClientId=self._client_id)
            print(f"[BODY-AUDIT] {tag}  num_bodies={n}")
            print(f"  agent_body_id={self._agent_body_id}  box_body_id={self._box_body_id}  "
                  f"obstacle_body_ids={self._obstacle_body_ids}")
            print(f"  table_body_id={self._table_body_id}  target_sphere_id={self._target_sphere_id}  "
                  f"start_sphere_id={self._start_sphere_id}")
            print(f"  obstacle_line_count={len(self._obstacle_safety_zone_ids)}  "
                  f"obstacle_label_count={len(self._obstacle_label_ids)}")

            # Track obstacle candidates by position
            _obs_candidates = []
            for i in range(n):
                body_id = pb.getBodyUniqueId(i, physicsClientId=self._client_id)
                pos, _ = pb.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)
                shape = pb.getVisualShapeData(body_id, physicsClientId=self._client_id)
                n_shapes = len(shape) if shape else 0

                # Try AABB
                _aabb_min = _aabb_max = None
                try:
                    _aabb_min, _aabb_max = pb.getAABB(body_id, physicsClientId=self._client_id)
                    _aabb_str = (f"aabb=({_aabb_min[0]:.4f},{_aabb_min[1]:.4f},{_aabb_min[2]:.4f})"
                                 f"-({_aabb_max[0]:.4f},{_aabb_max[1]:.4f},{_aabb_max[2]:.4f})")
                except Exception:
                    _aabb_str = "aabb=N/A"

                print(f"    [body_id={body_id}]  pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})  "
                      f"{_aabb_str}  n_shapes={n_shapes}")

                # Collect obstacle candidates (non-table, non-agent, non-start/target)
                _is_obstacle = (
                    body_id != self._agent_body_id
                    and body_id != self._table_body_id
                    and body_id != self._start_sphere_id
                    and body_id != self._target_sphere_id
                )
                if _is_obstacle and self._box_center is not None:
                    _dist_xy = ((pos[0] - float(self._box_center[0]))**2
                                + (pos[1] - float(self._box_center[1]))**2)**0.5
                    _obs_candidates.append((body_id, pos, _dist_xy))

            # Duplicate obstacle check
            if len(_obs_candidates) > 1:
                print(f"[BODY-AUDIT][ERROR] DUPLICATE obstacle bodies detected: {len(_obs_candidates)}")
                for _cid, _cpos, _cdist in sorted(_obs_candidates, key=lambda x: x[2]):
                    print(f"         body_id={_cid}  pos=({_cpos[0]:.4f},{_cpos[1]:.4f},{_cpos[2]:.4f})  dist_xy={_cdist:.4f}")
            elif len(_obs_candidates) == 1:
                print(f"[BODY-AUDIT] OK: 1 obstacle body candidate: body_id={_obs_candidates[0][0]}")
            else:
                print(f"[BODY-AUDIT] no obstacle body candidates found")

        except Exception as exc:
            print(f"[BODY-AUDIT] {tag}  ERROR: {exc}")

    # ------------------------------------------------------------------ #
    #   Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "PyBulletPathPlanningViewer":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class FrameViewer:
    """
    High-level PyBullet visualisation backend for the Cartesian frame environment.

    This class owns the PyBullet viewer and maps env lifecycle events to viewer
    calls.  It is instantiated lazily when ``enable_visualization=True`` so that
    training runs without PyBullet installed.

    Can be used in two ways:

    1. **Standalone** (for test/eval scripts)::

           viewer = FrameViewer.from_scene(FrameViewerSceneSpec(
               workspace_min=[...], workspace_max=[...],
               target_region_min=[...], target_region_max=[...],
               gui=True,
           ))

    2. **Env-driven** (for training via CartesianPathPlanningEnv)::

           viewer = FrameViewer(render_mode="human", viz_config={...})
           viewer.initialise(env_cfg)
           # env.reset() / env.step() call viewer internally

    Environment behavior is driven by config flags (obstacle.enabled, collision.enabled, etc.),
    not by a mode string parameter.

    Parameters
    ----------
    gui
        Open a PyBullet GUI window.  ``False`` for headless/DIRECT mode.
    viz_config
        Dict of visualisation overrides matching the ``VisualizationConfig`` YAML
        structure.  Used by the env-driven path (``initialise``).
    render_mode
        Gymnasium ``render_mode`` string (``None`` or ``"human"``).
    enable_visualization
        Explicit flag to enable the viewer (overrides ``render_mode``).
    style
        Optional dict of style overrides.
    """

    def __init__(
        self,
        gui: bool = True,
        viz_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        enable_visualization: bool = False,
        style: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._gui = gui
        self._viz_config = viz_config or {}
        self._render_mode = render_mode
        self._style = style or {}

        self._enabled = bool(
            enable_visualization
            or render_mode == "human"
        ) and HAVE_PYBULLET

        self._viewer: Optional[PyBulletPathPlanningViewer] = None
        self._scene_ready = False

    # ------------------------------------------------------------------ #
    #   Public factory — for standalone scripts
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scene(
        cls,
        scene: "FrameViewerSceneSpec",
        style: Optional[Dict[str, Any]] = None,
    ) -> "FrameViewer":
        """
        Create and initialise a FrameViewer from a scene spec.

        This is the preferred constructor for test/evaluation scripts that
        do not use the Gymnasium env's ``__init__``.
        """
        effective_gui = scene.gui and HAVE_PYBULLET
        viewer = cls(gui=effective_gui, enable_visualization=effective_gui)

        if effective_gui and PyBulletPathPlanningViewer is not None:
            viewer._viewer = PyBulletPathPlanningViewer(
                gui=effective_gui,
                time_step=scene.time_step,
                show_workspace=scene.show_workspace,
                show_target_region=scene.show_target_region,
                show_start_frame=scene.show_start_frame,
                show_target_frame=scene.show_target_frame,
                show_agent_frame=scene.show_agent_frame,
                show_table=scene.show_table,
                show_path=scene.show_path,
                show_ground_plane=False,
                show_labels=scene.show_labels,
                show_plane=False,
                show_fixed_objects=scene.show_table,
                show_start_sphere=True,
                show_target_sphere=True,
                show_base_link=True,
                hide_debug_ui=scene.hide_debug_ui,
                show_expected_path=True,
                expected_path_color=scene.expected_path_color or [1.0, 0.0, 0.0],
                expected_path_line_width=scene.expected_path_width,
                path_line_width=scene.path_line_width,
                frame_axis_length=0.05,
                agent_radius=scene.agent_radius or 0.015,
                start_sphere_radius=scene.start_radius,
                target_sphere_radius=scene.target_radius,
                base_link_axis_length=0.10,
                camera_distance=scene.camera_distance,
                camera_yaw=scene.camera_yaw,
                camera_pitch=scene.camera_pitch,
                camera_target=scene.camera_target or [0.0, -0.45, 0.1],
                ground_z=-0.330,
                agent_color=scene.agent_color,
                path_color=scene.path_color,
                table_center=scene.table_center,
                table_half_extent=scene.table_half_extent,
                table_color=scene.table_color,
                box_center=scene.box_center,
                box_half_extent=scene.box_half_extent,
                box_color=scene.box_color,
            )

            viewer._viewer.connect()
            viewer._viewer.setup_camera()

            if scene.workspace_min is not None and scene.workspace_max is not None:
                viewer._viewer._draw_workspace_static(
                    np.asarray(scene.workspace_min, dtype=np.float32),
                    np.asarray(scene.workspace_max, dtype=np.float32),
                )

            if scene.target_region_min is not None and scene.target_region_max is not None:
                viewer._viewer._draw_target_region_static(
                    np.asarray(scene.target_region_min, dtype=np.float32),
                    np.asarray(scene.target_region_max, dtype=np.float32),
                )

            # Draw table (always visible unless show_table=False).
            if scene.show_table and scene.table_center is not None and scene.table_half_extent is not None:
                viewer._viewer._draw_table_static(
                    np.asarray(scene.table_center, dtype=np.float32),
                    np.asarray(scene.table_half_extent, dtype=np.float32),
                    scene.table_color,
                )

            # Draw box obstacle via _draw_box_static so collision bounds can be drawn.
            # obstacle_cfg is passed from the scene's parent config if available.
            _obs_cfg = getattr(scene, "_obstacle_cfg", None)
            if scene.box_center is not None and scene.box_half_extent is not None:
                _box_color = scene.box_color or [0.1, 0.1, 0.1, 1.0]
                viewer._viewer._draw_box_static(
                    np.asarray(scene.box_center, dtype=np.float32),
                    np.asarray(scene.box_half_extent, dtype=np.float32),
                    color=_box_color,
                    obstacle_cfg=_obs_cfg,
                )

            # Draw origin/world frame at [0, 0, 0] — always visible.
            viewer._viewer._draw_base_link_static()

            # Track initial start/target sphere bodies so clear_episode_items() can remove them
            if viewer._viewer._start_sphere_id >= 0:
                viewer._viewer._episode_body_ids.append(viewer._viewer._start_sphere_id)
            if viewer._viewer._target_sphere_id >= 0:
                viewer._viewer._episode_body_ids.append(viewer._viewer._target_sphere_id)

            viewer._scene_ready = True

        return viewer

    # ------------------------------------------------------------------ #
    #   Properties
    # ------------------------------------------------------------------ #
    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def viewer(self) -> Optional[PyBulletPathPlanningViewer]:
        return self._viewer

    # ------------------------------------------------------------------ #
    #   Initialisation — called once by the env's __init__ (env-driven path)
    # ------------------------------------------------------------------ #
    def initialise(
        self,
        cfg: "drl_pathplanning.gymnasium.config.Config",
        box_center: Optional[np.ndarray] = None,
        box_half_extent: Optional[np.ndarray] = None,
        collision_half_extent: Optional[np.ndarray] = None,
    ) -> None:
        if not self._enabled or self._viewer is not None:
            return

        vis_cfg = cfg.visualization
        vc = self._viz_config

        effective_gui = vc.get("gui", self._render_mode == "human")

        # Obstacle box is drawn in draw_static_scene using obstacle.visual.enabled.
        self._viewer = PyBulletPathPlanningViewer(
            gui=effective_gui,
            time_step=vc.get("time_step", 0.01),
            show_workspace=vc.get("show_workspace", vis_cfg.show_workspace),
            show_target_region=vc.get("show_target_region", vis_cfg.show_target_region),
            show_start_frame=vc.get("show_start_frame", vis_cfg.show_start_frame),
            show_target_frame=vc.get("show_target_frame", vis_cfg.show_target_frame),
            show_agent_frame=vc.get("show_agent_frame", vis_cfg.show_agent_frame),
            show_table=vc.get("show_table", vis_cfg.show_table),
            show_path=vc.get("show_path", vis_cfg.show_path),
            show_ground_plane=vc.get("show_ground_plane", vis_cfg.show_ground_plane),
            show_labels=vc.get("show_labels", vis_cfg.show_labels),
            show_plane=vc.get("show_plane", getattr(vis_cfg, "show_plane", True)),
            show_fixed_objects=vc.get(
                "show_table", vis_cfg.show_table
            ),
            show_start_sphere=vc.get(
                "show_start_sphere", getattr(vis_cfg, "show_start_sphere", True)
            ),
            show_target_sphere=vc.get(
                "show_target_sphere", getattr(vis_cfg, "show_target_sphere", True)
            ),
            show_base_link=False,
            hide_debug_ui=vc.get("hide_debug_ui", True),
            show_expected_path=vc.get("show_expected_path", True),
            expected_path_color=vc.get("expected_path_color", [1.0, 0.0, 0.0]),
            expected_path_line_width=vc.get("expected_path_line_width", 3),
            path_line_width=vc.get("path_line_width", 5),
            path_color=vc.get("actual_path_color"),
            frame_axis_length=vc.get("frame_axis_length", 0.05),
            agent_radius=vc.get("agent_radius", vis_cfg.style.agent_radius),
            start_sphere_radius=vc.get(
                "start_sphere_radius", getattr(vis_cfg.style, "start_sphere_radius", 0.015)
            ),
            target_sphere_radius=vc.get(
                "target_sphere_radius", getattr(vis_cfg.style, "target_sphere_radius", 0.015)
            ),
            base_link_axis_length=vc.get(
                "base_link_axis_length", getattr(vis_cfg.style, "base_link_axis_length", 0.10)
            ),
            camera_distance=vc.get("camera_distance", vis_cfg.camera.distance),
            camera_yaw=vc.get("camera_yaw", vis_cfg.camera.yaw),
            camera_pitch=vc.get("camera_pitch", vis_cfg.camera.pitch),
            camera_target=vc.get("camera_target", vis_cfg.camera.target),
            ground_z=vc.get(
                "ground_z",
                getattr(cfg, "plane", None) and cfg.plane.z or -0.330,
            ),
        )

        self._viewer.connect()
        self._viewer.setup_camera()

        annotated_cfg = cfg
        if cfg.obstacle.enabled and box_center is not None:
            annotated_cfg._obstacle_center = box_center
            annotated_cfg._obstacle_half_extent = box_half_extent
            annotated_cfg._collision_half_extent = collision_half_extent

        self._viewer.draw_static_scene(annotated_cfg)
        # Track initial start/target sphere bodies so clear_episode_items() can remove them
        if self._viewer._start_sphere_id >= 0:
            self._viewer._episode_body_ids.append(self._viewer._start_sphere_id)
        if self._viewer._target_sphere_id >= 0:
            self._viewer._episode_body_ids.append(self._viewer._target_sphere_id)
        self._scene_ready = True

    # ------------------------------------------------------------------ #
    #   Per-episode
    # ------------------------------------------------------------------ #
    def reset_episode(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> None:
        """Set up per-episode visualisation elements."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.clear_episode_items()
        self._viewer.reset_agent(np.asarray(start_pos, dtype=np.float32))
        self._viewer.draw_start(np.asarray(start_pos, dtype=np.float32))
        self._viewer.draw_target(np.asarray(target_pos, dtype=np.float32))

    def update_obstacle(
        self,
        box_center: Optional[np.ndarray],
        box_half_extent: Optional[np.ndarray],
        box_color: Optional[List[float]] = None,
        obstacle_cfg: Optional[object] = None,
    ) -> None:
        """Update the obstacle box visual to a new center.

        Called per-episode when obstacle.mode == 'random' to reposition the
        yellow box and collision wireframe.
        """
        if not self._enabled or self._viewer is None:
            return
        self._viewer.update_obstacle(
            box_center=box_center,
            box_half_extent=box_half_extent,
            box_color=box_color,
            obstacle_cfg=obstacle_cfg,
        )

    # ------------------------------------------------------------------ #
    #   Per-step
    # ------------------------------------------------------------------ #
    def update_agent(self, current_pos: np.ndarray) -> None:
        """Reposition the agent sphere to match the env's current position."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.update_agent(np.asarray(current_pos, dtype=np.float32))

    update_point = update_agent

    def step(self) -> None:
        """Advance the PyBullet simulation one step."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.step()

    def draw_path_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        color: Optional[List[float]] = None,
        line_width: Optional[int] = None,
    ) -> None:
        """Draw a single actual-path segment between two positions."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.draw_path_segment(
            np.asarray(p0, dtype=np.float32),
            np.asarray(p1, dtype=np.float32),
            color=color,
            line_width=line_width,
        )

    # ------------------------------------------------------------------ #
    #   Reference / expected path
    # ------------------------------------------------------------------ #
    def draw_expected_path(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        color: Optional[List[float]] = None,
        line_width: Optional[int] = None,
    ) -> None:
        """Draw the straight-line reference from start to target."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.draw_expected_path(
            np.asarray(start_pos, dtype=np.float32),
            np.asarray(target_pos, dtype=np.float32),
            color=color,
            line_width=line_width,
        )

    def clear_expected_path(self) -> None:
        """Remove the expected-path reference line."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.clear_expected_path()

    # ------------------------------------------------------------------ #
    #   Path management
    # ------------------------------------------------------------------ #
    def clear_path(self) -> None:
        """Clear all recorded actual-path segments."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.clear_path()

    def set_path_color(self, color: List[float]) -> None:
        """Set the default colour for subsequent path segments."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.set_path_color(list(color))

    # ------------------------------------------------------------------ #
    #   Direct marker helpers
    # ------------------------------------------------------------------ #
    def draw_start(self, start_pos: np.ndarray) -> None:
        """Draw the start marker: green sphere + RGB frame + 'START' label."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.draw_start(np.asarray(start_pos, dtype=np.float32))

    def draw_target(
        self,
        target_pos: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        """Draw the target marker: yellow sphere + RGB frame + 'TARGET' label."""
        if not self._enabled or self._viewer is None:
            return
        self._viewer.draw_target(
            np.asarray(target_pos, dtype=np.float32), label=label
        )

    # ------------------------------------------------------------------ #
    #   Cleanup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Disconnect the PyBullet client and release resources."""
        if self._viewer is not None:
            try:
                self._viewer.disconnect()
            except Exception:
                pass
            self._viewer = None
        self._scene_ready = False


@dataclasses.dataclass
class FixedObjectSpec:
    """
    A fixed visual object in the scene (e.g. a solid table block).

    Distinct from the RL box obstacle — fixed objects are part of the static scene
    and are shown by default regardless of box settings.
    """
    center: np.ndarray
    half_extent: np.ndarray
    color: Optional[List[float]] = None
    name: str = "FIXED_OBJECT"


# Back-compat alias — FixedObjectSpec was the previous name.
TableSpec = FixedObjectSpec


@dataclasses.dataclass
class FrameViewerSceneSpec:
    """
    Parameters for setting up the static PyBullet scene (drawn once).

    All fields are optional — defaults are applied when not specified.

    Naming convention:
    - ``table_*``   : fixed black table (always visible, always collidable).
    - ``box_*``     : optional box obstacle (hidden by default).
    """

    workspace_min: Optional[List[float]] = None
    workspace_max: Optional[List[float]] = None
    target_region_min: Optional[List[float]] = None
    target_region_max: Optional[List[float]] = None

    # Table (fixed black block).
    table_center: Optional[List[float]] = None
    table_half_extent: Optional[List[float]] = None
    table_color: Optional[List[float]] = None

    # Box (optional obstacle).
    box_center: Optional[List[float]] = None
    box_half_extent: Optional[List[float]] = None
    box_color: Optional[List[float]] = None
    # Pass full obstacle config so _draw_box_static can render collision bounds
    _obstacle_cfg: Optional[object] = None

    gui: bool = True
    time_step: float = 0.01

    # Visual styling
    agent_color: Optional[List[float]] = None
    expected_path_color: Optional[List[float]] = None
    path_color: Optional[List[float]] = None  # actual agent path color (default: blue)
    agent_radius: Optional[float] = None
    expected_path_width: int = 3
    path_line_width: int = 5  # actual agent path line width
    start_radius: float = 0.015
    target_radius: float = 0.015

    # Display toggles
    show_workspace: bool = True
    show_target_region: bool = True
    show_table: bool = True
    show_path: bool = True
    show_labels: bool = True
    show_start_frame: bool = True
    show_target_frame: bool = True
    show_agent_frame: bool = False
    hide_debug_ui: bool = True

    # Camera
    camera_distance: float = 1.2
    camera_yaw: float = 45.0
    camera_pitch: float = -35.0
    camera_target: Optional[List[float]] = None
