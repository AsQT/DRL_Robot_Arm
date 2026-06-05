"""
Low-level PyBullet drawing primitives.

Provides colour constants and bare drawing helpers that delegate directly to the
pybullet debug drawing API.  These are consumed exclusively by PyBulletPathPlanningViewer
— application code should not need this module directly.

No robot URDF, no IK/FK, no joint control, no robot dynamics.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import pybullet as _pb

    _HAVE_PYBULLET = True
except ImportError:
    _pb = None  # type: ignore[assignment]
    _HAVE_PYBULLET = False

# Re-export so callers can check availability without importing pybullet directly.
HAVE_PYBULLET: bool = _HAVE_PYBULLET

# Threshold for debug draw failures before silencing.
_DEBUG_DRAW_FAILURE_LIMIT = 20

# --- Opaque colours ---
_COL_RED     = [1.0, 0.0, 0.0, 1.0]
_COL_GREEN   = [0.0, 1.0, 0.0, 1.0]
_COL_BLUE    = [0.0, 0.4, 1.0, 1.0]
_COL_YELLOW  = [1.0, 1.0, 0.0, 1.0]
_COL_CYAN    = [0.0, 1.0, 1.0, 1.0]
_COL_MAGENTA = [1.0, 0.0, 1.0, 1.0]
_COL_WHITE   = [1.0, 1.0, 1.0, 1.0]
_COL_BLACK   = [0.0, 0.0, 0.0, 1.0]
_COL_GRAY    = [0.5, 0.5, 0.5, 1.0]
_COL_ORANGE  = [1.0, 0.5, 0.0, 1.0]

# --- Wireframe colours ---
_COL_WORKSPACE_WIRE = [0.55, 0.55, 0.55]
_COL_TARGET_WIRE    = [1.0,  0.85, 0.0]
_COL_PATH_YELLOW    = [1.0,  1.0,  0.0]

# --- Transparent colours ---
_COL_TRANSPARENT_GREEN   = [0.0, 0.8, 0.0, 0.5]
_COL_TRANSPARENT_RED     = [1.0, 0.2, 0.2, 0.25]
_COL_TRANSPARENT_BLUE    = [0.2, 0.4, 1.0, 0.4]
_COL_TRANSPARENT_YELLOW  = [1.0, 1.0, 0.0, 0.6]
_COL_TRANSPARENT_GRAY    = [0.5, 0.5, 0.5, 0.3]
_COL_TRANSPARENT_CYAN   = [0.0, 1.0, 1.0, 0.3]

# --- Solid colours ---
_COL_GROUND       = [0.82, 0.82, 0.82, 0.35]
_COL_PLANE        = [0.82, 0.82, 0.82, 0.35]
_COL_BLACK_BLOCK  = [0.0,  0.0,  0.0,  1.0]
_COL_AGENT        = [0.15, 1.0,  0.15, 1.0]

# --- Axis colours ---
_COL_AXIS_X = [1.0, 0.0, 0.0, 1.0]
_COL_AXIS_Y = [0.0, 1.0, 0.0, 1.0]
_COL_AXIS_Z = [0.0, 0.4, 1.0, 1.0]

# --- Label colours ---
_COL_LABEL_WORKSPACE    = [0.7, 0.7, 0.7, 1.0]
_COL_LABEL_TARGET       = [1.0, 0.9, 0.3, 1.0]
_COL_LABEL_OBSTACLE    = [1.0, 0.85, 0.0, 1.0]
_COL_LABEL_AGENT       = [0.5, 1.0, 0.5, 1.0]
_COL_LABEL_START       = [1.0, 1.0, 1.0, 1.0]
_COL_LABEL_END         = [1.0, 1.0, 1.0, 1.0]
_COL_LABEL_PLANE       = [0.7, 0.7, 0.7, 1.0]
_COL_LABEL_BLACK_BLOCK  = [0.5, 0.5, 0.5, 1.0]

# --- Default palette for PyBulletPathPlanningViewer ---
_DEF_WORKSPACE_COLOR     = _COL_TRANSPARENT_GRAY
_DEF_OBSTACLE_COLOR      = _COL_TRANSPARENT_YELLOW
_DEF_AGENT_COLOR         = _COL_AGENT
_DEF_TARGET_COLOR        = _COL_TRANSPARENT_BLUE
_DEF_PATH_COLOR          = _COL_BLUE
_DEF_PLANE_COLOR         = _COL_PLANE
_DEF_BLACK_BLOCK_COLOR   = _COL_BLACK_BLOCK
_DEF_START_COLOR         = _COL_GREEN
_DEF_TARGET_SPHERE_COLOR = _COL_YELLOW


# ===========================================================================
# Colour helpers
# ===========================================================================

def _assert_rgba(color: List[float]) -> List[float]:
    """Ensure color is a 4-element RGBA list."""
    c = list(color)
    if len(c) == 3:
        c.append(1.0)
    return [float(x) for x in c[:4]]


# ===========================================================================
# Box primitives
# ===========================================================================

def draw_box_wireframe(
    client_id: int,
    center: np.ndarray,
    half_extent: np.ndarray,
    color: List[float],
    line_width: int = 1,
    gui: bool = True,
) -> List[int]:
    """Create a wireframe (line-based) AABB box."""
    pb = _pb

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    hx, hy, hz = float(half_extent[0]), float(half_extent[1]), float(half_extent[2])
    color = _assert_rgba(color)

    corners = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    item_ids: List[int] = []
    if not gui:
        return item_ids
    for i0, i1 in edges:
        uid = pb.addUserDebugLine(
            lineFromXYZ=corners[i0].tolist(),
            lineToXYZ=corners[i1].tolist(),
            lineColorRGB=color[:3],
            lineWidth=line_width,
            lifeTime=0,
            physicsClientId=client_id,
        )
        item_ids.append(uid)
    return item_ids


def draw_solid_box(
    client_id: int,
    center: np.ndarray,
    half_extent: np.ndarray,
    color: List[float],
) -> int:
    """Create an opaque visual box (mass=0, no physics)."""
    pb = _pb
    color = _assert_rgba(color)
    half_extent_list = [float(half_extent[0]), float(half_extent[1]), float(half_extent[2])]

    visual_shape_id = pb.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=half_extent_list,
        rgbaColor=color,
        visualFramePosition=[0.0, 0.0, 0.0],
        physicsClientId=client_id,
    )

    body_id = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[float(center[0]), float(center[1]), float(center[2])],
        physicsClientId=client_id,
    )
    return body_id


def draw_transparent_box(
    client_id: int,
    center: np.ndarray,
    half_extent: np.ndarray,
    color: List[float],
) -> int:
    """Create a semi-transparent visual box (mass=0, no physics)."""
    pb = _pb
    color = _assert_rgba(color)
    half_extent_list = [float(half_extent[0]), float(half_extent[1]), float(half_extent[2])]

    visual_shape_id = pb.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=half_extent_list,
        rgbaColor=color,
        visualFramePosition=[0.0, 0.0, 0.0],
        physicsClientId=client_id,
    )

    pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[float(center[0]), float(center[1]), float(center[2])],
        physicsClientId=client_id,
    )
    return visual_shape_id


def draw_obstacle_box(
    client_id: int,
    center: np.ndarray,
    half_extent: np.ndarray,
    color: List[float],
) -> int:
    """Alias for draw_transparent_box (convenience name for obstacle rendering)."""
    return draw_transparent_box(client_id, center, half_extent, color)


# ===========================================================================
# Frame / axes primitive
# ===========================================================================

def draw_frame(
    client_id: int,
    origin: np.ndarray,
    axis_length: float = 0.05,
    line_width: float = 2.5,
    label: Optional[str] = None,
    label_color: Optional[List[float]] = None,
    gui: bool = True,
) -> List[int]:
    """Create a coordinate frame at ``origin`` with coloured axis arrows."""
    pb = _pb

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    ax = float(axis_length)
    item_ids: List[int] = []

    if not gui:
        return item_ids

    lw = float(line_width)

    uid = pb.addUserDebugLine(
        lineFromXYZ=[ox, oy, oz],
        lineToXYZ=[ox + ax, oy, oz],
        lineColorRGB=_COL_AXIS_X[:3],
        lineWidth=lw, lifeTime=0, physicsClientId=client_id,
    )
    item_ids.append(uid)

    uid = pb.addUserDebugLine(
        lineFromXYZ=[ox, oy, oz],
        lineToXYZ=[ox, oy + ax, oz],
        lineColorRGB=_COL_AXIS_Y[:3],
        lineWidth=lw, lifeTime=0, physicsClientId=client_id,
    )
    item_ids.append(uid)

    uid = pb.addUserDebugLine(
        lineFromXYZ=[ox, oy, oz],
        lineToXYZ=[ox, oy, oz + ax],
        lineColorRGB=_COL_AXIS_Z[:3],
        lineWidth=lw, lifeTime=0, physicsClientId=client_id,
    )
    item_ids.append(uid)

    if label is not None:
        lc = label_color[:3] if label_color else _COL_WHITE[:3]
        uid = pb.addUserDebugText(
            text=label,
            textPosition=[ox, oy, oz + ax * 1.8],
            textColorRGB=lc,
            textSize=1.0,
            lifeTime=0,
            parentObjectUniqueId=-1,
            physicsClientId=client_id,
        )
        item_ids.append(uid)

    return item_ids


# ===========================================================================
# Point / sphere primitives
# ===========================================================================

def draw_sphere(
    client_id: int,
    position: np.ndarray,
    radius: float = 0.01,
    color: Optional[List[float]] = None,
) -> int:
    """Create a visual sphere (mass=0, no physics)."""
    pb = _pb
    color = _assert_rgba(color if color is not None else _COL_TRANSPARENT_GREEN)

    sphere_shape = pb.createVisualShape(
        shapeType=pb.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=color,
        physicsClientId=client_id,
    )

    body_id = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=sphere_shape,
        basePosition=[float(position[0]), float(position[1]), float(position[2])],
        physicsClientId=client_id,
    )
    return body_id


def move_sphere(
    client_id: int,
    body_id: int,
    position: np.ndarray,
) -> None:
    """Move an existing sphere body to a new position."""
    pb = _pb
    pb.resetBasePositionAndOrientation(
        bodyUniqueId=body_id,
        posObj=[float(position[0]), float(position[1]), float(position[2])],
        ornObj=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=client_id,
    )


# ===========================================================================
# Line primitives
# ===========================================================================

def draw_line(
    client_id: int,
    p0: np.ndarray,
    p1: np.ndarray,
    color: List[float],
    line_width: int = 1,
    gui: bool = True,
) -> int:
    """Draw a single debug line segment. Returns the debug item unique ID."""
    if not gui:
        return -1
    pb = _pb
    color = _assert_rgba(color)
    uid = pb.addUserDebugLine(
        lineFromXYZ=[float(p0[0]), float(p0[1]), float(p0[2])],
        lineToXYZ=[float(p1[0]), float(p1[1]), float(p1[2])],
        lineColorRGB=color[:3],
        lineWidth=line_width,
        lifeTime=0,
        physicsClientId=client_id,
    )
    return uid


def draw_polyline(
    client_id: int,
    points: List[np.ndarray],
    color: List[float],
    line_width: int = 4,
    gui: bool = True,
) -> List[int]:
    """Draw a polyline (connected segments) through a list of 3-D points."""
    item_ids: List[int] = []
    if not gui:
        return item_ids
    pb = _pb
    color = _assert_rgba(color)
    for i in range(len(points) - 1):
        uid = pb.addUserDebugLine(
            lineFromXYZ=[float(points[i][0]), float(points[i][1]), float(points[i][2])],
            lineToXYZ=[float(points[i + 1][0]), float(points[i + 1][1]), float(points[i + 1][2])],
            lineColorRGB=color[:3],
            lineWidth=line_width,
            lifeTime=0,
            physicsClientId=client_id,
        )
        item_ids.append(uid)
    return item_ids


# ===========================================================================
# Ground plane
# ===========================================================================

def draw_ground_plane(
    client_id: int,
    center: np.ndarray,
    size: float = 1.0,
    color: Optional[List[float]] = None,
) -> int:
    """Create a thin flat ground/table surface (mass=0, no physics)."""
    pb = _pb
    color = _assert_rgba(color if color is not None else _COL_GROUND)
    half = float(size) / 2.0

    visual_shape_id = pb.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=[half, half, 0.001],
        rgbaColor=color,
        visualFramePosition=[float(center[0]), float(center[1]), float(center[2])],
        physicsClientId=client_id,
    )

    body_id = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[float(center[0]), float(center[1]), float(center[2])],
        physicsClientId=client_id,
    )
    return body_id


# ===========================================================================
# Text label
# ===========================================================================

def draw_text(
    client_id: int,
    text: str,
    position: np.ndarray,
    color: Optional[List[float]] = None,
    text_size: float = 1.0,
    gui: bool = True,
) -> int:
    """Add a text label at a world-space position. Returns the debug item unique ID."""
    if not gui:
        return -1
    pb = _pb
    color = list(color[:3]) if color else _COL_WHITE[:3]
    try:
        uid = pb.addUserDebugText(
            text=text,
            textPosition=[float(position[0]), float(position[1]), float(position[2])],
            textColorRGB=color,
            textSize=text_size,
            lifeTime=0,
            parentObjectUniqueId=-1,
            physicsClientId=client_id,
        )
        return uid
    except Exception:
        return -1


# ===========================================================================
# Debug item removal
# ===========================================================================

def remove_debug_items(client_id: int, item_ids: List[int]) -> None:
    """Remove debug items using their unique IDs."""
    pb = _pb
    for uid in item_ids:
        try:
            pb.removeUserDebugItem(uid, physicsClientId=client_id)
        except Exception:
            pass


def hide_debug_ui(client_id: int) -> None:
    """Disable all PyBullet GUI overlays (toolbar, axis, etc.)."""
    pb = _pb
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=client_id)
    pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)
    pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=client_id)
