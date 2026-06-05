"""
Analytic geometry and collision checking for the Cartesian path planning environment.

Provides pure Python analytic primitives (no PyBullet, no physics simulation):
- AABB operations (point-in-box, segment-box intersection, workspace validation)
- GeometryCollisionChecker (workspace bounds + obstacle collision)
- Normalization helpers

Usage:
    from drl_pathplanning.geometry import (
        validate_workspace,
        sample_point_in_workspace,
        GeometryCollisionChecker,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, NamedTuple


# --------------------------------------------------------------------------- #
# AABB geometry helpers
# --------------------------------------------------------------------------- #

def validate_workspace(pos: np.ndarray, ws_min: np.ndarray, ws_max: np.ndarray) -> bool:
    """Return True if ``pos`` is inside the workspace [ws_min, ws_max]."""
    pos = np.asarray(pos, dtype=np.float32)
    ws_min = np.asarray(ws_min, dtype=np.float32)
    ws_max = np.asarray(ws_max, dtype=np.float32)
    return bool(np.all(pos >= ws_min) and np.all(pos <= ws_max))


check_point_in_workspace = validate_workspace


def check_point_in_box(
    point: np.ndarray,
    box_center: np.ndarray,
    box_half_extent: np.ndarray,
) -> bool:
    """Return True if ``point`` is inside an axis-aligned box."""
    point = np.asarray(point, dtype=np.float32)
    box_center = np.asarray(box_center, dtype=np.float32)
    box_half_extent = np.asarray(box_half_extent, dtype=np.float32)
    lb = box_center - box_half_extent
    ub = box_center + box_half_extent
    return bool(np.all(point >= lb) and np.all(point <= ub))


def _segment_axis_overlap(p0: float, p1: float, bmin: float, bmax: float) -> bool:
    """1-D slab intersection test."""
    if p0 <= bmax and p1 >= bmin:
        return True
    t0 = (bmin - p0) / (p1 - p0) if abs(p1 - p0) > 1e-12 else -1.0
    t1 = (bmax - p0) / (p1 - p0) if abs(p1 - p0) > 1e-12 else -1.0
    if t0 >= 0.0 and t0 <= 1.0:
        return True
    if t1 >= 0.0 and t1 <= 1.0:
        return True
    return False


def check_segment_intersects_box(
    p0: np.ndarray,
    p1: np.ndarray,
    box_center: np.ndarray,
    box_half_extent: np.ndarray,
) -> bool:
    """Return True if the segment from ``p0`` to ``p1`` intersects the box."""
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    box_center = np.asarray(box_center, dtype=np.float32)
    box_half_extent = np.asarray(box_half_extent, dtype=np.float32)
    lb = box_center - box_half_extent
    ub = box_center + box_half_extent
    return bool(
        _segment_axis_overlap(p0[0], p1[0], lb[0], ub[0])
        and _segment_axis_overlap(p0[1], p1[1], lb[1], ub[1])
        and _segment_axis_overlap(p0[2], p1[2], lb[2], ub[2])
    )


def sample_point_in_workspace(
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a uniformly random 3-D point inside the workspace."""
    ws_min = np.asarray(ws_min, dtype=np.float32)
    ws_max = np.asarray(ws_max, dtype=np.float32)
    return rng.uniform(ws_min, ws_max).astype(np.float32)


def workspace_bounds(min_np: np.ndarray, max_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute workspace center, half-extent, and range from min/max arrays."""
    min_np = np.asarray(min_np, dtype=np.float32)
    max_np = np.asarray(max_np, dtype=np.float32)
    center = (min_np + max_np) / 2.0
    half_extent = (max_np - min_np) / 2.0
    range_ = max_np - min_np
    return center.astype(np.float32), half_extent.astype(np.float32), range_.astype(np.float32)


def segment_length(p0: np.ndarray, p1: np.ndarray) -> float:
    """Cartesian length of a 3-D segment."""
    return float(np.linalg.norm(
        np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)))


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length. Returns zero vector if norm < 1e-6."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def cosine_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 3-D direction vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_obstacle_info(
    obstacle_center: np.ndarray,
    current_pos: np.ndarray,
    workspace_range: np.ndarray,
    obstacle_half_extent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute obstacle-related observation features."""
    obstacle_center = np.asarray(obstacle_center, dtype=np.float32)
    current_pos = np.asarray(current_pos, dtype=np.float32)
    workspace_range = np.asarray(workspace_range, dtype=np.float32)
    obstacle_half_extent = np.asarray(obstacle_half_extent, dtype=np.float32)
    rel_obs = (obstacle_center - current_pos) / workspace_range
    obs_size = obstacle_half_extent / workspace_range
    return rel_obs.astype(np.float32), obs_size.astype(np.float32)


def is_xy_near_box(
    point: np.ndarray,
    box_center: np.ndarray,
    box_half_extent: np.ndarray,
    margin: float = 0.08,
) -> bool:
    """
    Return True when the 2-D (x, y) projection of ``point`` is within
    the box xy-extents plus ``margin`` in each direction.

    Parameters
    ----------
    point
        Query point [x, y, z].
    box_center
        Center of the axis-aligned box [x, y, z].
    box_half_extent
        Half-extent (half-size) of the box along each axis [hx, hy, hz].
    margin
        Extra margin (metres) added to each xy half-extent (default: 0.08).

    Returns
    -------
    bool
        True when |px - cx| <= hx + margin  AND  |py - cy| <= hy + margin.
    """
    point = np.asarray(point, dtype=np.float32)
    box_center = np.asarray(box_center, dtype=np.float32)
    box_half_extent = np.asarray(box_half_extent, dtype=np.float32)
    xy_delta = np.abs(point[:2] - box_center[:2])
    xy_extent_with_margin = box_half_extent[:2] + margin
    return bool(np.all(xy_delta <= xy_extent_with_margin))


def distance_point_to_box(
    point: np.ndarray,
    box_center: np.ndarray,
    box_half_extent: np.ndarray,
) -> float:
    """
    Compute the shortest Euclidean distance from a 3-D point to the surface of an AABB.

    Parameters
    ----------
    point
        Query point [x, y, z].
    box_center
        Center of the axis-aligned box [x, y, z].
    box_half_extent
        Half-extent (half-size) of the box along each axis [hx, hy, hz].

    Returns
    -------
    float
        Shortest distance from the point to the box surface.
        0.0 if the point is inside or on the box surface.
    """
    point = np.asarray(point, dtype=np.float32)
    box_center = np.asarray(box_center, dtype=np.float32)
    box_half_extent = np.asarray(box_half_extent, dtype=np.float32)
    delta = np.abs(point - box_center) - box_half_extent
    outside = np.maximum(delta, 0.0)
    return float(np.linalg.norm(outside))


# --------------------------------------------------------------------------- #
# Collision result and checker
# --------------------------------------------------------------------------- #

class CollisionResult(NamedTuple):
    """Result of a collision check."""

    collides: bool
    collision_type: str  # "none" | "table_point" | "table_segment" | "box_point" | "box_segment" | "workspace"
    obstacle_name: Optional[str]  # "table" | "box" | None


class GeometryCollisionChecker:
    """
    Geometry-based collision checker for Cartesian path planning.

    Behavior is driven by explicit config flags, not string modes:
    - Workspace bounds: always checked
    - Table: always checked
    - Obstacle/box: checked only when check_box=True
    """

    def __init__(
        self,
        check_box: bool,
        obstacle_center: Optional[np.ndarray],
        obstacle_half_extent: Optional[np.ndarray],
        collision_half_extent: Optional[np.ndarray],
        obstacle_name: str = "box",
        ws_min: Optional[np.ndarray] = None,
        ws_max: Optional[np.ndarray] = None,
        table_center: Optional[np.ndarray] = None,
        table_half_extent: Optional[np.ndarray] = None,
    ) -> None:
        self._check_box = check_box
        self._obstacle_center = (
            np.asarray(obstacle_center, dtype=np.float32)
            if obstacle_center is not None else None
        )
        self._obstacle_half_extent = (
            np.asarray(obstacle_half_extent, dtype=np.float32)
            if obstacle_half_extent is not None else None
        )
        self._collision_half_extent = (
            np.asarray(collision_half_extent, dtype=np.float32)
            if collision_half_extent is not None else None
        )
        self._obstacle_name = obstacle_name
        self._ws_min = np.asarray(ws_min, dtype=np.float32) if ws_min is not None else None
        self._ws_max = np.asarray(ws_max, dtype=np.float32) if ws_max is not None else None
        self._table_center = (
            np.asarray(table_center, dtype=np.float32)
            if table_center is not None else None
        )
        self._table_half_extent = (
            np.asarray(table_half_extent, dtype=np.float32)
            if table_half_extent is not None else None
        )

    @property
    def is_collision_enabled(self) -> bool:
        return self._check_box

    def check_workspace(self, pos: np.ndarray) -> CollisionResult:
        """Check whether ``pos`` is inside the workspace bounds."""
        if self._ws_min is None or self._ws_max is None:
            return CollisionResult(False, "none", None)
        pos = np.asarray(pos, dtype=np.float32)
        if not validate_workspace(pos, self._ws_min, self._ws_max):
            return CollisionResult(True, "workspace", None)
        return CollisionResult(False, "none", None)

    def check_table(
        self, current_pos: np.ndarray, next_pos: np.ndarray
    ) -> CollisionResult:
        """Check collision with the fixed table (always collidable)."""
        if self._table_center is None or self._table_half_extent is None:
            return CollisionResult(False, "none", None)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        next_pos = np.asarray(next_pos, dtype=np.float32)
        if check_point_in_box(next_pos, self._table_center, self._table_half_extent):
            return CollisionResult(True, "table_point", "table")
        if check_segment_intersects_box(
            current_pos, next_pos, self._table_center, self._table_half_extent
        ):
            return CollisionResult(True, "table_segment", "table")
        return CollisionResult(False, "none", None)

    def check_box(
        self, current_pos: np.ndarray, next_pos: np.ndarray
    ) -> CollisionResult:
        """Check whether the segment from ``current_pos`` to ``next_pos`` collides with the box AABB."""
        if not self._check_box:
            return CollisionResult(False, "none", None)
        if self._obstacle_center is None or self._collision_half_extent is None:
            return CollisionResult(False, "none", None)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        next_pos = np.asarray(next_pos, dtype=np.float32)
        if check_point_in_box(next_pos, self._obstacle_center, self._collision_half_extent):
            return CollisionResult(True, "box_point", self._obstacle_name)
        if check_segment_intersects_box(
            current_pos, next_pos, self._obstacle_center, self._collision_half_extent
        ):
            return CollisionResult(True, "box_segment", self._obstacle_name)
        return CollisionResult(False, "none", None)

    def update_obstacle(
        self,
        obstacle_center: Optional[np.ndarray],
        collision_half_extent: Optional[np.ndarray],
    ) -> None:
        """Update obstacle geometry. Called on env reset when obstacle mode is random."""
        if obstacle_center is not None:
            self._obstacle_center = np.asarray(obstacle_center, dtype=np.float32)
        else:
            self._obstacle_center = None
        if collision_half_extent is not None:
            self._collision_half_extent = np.asarray(collision_half_extent, dtype=np.float32)
        else:
            self._collision_half_extent = None

    def check(
        self, current_pos: np.ndarray, next_pos: np.ndarray
    ) -> CollisionResult:
        """Full collision check: workspace -> table -> box. Workspace takes priority."""
        ws_result = self.check_workspace(next_pos)
        if ws_result.collides:
            return ws_result
        table_result = self.check_table(current_pos, next_pos)
        if table_result.collides:
            return table_result
        return self.check_box(current_pos, next_pos)


__all__ = [
    # geometry primitives
    "validate_workspace",
    "check_point_in_box",
    "check_point_in_workspace",
    "check_segment_intersects_box",
    "sample_point_in_workspace",
    "workspace_bounds",
    "segment_length",
    "normalize",
    "cosine_alignment",
    "normalize_obstacle_info",
    "distance_point_to_box",
    "is_xy_near_box",
    # collision
    "CollisionResult",
    "GeometryCollisionChecker",
]
