"""
Geometry module for the Cartesian path planning environment.

Provides pure Python analytic geometry functions:
- AABB operations (point-in-box, segment-box intersection, workspace validation)
- GeometryCollisionChecker (workspace bounds + obstacle collision)
- Path quality metrics (path_length, path_efficiency, cosine_smoothness, progress_ratio)
- Normalization helpers

No Gymnasium, no PyBullet, no training code.

Usage:
    from drl_pathplanning.geometry import (
        validate_workspace,
        sample_point_in_workspace,
        GeometryCollisionChecker,
        path_length,
    )
"""

from drl_pathplanning.geometry.collision_geometry import (
    validate_workspace,
    check_point_in_box,
    check_point_in_workspace,
    check_segment_intersects_box,
    sample_point_in_workspace,
    workspace_bounds,
    segment_length,
    normalize,
    cosine_alignment,
    normalize_obstacle_info,
    distance_point_to_box,
    is_xy_near_box,
    CollisionResult,
    GeometryCollisionChecker,
)

from drl_pathplanning.geometry.path_metrics import (
    path_length,
    path_efficiency,
    cosine_smoothness,
    progress_ratio,
)

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
    # path metrics
    "path_length",
    "path_efficiency",
    "cosine_smoothness",
    "progress_ratio",
]
