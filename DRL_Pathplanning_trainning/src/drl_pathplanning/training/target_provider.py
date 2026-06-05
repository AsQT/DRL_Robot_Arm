"""
Target position selection for the Cartesian path planning environment.

Single source of truth for all target sampling logic used by both training
and evaluation. The ``TargetProvider`` class and ``build_target_sequence()``
function handle:

- **random**  : uniform sample inside the target region (optionally avoiding the obstacle)
- **static**  : one of 8 corner vertices of the target region AABB
- **fixed**   : user-provided override position

Usage in environment reset::

    provider = TargetProvider(
        target_region_min=cfg.target_region.min_np,
        target_region_max=cfg.target_region.max_np,
        avoid_box_center=obstacle_center,
        avoid_box_half_extent=collision_half_extent,
    )
    target = provider.get_target(
        mode="random", rng=np_random, fixed_target=None, corner_index=None
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def get_target_region_corners(
    target_min: np.ndarray, target_max: np.ndarray
) -> List[np.ndarray]:
    """
    Generate the 8 corner vertices of an AABB target region.

    Corners are returned in canonical order::

        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),

    Parameters
    ----------
    target_min
        Lower corner [x_min, y_min, z_min].
    target_max
        Upper corner [x_max, y_max, z_max].

    Returns
    -------
    List[np.ndarray]
        List of 8 corner positions as 3-D float arrays.
    """
    target_min = np.asarray(target_min, dtype=np.float32)
    target_max = np.asarray(target_max, dtype=np.float32)

    x0, y0, z0 = target_min
    x1, y1, z1 = target_max

    return [
        np.array([x0, y0, z0], dtype=np.float32),
        np.array([x0, y0, z1], dtype=np.float32),
        np.array([x0, y1, z0], dtype=np.float32),
        np.array([x0, y1, z1], dtype=np.float32),
        np.array([x1, y0, z0], dtype=np.float32),
        np.array([x1, y0, z1], dtype=np.float32),
        np.array([x1, y1, z0], dtype=np.float32),
        np.array([x1, y1, z1], dtype=np.float32),
    ]


class TargetProvider:
    """
    Provides target positions for the Cartesian path planning environment.

    Parameters
    ----------
    target_region_min, target_region_max : np.ndarray or None
        Bounding box of the target sampling region. If None, falls back to
        the workspace bounds (from ``workspace_min`` / ``workspace_max``).
    workspace_min, workspace_max : np.ndarray or None
        Workspace bounds. Used as fallback when target_region is not set.
    avoid_box_center, avoid_box_half_extent : np.ndarray or None
        If provided, random samples are rejected when they fall inside this box
        (inflated collision zone of the obstacle).
    """

    def __init__(
        self,
        target_region_min: Optional[np.ndarray] = None,
        target_region_max: Optional[np.ndarray] = None,
        workspace_min: Optional[np.ndarray] = None,
        workspace_max: Optional[np.ndarray] = None,
        avoid_box_center: Optional[np.ndarray] = None,
        avoid_box_half_extent: Optional[np.ndarray] = None,
    ) -> None:
        self._tr_min = (
            np.asarray(target_region_min, dtype=np.float32)
            if target_region_min is not None
            else None
        )
        self._tr_max = (
            np.asarray(target_region_max, dtype=np.float32)
            if target_region_max is not None
            else None
        )
        self._ws_min = (
            np.asarray(workspace_min, dtype=np.float32)
            if workspace_min is not None
            else None
        )
        self._ws_max = (
            np.asarray(workspace_max, dtype=np.float32)
            if workspace_max is not None
            else None
        )
        self._avoid_center = (
            np.asarray(avoid_box_center, dtype=np.float32)
            if avoid_box_center is not None
            else None
        )
        self._avoid_half = (
            np.asarray(avoid_box_half_extent, dtype=np.float32)
            if avoid_box_half_extent is not None
            else None
        )

        # Cache static corners.
        self._corners: Optional[List[np.ndarray]] = None
        if self._tr_min is not None and self._tr_max is not None:
            self._corners = get_target_region_corners(self._tr_min, self._tr_max)

    @property
    def target_region_min(self) -> Optional[np.ndarray]:
        return self._tr_min.copy() if self._tr_min is not None else None

    @property
    def target_region_max(self) -> Optional[np.ndarray]:
        return self._tr_max.copy() if self._tr_max is not None else None

    @property
    def corners(self) -> Optional[List[np.ndarray]]:
        """The 8 corner positions of the target region, or None if region is not set."""
        return list(self._corners) if self._corners is not None else None

    def _sampling_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the (min, max) arrays for random sampling.

        Prefers target_region bounds; falls back to workspace bounds.
        """
        if self._tr_min is not None and self._tr_max is not None:
            return self._tr_min, self._tr_max
        if self._ws_min is not None and self._ws_max is not None:
            return self._ws_min, self._ws_max
        # Ultimate fallback — should never reach here in normal operation.
        return (
            np.array([-0.2, -0.8, 0.02], dtype=np.float32),
            np.array([0.5, 0.0, 0.32], dtype=np.float32),
        )

    def _sample_random(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a uniform random target, optionally avoiding the exclusion box.
        Target z is always forced to exactly 0.10.
        """
        from drl_pathplanning import geometry as _geo

        smin, smax = self._sampling_bounds()

        for _ in range(100):
            candidate = rng.uniform(smin, smax).astype(np.float32)
            # Force z to exactly 0.10 as required.
            candidate[2] = 0.10

            if self._avoid_center is not None and self._avoid_half is not None:
                if _geo.check_point_in_box(candidate, self._avoid_center, self._avoid_half):
                    continue

            return candidate

        # Fallback: return the centre of the sampling region, z forced to 0.10.
        fallback = ((smin + smax) / 2.0).astype(np.float32)
        fallback[2] = 0.10
        return fallback

    def get_target(
        self,
        mode: str,
        rng: np.random.Generator,
        fixed_target: Optional[np.ndarray] = None,
        corner_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return a target position based on the specified mode.

        Parameters
        ----------
        mode : str
            One of ``"random"``, ``"static"``, or ``"fixed"``.
        rng
            NumPy random Generator used for random sampling.
        fixed_target
            Used when ``mode="fixed"``. Must be a 3-element array.
        corner_index
            Used when ``mode="static"``. Integer in 0..7 selecting one of the
            8 corner vertices. Out-of-range values are clamped.

        Returns
        -------
        np.ndarray
            3-D target position.
        """
        if mode == "fixed":
            if fixed_target is None:
                raise ValueError(
                    "mode='fixed' requires a fixed_target to be provided"
                )
            return np.asarray(fixed_target, dtype=np.float32).copy()

        if mode == "static":
            if self._corners is None:
                raise ValueError(
                    "Cannot use mode='static': target_region min/max are not set"
                )
            idx = int(corner_index) if corner_index is not None else 0
            idx = max(0, min(7, idx))
            return self._corners[idx].copy()

        if mode == "random":
            return self._sample_random(rng)

        raise ValueError(
            f"mode must be 'random', 'static', or 'fixed', got '{mode}'"
        )


def build_target_sequence(
    target_region_min: np.ndarray,
    target_region_max: np.ndarray,
    repeat: int = 1,
) -> List[tuple[int, int, np.ndarray]]:
    """
    Build a deterministic sequence of (corner_index, total_corners, corner_pos)
    tuples covering all 8 corners of the target region.

    This is useful for evaluation scripts that want to sweep all corners.

    Parameters
    ----------
    target_region_min
        Lower corner of the target region.
    target_region_max
        Upper corner of the target region.
    repeat
        Number of times to repeat the full 8-corner cycle.

    Returns
    -------
    List[tuple]
        List of ``(corner_idx, 8, corner_pos)`` tuples in corner order 0..7.
    """
    corners = get_target_region_corners(target_region_min, target_region_max)
    sequence: List[tuple] = []
    for _ in range(repeat):
        for corner_idx, corner_pos in enumerate(corners):
            sequence.append((corner_idx, len(corners), corner_pos))
    return sequence
