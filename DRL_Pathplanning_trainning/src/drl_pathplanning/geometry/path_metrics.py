"""
Path quality metrics computation.

Provides pure functions for computing path efficiency, smoothness,
and other trajectory-level quality metrics.
No Gymnasium, no PyBullet, no training code.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "path_efficiency",
    "path_length",
    "cosine_smoothness",
]


def path_length(waypoints: list[np.ndarray]) -> float:
    """
    Total Cartesian length of a waypoint list.

    Parameters
    ----------
    waypoints : list of np.ndarray
        List of 3-D positions.

    Returns
    -------
    float
        Total path length in metres.
    """
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(waypoints)):
        total += float(np.linalg.norm(
            np.asarray(waypoints[i], dtype=np.float32)
            - np.asarray(waypoints[i - 1], dtype=np.float32)
        ))
    return total


def path_efficiency(straight_line: float, actual_length: float) -> float:
    """
    Path efficiency = straight_line / actual_length.

    Returns 1.0 for a perfect straight-line path.
    Returns 0.0 for an infinitely long path.

    Parameters
    ----------
    straight_line : float
        Direct distance from start to target.
    actual_length : float
        Total path length.

    Returns
    -------
    float
        Efficiency in [0, 1]. Returns 0.0 if actual_length <= 0.
    """
    if actual_length <= 0.0:
        return 0.0
    return float(np.clip(straight_line / actual_length, 0.0, 1.0))


def cosine_smoothness(action_deltas: list[float] | np.ndarray) -> float:
    """
    Mean cosine similarity between consecutive action delta directions.

    Measures trajectory smoothness. Returns 1.0 for perfectly smooth
    (same direction) actions, 0.0 for orthogonal, -1.0 for opposite.

    Parameters
    ----------
    action_deltas : list of floats or np.ndarray
        Per-step action delta magnitudes.

    Returns
    -------
    float
        Mean cosine alignment. Returns 0.0 if fewer than 2 steps.
    """
    if len(action_deltas) < 2:
        return 0.0
    arr = np.asarray(action_deltas, dtype=np.float32)
    total = 0.0
    count = 0
    for i in range(1, len(arr)):
        d0 = max(abs(arr[i - 1]), 1e-6)
        d1 = max(abs(arr[i]), 1e-6)
        # Sign similarity — positive means same direction trend
        total += float(np.sign(arr[i] * arr[i - 1]))
        count += 1
    return total / count if count > 0 else 0.0


def progress_ratio(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_dist: float,
) -> float:
    """
    Progress toward target since last step.

    Parameters
    ----------
    current_pos, target_pos : np.ndarray
        Positions.
    prev_dist : float
        Distance at the previous step.

    Returns
    -------
    float
        prev_dist - current_distance (positive = progress, negative = regress).
    """
    current_dist = float(np.linalg.norm(
        np.asarray(target_pos, dtype=np.float32) - np.asarray(current_pos, dtype=np.float32)
    ))
    return prev_dist - current_dist
