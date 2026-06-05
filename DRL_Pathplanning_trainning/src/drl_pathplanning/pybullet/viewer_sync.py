"""
Shared viewer-synchronisation helpers for the Cartesian frame environment.

Provides a single source of truth for syncing the PyBullet viewer with the
current state of the environment (obstacle geometry, episode positions, etc.)
across all callers — evaluation scripts, test scripts, and training callbacks.

No PyBullet logic lives here; only attribute resolution and viewer dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ObstacleGeometry:
    """Current obstacle geometry and metadata."""
    center: "np.ndarray"
    size: "np.ndarray"
    half_extent: "np.ndarray"
    source: str  # "info" | "env_property" | "cfg_fallback"
    enabled: bool


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def unwrap_cartesian_env(env: Any) -> Any:
    """
    Drill through SB3 / Gymnasium wrappers to find the bare CartesianPathPlanningEnv.

    Wrapper chain in training::

        CartesianPathPlanningEnv → Monitor → DummyVecEnv → (training_env)

    For n_envs > 1::

        CartesianPathPlanningEnv → Monitor → SubprocVecEnv → (training_env)

    Parameters
    ----------
    env
        Any env in the wrapper chain, including ``None``.

    Returns
    -------
    Any
        The bare ``CartesianPathPlanningEnv``, or ``None`` if the chain cannot
        be resolved.
    """
    if env is None:
        return None

    # VecEnv (DummyVecEnv / SubprocVecEnv) — take the first worker
    if hasattr(env, "envs"):
        if env.envs:
            env = env.envs[0]

    # Drill through any .env wrappers (Monitor, VecNormalize, …)
    while hasattr(env, "env"):
        env = env.env

    return env


def get_current_obstacle_geometry(
    env: Any,
    info: Optional[dict] = None,
) -> Optional[ObstacleGeometry]:
    """
    Resolve the current obstacle geometry from the environment and/or info dict.

    Resolution order
    ---------------
    1. ``info["obstacle_center"]`` / ``info["obstacle_size"]``  — populated by
       ``CartesianPathPlanningEnv`` on every ``reset()`` / ``step()``.
    2. ``env.obstacle_center`` / ``env.obstacle_size``          — bare env properties.
    3. ``None`` if the obstacle is disabled or geometry is unavailable.

    Parameters
    ----------
    env
        Any env in the wrapper chain (will be unwrapped).
    info
        SB3 ``infos`` dict from the latest step.  May be ``None``.

    Returns
    -------
    ObstacleGeometry or None
        Current geometry with source attribution, or ``None`` if the obstacle
        is disabled or geometry cannot be resolved.
    """
    import numpy as np

    # ── Step 1: Try info dict (preferred — populated by env on every call) ──
    if info is not None and isinstance(info, (list, tuple)) and len(info) > 0:
        info = info[0]

    if info is not None:
        raw_center = info.get("obstacle_center")
        raw_size   = info.get("obstacle_size")
        raw_enabled = info.get("obstacle_enabled", True)
        if raw_center is not None and raw_size is not None and raw_enabled:
            size = np.asarray(raw_size, dtype=np.float32)
            return ObstacleGeometry(
                center=np.asarray(raw_center, dtype=np.float32),
                size=size,
                half_extent=size / 2.0,
                source="info",
                enabled=True,
            )

    # ── Step 2: Try bare env properties ──────────────────────────────────────
    bare = unwrap_cartesian_env(env)
    if bare is not None:
        try:
            center = bare.obstacle_center
            size    = bare.obstacle_size
            if center is not None and size is not None:
                size_f = np.asarray(size, dtype=np.float32)
                return ObstacleGeometry(
                    center=np.asarray(center, dtype=np.float32),
                    size=size_f,
                    half_extent=size_f / 2.0,
                    source="env_property",
                    enabled=True,
                )
        except AttributeError:
            pass

    # ── Step 3: Obstacle disabled or unavailable ────────────────────────────
    return None


def sync_obstacle_to_viewer(
    viewer: Any,
    env: Any,
    cfg: Any,
    info: Optional[dict] = None,
    debug: bool = False,
    prefix: str = "VIEWER-SYNC",
) -> None:
    """
    Synchronise the PyBullet viewer obstacle box with the current env state.

    This is the **single entry point** for updating the obstacle visualisation
    per episode across all callers (test scripts, training callbacks, etc.).

    Resolution order for geometry: :func:`get_current_obstacle_geometry`.

    If the obstacle is disabled or geometry cannot be resolved, this is a no-op.

    Parameters
    ----------
    viewer
        ``FrameViewer`` or ``FrameViewerWrapper`` with an ``update_obstacle()`` method.
    env
        Any env in the wrapper chain (will be unwrapped internally).
    cfg
        Populated ``Config`` (or ``ObstacleConfig``) object providing
        ``cfg.obstacle.visual.color`` and ``cfg.obstacle`` for collision settings.
    info
        SB3 ``infos`` dict from the latest step.  May be ``None``.
    debug
        If ``True``, print diagnostic lines before and after the update.
    prefix
        Label prepended to debug lines (e.g. ``"TEST"`` or ``"TRAIN"``).
    """
    import numpy as np

    geometry = get_current_obstacle_geometry(env, info)
    env_cfg = getattr(cfg, "obstacle", None)

    # ── Resolve box colour ──────────────────────────────────────────────────
    box_color = None
    if env_cfg is not None:
        visual = getattr(env_cfg, "visual", None)
        if visual is not None and getattr(visual, "enabled", False):
            box_color = getattr(visual, "color", None)

    # ── Debug: pre-update state ──────────────────────────────────────────────
    if debug:
        _vw_c = getattr(viewer, "_box_center", None)
        _vw_h = getattr(viewer, "_box_half_extent", None)
        _vw_s = getattr(viewer, "viewer", None)
        _pre_c = None
        _pre_h = None
        if _vw_s is not None:
            _pre_c = getattr(_vw_s, "_box_center", None)
            _pre_h = getattr(_vw_s, "_box_half_extent", None)

        def _arr_str(a):
            if a is None:
                return "None"
            try:
                return f"[{float(a[0]):+.4f},{float(a[1]):+.4f},{float(a[2]):+.4f}]"
            except (TypeError, IndexError):
                return str(a)

        print(f"[{prefix}] pre  viewer_center={_arr_str(_pre_c)}  viewer_half={_arr_str(_pre_h)}")
        if geometry is None:
            print(f"[{prefix}]       env.obstacle=None (disabled or unavailable)")
        else:
            print(
                f"[{prefix}]       geometry  center={_arr_str(geometry.center)}  "
                f"size={_arr_str(geometry.size)}  half={_arr_str(geometry.half_extent)}  "
                f"source={geometry.source}"
            )

    # ── No valid geometry — nothing to do ───────────────────────────────────
    if geometry is None or not geometry.enabled:
        if debug:
            print(f"[{prefix}]       update_obstacle=SKIP (no geometry)")
        return

    # ── Dispatch to viewer ──────────────────────────────────────────────────
    if not hasattr(viewer, "update_obstacle"):
        if debug:
            print(f"[{prefix}]       update_obstacle=SKIP (viewer has no method)")
        return

    try:
        viewer.update_obstacle(
            box_center=geometry.center,
            box_half_extent=geometry.half_extent,
            box_color=box_color,
            obstacle_cfg=env_cfg,
        )
        if debug:
            print(f"[{prefix}]       update_obstacle=OK")
    except Exception as exc:
        if debug:
            print(f"[{prefix}]       update_obstacle=FAIL ({exc})")
