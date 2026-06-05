"""
Training GUI viewer callback for the Cartesian path planning environment.

This module provides :class:`TrainingViewerCallback` — an SB3 callback that
drives a shared :class:`FrameViewer` during training for live visual debug.

It is intentionally separate from the Gymnasium environment and from the
training core so that normal headless training carries no PyBullet dependency.

Usage::

    from drl_pathplanning.training.viewer_callback import create_training_viewer_callback

    callback = create_training_viewer_callback(
        env_cfg=cfg,
        raw_env=gym_env,
        gui=True,
        show=True,
        render_sleep=0.05,
        render_first_episodes=5,
    )
    model.learn(total_timesteps=10000, callback=[..., callback])

The callback:
- Creates a :class:`FrameViewer` from the env config using ``FrameViewerSceneSpec``.
- Listens to env reset via the ``done`` flag in ``_on_step``.
- Reads ``start_pos``, ``target_pos``, ``current_pos`` from the SB3 ``infos`` dict.
- Draws the expected (red) straight-line path from start to target.
- Updates the agent marker and draws actual path segments every step.
- Cleans up the viewer on ``close()``.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import stable_baselines3.common.callbacks as sb3_cb

from drl_pathplanning.pybullet.viewer_sync import (
    sync_obstacle_to_viewer,
    unwrap_cartesian_env,
)

# Suppress Gymnasium deprecation warnings triggered by VecEnv wrapper access to
# env.current_pos / env.target_pos.  These warnings come from SB3's VecEnv
# internals and do not affect correctness.
warnings.filterwarnings("ignore", message=".*env\\.current_pos to get variables.*")
warnings.filterwarnings("ignore", message=".*env\\.target_pos to get variables.*")

if TYPE_CHECKING:
    from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
    from drl_pathplanning.gymnasium.config import Config
    from drl_pathplanning.pybullet.frame_viewer import FrameViewer


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #

def create_training_viewer_callback(
    env_cfg: "Config",
    raw_env: "CartesianPathPlanningEnv | list[CartesianPathPlanningEnv]",
    training_env: "Any",
    gui: bool = True,
    show: bool = False,
    render_sleep: float = 0.0,
    render_first_episodes: int = 0,
) -> Optional["TrainingViewerCallback"]:
    """
    Create a :class:`TrainingViewerCallback` wired to a shared :class:`FrameViewer`.

    Environment behavior is driven by config flags (obstacle.enabled, etc.),
    not by a mode string parameter.

    Parameters
    ----------
    env_cfg
        Populated environment :class:`Config` object.
    raw_env
        The bare ``CartesianPathPlanningEnv`` (or list of them for multi-env).
    training_env
        The ``DummyVecEnv`` or ``SubprocVecEnv`` that the SB3 model
        trains on (VecNormalize is disabled in the raw-observation pipeline).
        The callback reads ``current_pos`` from this env so it gets
        the positions that SB3 actually sees.
    gui
        Open the PyBullet GUI window.
    show
        Print per-episode start/target positions to stdout.
    render_sleep
        Seconds to sleep between steps (for real-time viewing).
    render_first_episodes
        Render only the first N episodes. ``0`` means render all episodes.

    Returns
    -------
    TrainingViewerCallback or None
        The callback, or ``None`` if PyBullet is not available.
    """
    from drl_pathplanning.pybullet import (
        FrameViewer,
        FrameViewerSceneSpec,
        build_viz_config,
        HAVE_PYBULLET,
    )

    if not HAVE_PYBULLET:
        print("[WARN] PyBullet not available; GUI viewer disabled.")
        return None

    viz_cfg = build_viz_config(env_cfg, gui=gui)
    show_table = env_cfg.table.enabled

    scene = FrameViewerSceneSpec(
        workspace_min=env_cfg.workspace.min_np.tolist(),
        workspace_max=env_cfg.workspace.max_np.tolist(),
        target_region_min=env_cfg.target_region.min_np.tolist(),
        target_region_max=env_cfg.target_region.max_np.tolist(),
        table_center=env_cfg.table.center,
        table_half_extent=env_cfg.table.half_extent_np.tolist(),
        table_color=env_cfg.table.color,
        # Obstacle box is drawn in draw_static_scene via obstacle config,
        # not via scene.box_center here.
        box_center=(
            env_cfg.obstacle.center_np.tolist() if env_cfg.obstacle.enabled else None
        ),
        box_half_extent=(
            env_cfg.obstacle.half_extent_np.tolist() if env_cfg.obstacle.enabled else None
        ),
        box_color=(
            env_cfg.obstacle.visual.color if (
                env_cfg.obstacle.enabled
                and getattr(env_cfg.obstacle, "visual", None)
            ) else [0.1, 0.1, 0.1, 1.0]
        ),
        _obstacle_cfg=env_cfg.obstacle,
        gui=gui,
        show_workspace=viz_cfg.get("show_workspace", True),
        show_target_region=viz_cfg.get("show_target_region", True),
        show_table=show_table,
        show_path=viz_cfg.get("show_path", True),
        show_labels=viz_cfg.get("show_labels", True),
        show_start_frame=viz_cfg.get("show_start_frame", True),
        show_target_frame=viz_cfg.get("show_target_frame", True),
        show_agent_frame=viz_cfg.get("show_agent_frame", False),
        hide_debug_ui=viz_cfg.get("hide_debug_ui", True),
        expected_path_color=viz_cfg.get("expected_path_color"),
        expected_path_width=viz_cfg.get("expected_path_line_width", 3),
        path_color=viz_cfg.get("actual_path_color"),
        path_line_width=viz_cfg.get("path_line_width", 5),
        agent_radius=viz_cfg.get("agent_radius"),
        start_radius=viz_cfg.get("start_sphere_radius"),
        target_radius=viz_cfg.get("target_sphere_radius"),
        camera_distance=viz_cfg.get("camera_distance", 1.2),
        camera_yaw=viz_cfg.get("camera_yaw", 45.0),
        camera_pitch=viz_cfg.get("camera_pitch", -35.0),
        camera_target=viz_cfg.get("camera_target"),
    )

    viewer = FrameViewer.from_scene(scene)

    return TrainingViewerCallback(
        viewer=viewer,
        env_cfg=env_cfg,
        raw_env=raw_env,
        training_env=training_env,
        show=show,
        render_sleep=render_sleep,
        render_first_episodes=render_first_episodes,
    )


# --------------------------------------------------------------------------- #
# Callback implementation
# --------------------------------------------------------------------------- #

class TrainingViewerCallback(sb3_cb.BaseCallback):
    """
    SB3 callback that drives a :class:`FrameViewer` during training for visual debug.

    Renders the first N episodes (or all episodes when ``render_first_episodes=0``).
    Updates the viewer with start, target, and agent positions every step, and draws
    the expected (red) straight-line path plus actual path segments.

    The Gymnasium env remains completely unaware of the viewer — no PyBullet logic
    is added to ``CartesianPathPlanningEnv``.
    """

    def __init__(
        self,
        viewer: "FrameViewer",
        env_cfg: "Config",
        raw_env: "CartesianPathPlanningEnv | list[CartesianPathPlanningEnv]",
        training_env: "Any",
        show: bool = False,
        render_sleep: float = 0.0,
        render_first_episodes: int = 0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._viewer = viewer
        self._env_cfg = env_cfg
        self._raw_env = raw_env
        self._training_env = training_env  # DummyVecEnv or SubprocVecEnv (VecNormalize disabled)
        self._show = bool(show)
        self._render_sleep = float(render_sleep)
        self._render_first_episodes = int(render_first_episodes)
        self._render_all = bool(render_first_episodes <= 0)

        # Episode tracking
        self._current_episode_index: int = 0   # 1-based episode number
        self._rendered_episode_count: int = 0  # how many episodes we've fully set up

        # Per-episode state
        self._steps_in_episode: int = 0
        self._prev_pos: Optional[np.ndarray] = None

        # Render state machine
        self._active_rendering: bool = False  # currently drawing a rendered episode
        # Set True initially so the very first _on_step call initializes episode 1.
        self._waiting_for_next_episode: bool = True
        self._close_requested: bool = False  # viewer should be closed after last rendered ep

        # Cached episode-end info for the done-log print
        self._last_done_info: Optional[dict] = None

    @property
    def _cartesian_env(self) -> Any:
        """Resolve the bare CartesianPathPlanningEnv from the training env.

        Uses ``unwrap_cartesian_env`` to drill through Monitor, DummyVecEnv,
        SubprocVecEnv, and any other wrappers.
        """
        return unwrap_cartesian_env(self._training_env)

    def close(self) -> None:
        """Disconnect the PyBullet viewer on cleanup."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _should_render_episode(self) -> bool:
        """Check if the NEXT episode to start should be rendered."""
        if self._render_all:
            return True
        return self._rendered_episode_count < self._render_first_episodes

    def _update_viewer_for_new_episode(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
    ) -> None:
        """Clear path history, draw start/target/expected path, place agent."""
        self._viewer.clear_path()
        self._viewer.clear_expected_path()

        self._viewer.draw_start(start_pos)
        self._viewer.draw_target(target_pos)
        self._viewer.draw_expected_path(start_pos, target_pos)
        self._viewer.reset_episode(start_pos, target_pos)
        self._viewer.update_agent(current_pos)

        self._prev_pos = current_pos.copy()
        self._steps_in_episode = 0

        if self._show:
            print(
                f"[GUI] Ep {self._current_episode_index}: "
                f"start=({start_pos[0]:+.4f},{start_pos[1]:+.4f},{start_pos[2]:+.4f})  "
                f"target=({target_pos[0]:+.4f},{target_pos[1]:+.4f},{target_pos[2]:+.4f})"
            )

    # ------------------------------------------------------------------ #
    # SB3 lifecycle hooks
    # ------------------------------------------------------------------ #

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        # ── Phase 1: Handle episode-end ─────────────────────────────────────────
        if dones is not None and len(dones) > 0 and dones[0]:
            # Capture episode-end info from info dict for the done log.
            if infos is not None and len(infos) > 0:
                info = infos[0]
                self._last_done_info = {
                    "is_success": bool(info.get("is_success", False)),
                    "collision": bool(info.get("collision", info.get("is_collision", info.get("box_collision", False)))),
                    "workspace_violation": bool(info.get("out_of_workspace", False)),
                    "termination_reason": str(info.get("termination_reason", "unknown")),
                }
            else:
                self._last_done_info = {
                    "is_success": False,
                    "collision": False,
                    "workspace_violation": False,
                    "termination_reason": "unknown",
                }

            if self._active_rendering:
                # Print episode-end summary while viewer still shows the final state.
                if self._show:
                    final_dist = 0.0
                    if infos is not None and len(infos) > 0:
                        final_dist = float(infos[0].get("distance", 0.0))
                    print(
                        f"[GUI] Ep {self._current_episode_index} done: success={self._last_done_info['is_success']}, "
                        f"collision={self._last_done_info['collision']}, "
                        f"workspace_violation={self._last_done_info['workspace_violation']}, "
                        f"steps={self._steps_in_episode}, final_distance={final_dist:.4f}"
                    )

            self._active_rendering = False
            self._waiting_for_next_episode = True
            self._steps_in_episode = 0
            self._prev_pos = None

        # ── Phase 2: Detect first step of a new episode ─────────────────────────
        # When _waiting_for_next_episode is True, the current step is the first
        # step of the new episode (SB3 has already called env.reset()).
        # IMPORTANT: don't init the next episode on the SAME _on_step that ended
        # the current episode (done=True). SB3 calls reset() AFTER all on_step
        # callbacks return, so the next step (done=False) is when the new state
        # is valid. Defer init to the next callback invocation.
        if self._waiting_for_next_episode and not (dones is not None and len(dones) > 0 and dones[0]):
            self._waiting_for_next_episode = False

            # Advance episode index. Phase 1 increments at episode END (for done-log),
            # Phase 2 increments at episode START (for init/start-log). Together they
            # keep done-log and next-episode start-log consistent.
            self._current_episode_index += 1

            # Decide whether this episode should be rendered.
            if self._should_render_episode():
                # Read new episode start/target from info dict (preferred).
                if infos is not None and len(infos) > 0:
                    info = infos[0]
                    start_pos = np.asarray(
                        info.get("start_pos", [0.0, 0.0, 0.0]), dtype=np.float32
                    )
                    target_pos = np.asarray(
                        info.get("target_pos", [0.0, 0.0, 0.0]), dtype=np.float32
                    )
                else:
                    # Fallback: unwrap and read from bare CartesianPathPlanningEnv.
                    env = unwrap_cartesian_env(self._training_env)
                    if env is not None and hasattr(env, "start_pos_episode"):
                        start_pos = env.start_pos_episode
                    else:
                        start_pos = np.zeros(3, dtype=np.float32)
                    if env is not None and hasattr(env, "target_pos"):
                        target_pos = env.target_pos
                    else:
                        target_pos = np.zeros(3, dtype=np.float32)

                self._update_viewer_for_new_episode(start_pos, target_pos, start_pos)

                # ── Sync obstacle to viewer per episode ───────────────────────
                # Use the shared helper so test and train use identical logic.
                sync_obstacle_to_viewer(
                    viewer=self._viewer,
                    env=self._training_env,
                    cfg=self._env_cfg,
                    info=infos,
                    debug=False,
                    prefix=f"TRAIN",
                )

                self._rendered_episode_count += 1
                self._active_rendering = True
            else:
                # Beyond the render window — close viewer and disable rendering.
                if not self._close_requested:
                    self._close_requested = True
                    if self._show:
                        print("[GUI] Render window complete; closing viewer.")

        # ── Phase 3: Render every step of the active rendered episodes ──────────
        if self._active_rendering:
            env = unwrap_cartesian_env(self._training_env)
            try:
                current_pos = env.current_pos
            except Exception:
                if self.verbose:
                    print("[WARN] TrainingViewerCallback: could not read current_pos; skipping render step.")
                return True

            # Sync viewer agent position first so _prev_pos is meaningful.
            self._viewer.update_agent(current_pos)

            prev = self._prev_pos
            if prev is not None:
                delta = current_pos - prev
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > 1e-9:
                    self._viewer.draw_path_segment(prev, current_pos)
            else:
                # First step of episode — _prev_pos was set by _update_viewer_for_new_episode.
                pass

            self._prev_pos = current_pos.copy()
            self._steps_in_episode += 1

            if self._render_sleep > 0:
                time.sleep(self._render_sleep)

        return True
