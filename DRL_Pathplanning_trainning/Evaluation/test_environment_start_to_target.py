"""
Environment display test for the Cartesian frame environment.

Uses the same terminated/truncated logic as the training environment.
Episode ends when: done = terminated or truncated

The Gymnasium env is pure (no PyBullet).  This script creates the FrameViewer
directly and drives it using the info dict returned by reset() / step().

Run with defaults (clean output):
    python Evaluation/test_environment_start_to_target.py

With per-episode obstacle summary:
    python Evaluation/test_environment_start_to_target.py --debug-obstacle true

Full audit (body / AABB / obs detail):
    python Evaluation/test_environment_start_to_target.py --debug-obstacle true --debug-body-audit true --debug-aabb true --debug-obs-detail true
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from drl_pathplanning.gymnasium.config import Config
from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
from drl_pathplanning.pybullet import build_viz_config, HAVE_PYBULLET
from drl_pathplanning.pybullet.viewer_sync import sync_obstacle_to_viewer
from drl_pathplanning.pybullet.frame_viewer import FrameViewer, FrameViewerSceneSpec

_pb = None
if HAVE_PYBULLET:
    import pybullet as _pb


# --------------------------------------------------------------------------- #
# Observation layout constants
# --------------------------------------------------------------------------- #
_OBS_REL_START = 9
_OBS_REL_END = 12
_OBS_SIZE_START = 12
_OBS_SIZE_END = 15
_OBS_SEG_START = _OBS_REL_START
_OBS_SEG_END = _OBS_SIZE_END


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def straight_action(current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Deterministic geometric controller: move directly toward the target."""
    delta = np.asarray(target_pos, dtype=np.float32) - np.asarray(current_pos, dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    action = delta / norm
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def episode_end_reason(info: dict, terminated: bool, truncated: bool) -> str:
    """Determine human-readable end-of-episode reason from info + flags."""
    if info.get("is_success"):
        return "success"
    if info.get("box_collision"):
        return "collision"
    if info.get("table_collision"):
        return "collision"
    if info.get("obstacle_collision"):
        return "collision"
    if info.get("out_of_workspace"):
        return "workspace_violation"
    if truncated:
        return "max_steps"
    if terminated:
        return "terminated"
    return "unknown"


def _compute_expected_segment(
    env_center: np.ndarray,
    env_size: np.ndarray,
    current_pos: np.ndarray,
    ws_range: np.ndarray,
) -> np.ndarray:
    """Replicate normalize_obstacle_info formula."""
    half_extent = np.asarray(env_size, dtype=np.float32) / 2.0
    rel_obs = (
        np.asarray(env_center, dtype=np.float32)
        - np.asarray(current_pos, dtype=np.float32)
    ) / np.asarray(ws_range, dtype=np.float32)
    obs_sz = half_extent / np.asarray(ws_range, dtype=np.float32)
    return np.concatenate([rel_obs, obs_sz]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Episode runner
# --------------------------------------------------------------------------- #
def run_episode(
    env: CartesianPathPlanningEnv,
    viewer: FrameViewer | None,
    cfg,  # Config object for shared helper
    target_pos: np.ndarray | None,
    target_mode: str,
    max_steps: int,
    sleep_time: float,
    episode_idx: int,
    verbose_steps: bool = False,
    show_obstacle: bool = False,
    obstacle_cfg: object = None,
    debug_obstacle: bool = False,
    debug_body_audit: bool = False,
    debug_aabb: bool = False,
    debug_obs_detail: bool = False,
) -> dict:
    """
    Run one episode using the training environment's own terminated/truncated logic.
    Episode ends when: done = terminated or truncated
    """
    if target_mode == "random":
        obs, info = env.reset(seed=None)
        actual_target = env.target_pos.copy()
    elif target_mode == "fixed":
        obs, info = env.reset(seed=42)
        env.set_target(target_pos)
        actual_target = env.target_pos.copy()
    else:
        obs, info = env.reset(
            seed=42,
            options={"target_mode": "static", "static_corner_index": episode_idx - 1},
        )
        actual_target = env.target_pos.copy()

    start_pos_ep = env.start_pos_episode.copy()

    obs_mode = info.get("obstacle_mode", "unknown")
    obs_center = info.get("obstacle_center")
    obs_size = info.get("obstacle_size")
    obs_enabled = info.get("obstacle_enabled", False)

    # Body audit before reset_episode
    if debug_body_audit and viewer is not None and viewer.viewer is not None:
        viewer.viewer.debug_bodies(f"ep{episode_idx}-before_reset_episode")

    if viewer is not None:
        viewer.reset_episode(start_pos_ep, actual_target)
        viewer.draw_expected_path(
            start_pos_ep, actual_target, color=[1.0, 0.0, 0.0], line_width=3
        )

    # Body audit after reset_episode
    if debug_body_audit and viewer is not None and viewer.viewer is not None:
        viewer.viewer.debug_bodies(f"ep{episode_idx}-after_reset_episode")

    # Body audit before update_obstacle
    if debug_body_audit and viewer is not None and viewer.viewer is not None:
        viewer.viewer.debug_bodies(f"ep{episode_idx}-before_update_obstacle")

    # Update viewer obstacle per episode using the shared helper
    if viewer is not None and show_obstacle:
        sync_obstacle_to_viewer(
            viewer=viewer,
            env=env,
            cfg=cfg,
            info=info,
            debug=debug_obstacle,
            prefix=f"TEST ep{episode_idx}",
        )

    # Body audit after update_obstacle
    if debug_body_audit and viewer is not None and viewer.viewer is not None:
        viewer.viewer.debug_bodies(f"ep{episode_idx}-after_update_obstacle")

    # ------------------------------------------------------------------ #
    # [DEBUG-OBS] Clean per-episode obstacle summary (controlled by --debug-obstacle)
    # ------------------------------------------------------------------ #
    if debug_obstacle:
        _ws_range = env.unwrapped._cfg.workspace.range_np
        _env_obs_half = (np.asarray(obs_size, dtype=np.float32) / 2.0) if obs_size is not None else None
        _exp_seg = np.zeros(6, dtype=np.float32)
        _seg_match = False

        if obs_center is not None and _env_obs_half is not None and obs is not None:
            _exp_seg = _compute_expected_segment(obs_center, obs_size, obs[:3], _ws_range)
            _act_seg = obs[_OBS_SEG_START:_OBS_SEG_END].copy()
            _seg_match = bool(np.allclose(_act_seg, _exp_seg, atol=1e-5))

        _sr_enabled = (
            getattr(obstacle_cfg, "size_random", None) is not None
            and getattr(obstacle_cfg.size_random, "enabled", False)
        ) if obstacle_cfg else False

        _table_top_z = float(
            env.unwrapped._cfg.table.center_np[2]
            + env.unwrapped._cfg.table.size_np[2] / 2.0
        )
        _box_bottom_z = (
            float(obs_center[2]) - float(obs_size[2]) / 2.0
            if obs_center is not None and obs_size is not None else None
        )
        _box_top_z = (
            float(obs_center[2]) + float(obs_size[2]) / 2.0
            if obs_center is not None and obs_size is not None else None
        )

        _box_body_count = (
            len(viewer.viewer._obstacle_body_ids) if viewer is not None and viewer.viewer is not None else -1
        )

        _vi = "N/A"
        _vh = "N/A"
        if viewer is not None and viewer.viewer is not None:
            if viewer.viewer._box_center is not None:
                _vi = f"[{float(viewer.viewer._box_center[0]):+.4f},{float(viewer.viewer._box_center[1]):+.4f},{float(viewer.viewer._box_center[2]):+.4f}]"
            if viewer.viewer._box_half_extent is not None:
                _vh = f"[{float(viewer.viewer._box_half_extent[0]):.4f},{float(viewer.viewer._box_half_extent[1]):.4f},{float(viewer.viewer._box_half_extent[2]):.4f}]"

        _bbi_str = f"{_box_bottom_z:+.4f}" if _box_bottom_z is not None else "N/A"
        _bti_str = f"{_box_top_z:+.4f}" if _box_top_z is not None else "N/A"
        _cx = f"{obs_center[0]:+.4f}" if obs_center is not None else "N/A"
        _cy = f"{obs_center[1]:+.4f}" if obs_center is not None else "N/A"
        _cz = f"{obs_center[2]:+.4f}" if obs_center is not None else "N/A"
        _sz_str = (
            f"[{float(obs_size[0]):.4f},{float(obs_size[1]):.4f},{float(obs_size[2]):.4f}]"
            if obs_size is not None else "N/A"
        )
        _sm_str = "PASS" if _seg_match else "FAIL"

        print(
            f"  [DEBUG-OBS] ep={episode_idx}  mode={obs_mode}  SR={_sr_enabled}  enabled={obs_enabled}"
        )
        print(
            f"    center=[{_cx},{_cy},{_cz}]  size={_sz_str}  "
            f"table_top_z={_table_top_z:+.4f}"
        )
        print(
            f"    box_bottom_z={_bbi_str}  box_top_z={_bti_str}  "
            f"viewer_box={_vi}  viewer_half={_vh}  "
            f"body_count={_box_body_count}  obs_match={_sm_str}"
        )

    # ------------------------------------------------------------------ #
    # [DEBUG-AABB] Box-position geometry audit (controlled by --debug-aabb)
    # ------------------------------------------------------------------ #
    if debug_aabb and viewer is not None and viewer.viewer is not None:
        _env_cfg = env.unwrapped._cfg
        _tc = _env_cfg.table.center
        _ts = _env_cfg.table.size
        _table_top_z = _tc[2] + _ts[2] / 2.0

        _ecx = float(obs_center[0]) if obs_center is not None else None
        _ecy = float(obs_center[1]) if obs_center is not None else None
        _ecz = float(obs_center[2]) if obs_center is not None else None
        _esx = float(obs_size[0]) if obs_size is not None else None
        _esy = float(obs_size[1]) if obs_size is not None else None
        _esz = float(obs_size[2]) if obs_size is not None else None

        _ehz = _esz / 2.0 if _esz is not None else None
        _exp_cz = _table_top_z + _ehz if _ehz is not None else None
        _act_bot = _ecz - _ehz if _ecz is not None and _ehz is not None else None

        _chk_env_cz = bool(abs(_ecz - _exp_cz) < 1e-6) if _ecz is not None and _exp_cz is not None else False
        _chk_env_bot = bool(abs(_act_bot - _table_top_z) < 1e-6) if _act_bot is not None else False

        _has_vw = (
            viewer.viewer._box_center is not None
            and viewer.viewer._box_half_extent is not None
        )
        _chk_vw_c = False
        _chk_vw_bot = False
        if _has_vw and _ecx is not None:
            _vw_arr = np.array([float(viewer.viewer._box_center[0]),
                                  float(viewer.viewer._box_center[1]),
                                  float(viewer.viewer._box_center[2])], dtype=np.float32)
            _env_arr = np.array([_ecx, _ecy, _ecz], dtype=np.float32)
            _chk_vw_c = bool(np.linalg.norm(_vw_arr - _env_arr) < 1e-6)
            _vw_bot = float(viewer.viewer._box_center[2]) - float(viewer.viewer._box_half_extent[2])
            _chk_vw_bot = bool(abs(_vw_bot - _table_top_z) < 1e-6)

        def _f(v): return f"{v:+.6f}" if v is not None else "N/A"
        def _p(v): return "PASS" if v else "FAIL"

        print()
        print(f"[AABB-CHECK] ep={episode_idx}  mode={obs_mode}  SR={getattr(obstacle_cfg, 'size_random', None) and obstacle_cfg.size_random.enabled if obstacle_cfg else False}")
        print(f"  table_top_z={_table_top_z:+.6f}")
        print(f"  env_center=[{_f(_ecx)},{_f(_ecy)},{_f(_ecz)}]  env_size=[{_f(_esx)},{_f(_esy)},{_f(_esz)}]")
        print(f"  exp_center_z={_f(_exp_cz)}  act_center_z={_f(_ecz)}  => {_p(_chk_env_cz)}")
        print(f"  act_bottom_z={_f(_act_bot)}  table_top_z={_f(_table_top_z)} => {_p(_chk_env_bot)}")
        if _has_vw:
            print(f"  viewer_center=[{float(viewer.viewer._box_center[0]):+.4f},{float(viewer.viewer._box_center[1]):+.4f},{float(viewer.viewer._box_center[2]):+.4f}] => {_p(_chk_vw_c)}")
            print(f"  viewer_bottom_z={float(viewer.viewer._box_center[2])-float(viewer.viewer._box_half_extent[2]):+.6f} => {_p(_chk_vw_bot)}")

    # ------------------------------------------------------------------ #
    # [DEBUG-OBS-DETAIL] Full observation audit (controlled by --debug-obs-detail)
    # ------------------------------------------------------------------ #
    if debug_obs_detail:
        _ws_range = env.unwrapped._cfg.workspace.range_np
        _env_obs_center = info.get("obstacle_center")
        _env_obs_size = info.get("obstacle_size")
        _env_obs_half = (_env_obs_size / 2.0) if _env_obs_size is not None else None

        if _env_obs_center is not None and _env_obs_half is not None and obs is not None:
            _exp_seg = _compute_expected_segment(_env_obs_center, _env_obs_size, obs[:3], _ws_range)
        else:
            _exp_seg = np.zeros(6, dtype=np.float32)

        _act_seg = obs[_OBS_SEG_START:_OBS_SEG_END].copy() if obs is not None else None
        _seg_match = bool(np.allclose(_act_seg, _exp_seg, atol=1e-5)) if _act_seg is not None else False

        def _afmt(v): return f"{float(v):+.6f}" if v is not None else "N/A"

        print()
        print(f"[OBS-DETAIL] ep={episode_idx}")
        print(f"  obs_shape={obs.shape if obs is not None else None}")
        print(f"  obs[0:3]={[_afmt(obs[i]) for i in range(3)]}")
        print(f"  obs[3:6]={[_afmt(obs[i]) for i in range(3,6)]}")
        print(f"  obs[6:9]={[_afmt(obs[i]) for i in range(6,9)]}")
        print(f"  obs[9:12]={[_afmt(obs[i]) for i in range(9,12)]}")
        print(f"  obs[12:15]={[_afmt(obs[i]) for i in range(12,15)]}")
        print(f"  env_center={_env_obs_center}  env_size={_env_obs_size}")
        print(f"  exp_rel_obs={[_afmt(_exp_seg[i]) for i in range(3)]}")
        print(f"  exp_obs_size={[_afmt(_exp_seg[i]) for i in range(3,6)]}")
        print(f"  act_rel_obs={[_afmt(_act_seg[i]) for i in range(3)]}")
        print(f"  act_obs_size={[_afmt(_act_seg[i]) for i in range(3,6)]}")
        print(f"  diff={[_afmt(float(_act_seg[i])-float(_exp_seg[i])) for i in range(6)]}")
        print(f"  seg_match={'PASS' if _seg_match else 'FAIL'}")

        # size_random bug check
        if (obstacle_cfg is not None
                and getattr(obstacle_cfg, "size_random", None) is not None
                and getattr(obstacle_cfg.size_random, "enabled", False)
                and _env_obs_size is not None):
            _cfg_fixed = getattr(obstacle_cfg, "size", None)
            if _cfg_fixed is not None:
                _cfg_half = np.asarray(_cfg_fixed, dtype=np.float32) / 2.0
                _cfg_obs_sz = _cfg_half / np.asarray(_ws_range, dtype=np.float32)
                _uses_fixed = bool(np.allclose(obs[_OBS_SIZE_START:_OBS_SIZE_END], _cfg_obs_sz, atol=1e-5))
                if _uses_fixed:
                    print(f"  [BUG] obs_size uses cfg.fixed_size despite size_random.enabled=TRUE!")
                else:
                    print(f"  [OK] obs_size uses env_obstacle_size")

    # ------------------------------------------------------------------ #
    # Step loop
    # ------------------------------------------------------------------ #
    episode_reward = 0.0
    step_idx = 0
    _step_exp_seg = np.zeros(6, dtype=np.float32)
    _step_obs_center = None
    _step_obs_size = None

    if obs_center is not None and obs_size is not None:
        _ws_range = env.unwrapped._cfg.workspace.range_np
        _step_exp_seg = _compute_expected_segment(obs_center, obs_size, obs[:3], _ws_range)
        _step_obs_center = obs_center.copy()
        _step_obs_size = obs_size.copy()

    for step_idx in range(max_steps):
        action = straight_action(env.current_pos, env.target_pos)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)

        # [DEBUG-OBS-STEP] Verify obs_size is stable within episode
        if debug_obs_detail and step_idx < 3 and _step_obs_size is not None:
            _step_obs_full = obs
            _step_obs_sz = _step_obs_full[_OBS_SIZE_START:_OBS_SIZE_END]
            _step_sz_changed = not np.allclose(_step_obs_sz, _step_exp_seg[3:6], atol=1e-5)
            _af2 = lambda v: f"{float(v):+.6f}"
            print()
            print(f"[OBS-STEP] ep={episode_idx} step={step_idx}")
            print(f"  obs_size={[_af2(_step_obs_sz[i]) for i in range(3)]}  changed={_step_sz_changed}")

        if viewer is not None:
            prev_pos = info["prev_pos"]
            curr_pos = info["current_pos"]
            delta = curr_pos - prev_pos
            delta_norm = float(np.linalg.norm(delta))
            if verbose_steps and step_idx < 10:
                dist = float(np.linalg.norm(curr_pos - env.target_pos))
                print(
                    f"  [DEBUG] ep={episode_idx} "
                    f"step={step_idx}  "
                    f"prev=({prev_pos[0]:+.4f},{prev_pos[1]:+.4f},{prev_pos[2]:+.4f})  "
                    f"curr=({curr_pos[0]:+.4f},{curr_pos[1]:+.4f},{curr_pos[2]:+.4f})  "
                    f"delta={delta_norm:.6f}  dist={dist:.4f}"
                )
            if delta_norm > 1e-9:
                viewer.draw_path_segment(prev_pos, curr_pos)
            viewer.update_agent(curr_pos)
            viewer.step()

        if sleep_time > 0:
            time.sleep(sleep_time)

        done = terminated or truncated
        if done:
            break

    reason = episode_end_reason(info, terminated, truncated)
    collision = (
        info.get("box_collision")
        or info.get("table_collision")
        or info.get("obstacle_collision", False)
    )

    return {
        "target_pos": actual_target,
        "steps": step_idx + 1,
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated or truncated,
        "end_reason": reason,
        "is_success": bool(info.get("is_success", False)),
        "total_reward": episode_reward,
        "final_distance": float(info.get("distance", 0.0)),
        "path_length": float(info.get("path_length_so_far", 0.0)),
        "collision": bool(collision),
        "collision_object": str(info.get("collision_object", "none")),
        "table_collision": bool(info.get("table_collision", False)),
        "box_collision": bool(info.get("box_collision", False)),
        "workspace_violation": bool(info.get("out_of_workspace", False)),
        "termination_reason": str(info.get("termination_reason", "none")),
        "start_pos": start_pos_ep.copy(),
        "expected_path_length": float(info.get("expected_path_length", float("nan"))),
        "actual_path_length": float(info.get("actual_path_length", float("nan"))),
        "path_efficiency": float(info.get("path_efficiency", float("nan"))),
        "path_efficiency_percent": float(info.get("path_efficiency_percent", float("nan"))),
        "path_efficiency_valid": bool(info.get("path_efficiency_valid", False)),
        # debug fields for per-episode obstacle summary
        "obstacle_center": obs_center.copy() if obs_center is not None else None,
        "obstacle_size": obs_size.copy() if obs_size is not None else None,
        "obstacle_mode": obs_mode,
        "obstacle_enabled": obs_enabled,
    }


def _get_obstacle_cfg(cfg: Config):
    """Return obstacle config, preferring cfg.obstacle over legacy cfg.box."""
    if hasattr(cfg, "obstacle"):
        return cfg.obstacle
    if hasattr(cfg, "box"):
        return cfg.box
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    default_config = _SCRIPT_DIR.parent / "config" / "environment.yaml"
    fallback_config = _SCRIPT_DIR.parent / "config" / "environment.yaml"

    parser = argparse.ArgumentParser(description="Environment display test")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument(
        "--start", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed start position (m). Overrides --start-mode.",
    )
    parser.add_argument(
        "--start-mode", type=str, default="config",
        choices=["config", "fixed", "random"],
    )
    parser.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="0 = use env max_episode_steps (default: 0)",
    )
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--show", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--verbose", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--obstacle", type=str, default="config",
        choices=["true", "false", "config"],
    )
    parser.add_argument(
        "--origin", type=str, default="true",
        choices=["true", "false"],
    )
    # --- Debug flags ---
    parser.add_argument(
        "--debug-obstacle", type=str, default="false",
        choices=["true", "false"],
        help="Per-episode obstacle summary: center, size, bottom/top, obs_match.",
    )
    parser.add_argument(
        "--debug-body-audit", type=str, default="false",
        choices=["true", "false"],
        help="PyBullet body audit before/after reset/update.",
    )
    parser.add_argument(
        "--debug-aabb", type=str, default="false",
        choices=["true", "false"],
        help="AABB geometry check: center_z, bottom_z match expected.",
    )
    parser.add_argument(
        "--debug-obs-detail", type=str, default="false",
        choices=["true", "false"],
        help="Full observation audit: all obs values, expected, diff.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(Path(args.config))
    gui = args.gui.lower() == "true"
    verbose_show = args.show.lower() == "true"
    verbose_steps = args.verbose.lower() == "true"
    debug_obstacle = args.debug_obstacle.lower() == "true"
    debug_body_audit = args.debug_body_audit.lower() == "true"
    debug_aabb = args.debug_aabb.lower() == "true"
    debug_obs_detail = args.debug_obs_detail.lower() == "true"

    if args.obstacle == "true":
        show_obstacle = True
    elif args.obstacle == "false":
        show_obstacle = False
    else:
        obs_cfg = _get_obstacle_cfg(cfg)
        show_obstacle = bool(getattr(obs_cfg, "enabled", False)) if obs_cfg else False

    show_origin = args.origin.lower() == "true"
    obs_cfg = _get_obstacle_cfg(cfg)
    if obs_cfg is not None and getattr(obs_cfg, "enabled", False):
        obstacle_name = getattr(obs_cfg, "name", "small_obstacle")
        obstacle_center = getattr(obs_cfg, "center", None)
        obstacle_size = getattr(obs_cfg, "size", None)
        obstacle_half_extent = [s / 2.0 for s in obstacle_size] if obstacle_size else None
    else:
        obstacle_name = "none"
        obstacle_center = None
        obstacle_size = None
        obstacle_half_extent = None

    # Resolve start mode
    if args.start is not None:
        _start_mode = "fixed"
        start_pos = np.array(args.start, dtype=np.float32)
    elif args.start_mode == "fixed":
        _start_mode = "fixed"
        start_pos = np.array(cfg.start.fixed_position, dtype=np.float32)
    elif args.start_mode == "random":
        _start_mode = "random"
        start_pos = None
    else:
        _start_mode = "config"
        start_pos = None

    target_mode = "fixed" if args.target is not None else "random"
    target_pos = np.array(args.target, dtype=np.float32) if args.target is not None else None

    max_steps = args.steps if args.steps > 0 else cfg.environment.max_episode_steps

    env = CartesianPathPlanningEnv(
        env_cfg=cfg,
        start_mode=_start_mode,
        start_pos=tuple(start_pos.tolist()) if start_pos is not None else None,
    )

    viewer: FrameViewer | None = None
    if gui and HAVE_PYBULLET:
        scene = FrameViewerSceneSpec(
            workspace_min=cfg.workspace.min_np.tolist(),
            workspace_max=cfg.workspace.max_np.tolist(),
            target_region_min=cfg.target_region.min_np.tolist(),
            target_region_max=cfg.target_region.max_np.tolist(),
            table_center=cfg.table.center,
            table_half_extent=cfg.table.half_extent_np.tolist(),
            table_color=cfg.table.color,
            box_center=obstacle_center if show_obstacle else None,
            box_half_extent=obstacle_half_extent if show_obstacle else None,
            box_color=(
                cfg.obstacle.visual.color
                if (show_obstacle and getattr(cfg.obstacle, "visual", None))
                else [0.1, 0.1, 0.1, 1.0]
            ),
            _obstacle_cfg=cfg.obstacle,
            gui=True,
            show_labels=True,
            show_workspace=True,
            show_target_region=True,
            show_table=True,
            hide_debug_ui=True,
        )
        viewer = FrameViewer.from_scene(scene)

    if debug_body_audit and viewer is not None and viewer.viewer is not None:
        viewer.viewer.debug_bodies("after from_scene")

    if verbose_show:
        _env_target_min = cfg.target_region.min_np.tolist()
        _env_target_max = cfg.target_region.max_np.tolist()
        print("=" * 62)
        print("  Environment Display Test")
        print("=" * 62)
        print(f"  Config       : {args.config}")
        print(f"  Episodes    : {args.episodes}")
        print(f"  Sleep       : {args.sleep:.2f} s")
        print(f"  Target mode : {target_mode.upper()}")
        print(f"  Start mode  : {_start_mode.upper()}")
        print(f"  Action step: {cfg.environment.action_step}")
        print(f"  Max steps   : {max_steps}")
        print(f"  GUI         : {gui}")
        print("-" * 62)
        print(f"  table_center : {cfg.table.center}")
        print(f"  table_size   : {cfg.table.size}")
        print("-" * 62)
        print(f"  obstacle_enabled    : {cfg.obstacle.enabled}")
        print(f"  obstacle_mode       : {cfg.obstacle.resolved_mode}")
        print(f"  obstacle_center     : {obstacle_center}")
        print(f"  obstacle_size       : {obstacle_size}")
        print(f"  show_obstacle      : {show_obstacle}")
        print(f"  debug_obstacle     : {debug_obstacle}")
        print(f"  debug_body_audit   : {debug_body_audit}")
        print(f"  debug_aabb         : {debug_aabb}")
        print(f"  debug_obs_detail   : {debug_obs_detail}")
        print("=" * 62)

    results = []
    for ep in range(args.episodes):
        result = run_episode(
            env=env,
            viewer=viewer,
            cfg=cfg,
            target_pos=target_pos,
            target_mode=target_mode,
            max_steps=max_steps,
            sleep_time=args.sleep,
            episode_idx=ep + 1,
            verbose_steps=verbose_steps,
            show_obstacle=show_obstacle,
            obstacle_cfg=obs_cfg,
            debug_obstacle=debug_obstacle,
            debug_body_audit=debug_body_audit,
            debug_aabb=debug_aabb,
            debug_obs_detail=debug_obs_detail,
        )
        results.append(result)

        # Per-episode summary line
        t = result["target_pos"]
        s = result["start_pos"]
        r = result["end_reason"]
        ok = "OK" if result["is_success"] else "--"
        col = result["collision_object"] if result["collision"] else "none"
        eff_pct = result["path_efficiency_percent"]
        eff_str = (
            f"{eff_pct:.1f}%"
            if result["is_success"] and result["path_efficiency_valid"] and np.isfinite(eff_pct)
            else "N/A"
        )
        print(
            f"  Ep {ep + 1:2d}  "
            f"[{_start_mode.upper()}]  "
            f"start=[{s[0]:+.4f},{s[1]:+.4f},{s[2]:+.4f}]  "
            f"target=[{t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}]  "
            f"{ok}  "
            f"reason={r:<18s}  "
            f"steps={result['steps']:4d}  "
            f"dist={result['final_distance']:.4f}  "
            f"eff%={eff_str}  "
            f"COL={'Y' if result['collision'] else 'N'}  "
            f"COL_OBJ={col:<8s}  "
            f"rew={result['total_reward']:+.2f}"
        )

        if gui and args.sleep > 0:
            time.sleep(0.3)

    if viewer is not None:
        viewer.close()
    env.close()

    # Summary table
    n_ok = sum(1 for r in results if r["is_success"])
    n_col = sum(1 for r in results if r["collision"])
    n_ws = sum(1 for r in results if r["workspace_violation"])
    n_max = sum(1 for r in results if r["end_reason"] == "max_steps")
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_dist = sum(r["final_distance"] for r in results) / len(results)
    avg_path = sum(r["actual_path_length"] for r in results) / len(results)
    total_reward = sum(r["total_reward"] for r in results)

    success_results = [r for r in results if r["is_success"]]
    if success_results:
        s_eff = [
            r["path_efficiency_percent"]
            for r in success_results
            if np.isfinite(r["path_efficiency_percent"])
        ]
        avg_eff = np.mean(s_eff) if s_eff else float("nan")
        avg_exp_pl = np.mean([r["expected_path_length"] for r in success_results])
        avg_act_pl = np.mean([r["actual_path_length"] for r in success_results])
    else:
        avg_eff = float("nan")
        avg_exp_pl = float("nan")
        avg_act_pl = float("nan")

    print()
    print("=" * 62)
    print("  Summary")
    print("=" * 62)
    print(f"  Success          : {n_ok}/{len(results)}")
    print(f"  Collision        : {n_col}")
    print(f"  WS violation     : {n_ws}")
    print(f"  Max steps        : {n_max}")
    print(f"  Avg steps        : {avg_steps:.1f}")
    print(f"  Avg dist         : {avg_dist:.4f} m")
    print(f"  Avg act path     : {avg_path:.4f} m")
    if np.isfinite(avg_eff):
        print(f"  Avg success eff %: {avg_eff:.2f}%  (n={len(success_results)})")
        print(f"  Avg success exp_pl: {avg_exp_pl:.4f} m")
        print(f"  Avg success act_pl: {avg_act_pl:.4f} m")
    print(f"  Total reward     : {total_reward:+.4f}")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())
