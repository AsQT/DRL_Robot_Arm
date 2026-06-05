"""
Simple reward manual testing with scripted trajectories.

Tests the generalized reward:
  r_t = r_success + r_collision + r_distance + r_workspace + r_episode + r_shake

Usage::

    python Evaluation/test_obs_reward_manual.py --gui true
    python Evaluation/test_obs_reward_manual.py --gui false --interactive false --scene hard --run-hard-trajs true
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
from drl_pathplanning.gymnasium.config import Config
from drl_pathplanning.pybullet import HAVE_PYBULLET
from drl_pathplanning.pybullet.frame_viewer import FrameViewer, FrameViewerSceneSpec


# =============================================================================
# USER CONFIG
# =============================================================================

CONFIG_PATH = "config/environment.yaml"

GUI = True
SLEEP = 0.0
MAX_STEPS = 500

# --- Easy scene ---
MANUAL_START: list = [0.330, -0.350, 0.100]
MANUAL_TARGET: list = [-0.050, -0.550, 0.100]
MANUAL_OBSTACLE_ENABLED = True
MANUAL_OBSTACLE_CENTER: list = [0.145, -0.550, 0.080]
MANUAL_OBSTACLE_SIZE: list = [0.100, 0.100, 0.100]

# --- Hard scene: start/target in same y, obstacle in direct path ---
HARD_START: list = [0.330, -0.550, 0.100]
HARD_TARGET: list = [-0.050, -0.550, 0.100]
HARD_OBSTACLE_CENTER: list = [0.145, -0.550, 0.080]
HARD_OBSTACLE_SIZE: list = [0.100, 0.100, 0.100]

MANUAL_WAYPOINT_EPS = 0.001


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_vec3(s: str) -> list:
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 floats, got: {s!r}")
    return parts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple reward manual testing")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    p.add_argument("--sleep", type=float, default=SLEEP)
    p.add_argument("--interactive", type=str, default="true", choices=["true", "false"])
    p.add_argument("--scene", type=str, default="easy", choices=["easy", "hard"])
    p.add_argument("--run-hard-trajs", type=str, default="false", choices=["true", "false"])
    return p


def _apply_cli_overrides(args) -> None:
    global MAX_STEPS, GUI, SLEEP
    MAX_STEPS = args.max_steps
    GUI = args.gui.lower() == "true"
    SLEEP = args.sleep


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scalar(v):
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "item"):
        return float(v.item())
    if hasattr(v, "__float__"):
        return float(v)
    return 0.0


def _fmt_vec(arr) -> str:
    return f"[{float(arr[0]):+.4f},{float(arr[1]):+.4f},{float(arr[2]):+.4f}]"


def _create_viewer(cfg: Config) -> Optional[FrameViewer]:
    if not HAVE_PYBULLET:
        print("[WARN] PyBullet not available.")
        return None
    scene = FrameViewerSceneSpec(
        workspace_min=cfg.workspace.min_np.tolist(),
        workspace_max=cfg.workspace.max_np.tolist(),
        target_region_min=cfg.target_region.min_np.tolist(),
        target_region_max=cfg.target_region.max_np.tolist(),
        table_center=cfg.table.center,
        table_half_extent=cfg.table.half_extent_np.tolist(),
        table_color=cfg.table.color,
        start_position=cfg.start.fixed_position,
        target_position=cfg.target_region.fixed_position,
        obstacle_center=cfg.obstacle.center if cfg.obstacle.enabled else None,
        obstacle_half_extent=[s / 2.0 for s in cfg.obstacle.size] if cfg.obstacle.enabled else None,
        obstacle_color=getattr(cfg.obstacle, "color", [1.0, 0.5, 0.0]),
        frame_axis_length=0.05,
        frame_axis_width=2,
    )
    return FrameViewer(scene)


# --------------------------------------------------------------------------- #
# Scene management
# --------------------------------------------------------------------------- #

EASY_SCENE = {
    "start": MANUAL_START,
    "target": MANUAL_TARGET,
    "obstacle_enabled": True,
    "obstacle_center": MANUAL_OBSTACLE_CENTER,
    "obstacle_size": MANUAL_OBSTACLE_SIZE,
}

HARD_SCENE = {
    "start": HARD_START,
    "target": HARD_TARGET,
    "obstacle_enabled": True,
    "obstacle_center": HARD_OBSTACLE_CENTER,
    "obstacle_size": HARD_OBSTACLE_SIZE,
}


def _load_and_override_config(scene: dict) -> Config:
    cfg = Config.from_yaml(Path(CONFIG_PATH))
    cfg.start.mode = "fixed"
    cfg.start.fixed_position = scene["start"]
    cfg.target_region.enabled = True
    cfg.target_region.mode = "fixed"
    cfg.target_region.fixed_position = scene["target"]
    if scene["obstacle_enabled"]:
        cfg.obstacle.enabled = True
        cfg.obstacle.mode = "fixed"
        cfg.obstacle.center = scene["obstacle_center"]
        cfg.obstacle.size = scene["obstacle_size"]
        cfg.obstacle.size_random.enabled = False
    else:
        cfg.obstacle.enabled = False
    return cfg


def _print_scene_info(scene: dict):
    start = scene["start"]
    target = scene["target"]
    dist = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
    print(f"\n  Scene:")
    print(f"    start : {start}")
    print(f"    target: {target}")
    print(f"    obstacle_enabled: {scene['obstacle_enabled']}")
    if scene["obstacle_enabled"]:
        print(f"    obstacle_center: {scene['obstacle_center']}")
        print(f"    obstacle_size : {scene['obstacle_size']}")
    print(f"    dist(start->target): {dist:.4f} m")
    if scene["obstacle_enabled"]:
        obs_c = np.array(scene["obstacle_center"])
        obs_s = np.array(scene["obstacle_size"])
        start_p = np.array(start[:2])
        target_p = np.array(target[:2])
        obs_min = obs_c[:2] - obs_s[:2] / 2
        obs_max = obs_c[:2] + obs_s[:2] / 2
        # Check if segment crosses AABB
        crosses_x = start_p[0] < obs_max[0] and target_p[0] > obs_min[0]
        crosses_y = start_p[1] < obs_max[1] and target_p[1] > obs_min[1]
        in_obs_x = start_p[0] >= obs_min[0] and start_p[0] <= obs_max[0]
        in_obs_y = start_p[1] >= obs_min[1] and start_p[1] <= obs_max[1]
        crosses = (crosses_x and in_obs_y) or (crosses_y and in_obs_x)
        print(f"    direct path crosses obstacle: {crosses}")


# --------------------------------------------------------------------------- #
# Trajectory waypoint factories
# --------------------------------------------------------------------------- #

def _get_waypoints_correct(scene: dict) -> list[np.ndarray]:
    """Obstacle-aware: go high over obstacle."""
    obs_c = np.array(scene["obstacle_center"], dtype=np.float32)
    obs_s = np.array(scene["obstacle_size"], dtype=np.float32)
    safe_z = float(obs_c[2] + obs_s[2] / 2.0 + 0.050)
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    p1 = np.array([obs_c[0] + obs_s[0] / 2.0 + 0.050, start[1], start[2]], dtype=np.float32)
    p2 = np.array([p1[0], p1[1], safe_z], dtype=np.float32)
    p3 = np.array([target[0] - obs_s[0] / 2.0 - 0.050, target[1], safe_z], dtype=np.float32)
    p4 = np.array([target[0], target[1], target[2]], dtype=np.float32)
    return [p1, p2, p3, p4]


def _get_waypoints_around(scene: dict) -> list[np.ndarray]:
    """Go around obstacle on y- side."""
    obs_c = np.array(scene["obstacle_center"], dtype=np.float32)
    obs_s = np.array(scene["obstacle_size"], dtype=np.float32)
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    p1 = np.array([start[0], float(obs_c[1] - obs_s[1] / 2.0 - 0.100), start[2]], dtype=np.float32)
    p2 = np.array([target[0], float(obs_c[1] - obs_s[1] / 2.0 - 0.100), target[2]], dtype=np.float32)
    return [p1, p2]


def _get_waypoints_low_through(scene: dict) -> list[np.ndarray]:
    """Through obstacle at low height."""
    obs_c = np.array(scene["obstacle_center"], dtype=np.float32)
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    p1 = np.array([obs_c[0], obs_c[1], start[2]], dtype=np.float32)
    p2 = np.array([target[0], target[1], target[2]], dtype=np.float32)
    return [p1, p2]


def _get_hard_direct(scene: dict) -> list[np.ndarray]:
    """Straight line — crosses obstacle."""
    return []


def _get_hard_over(scene: dict) -> list[np.ndarray]:
    """Go high over obstacle."""
    obs_c = np.array(scene["obstacle_center"], dtype=np.float32)
    obs_s = np.array(scene["obstacle_size"], dtype=np.float32)
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    safe_z = float(obs_c[2] + obs_s[2] / 2.0 + 0.060)
    p1 = np.array([obs_c[0] + obs_s[0] / 2.0 + 0.060, start[1], start[2]], dtype=np.float32)
    p2 = np.array([p1[0], p1[1], safe_z], dtype=np.float32)
    p3 = np.array([target[0] - obs_s[0] / 2.0 - 0.060, target[1], safe_z], dtype=np.float32)
    p4 = np.array([target[0], target[1], target[2]], dtype=np.float32)
    return [p1, p2, p3, p4]


def _get_hard_around_y(scene: dict) -> list[np.ndarray]:
    """Go around obstacle on y- side."""
    obs_c = np.array(scene["obstacle_center"], dtype=np.float32)
    obs_s = np.array(scene["obstacle_size"], dtype=np.float32)
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    p1 = np.array([start[0], float(obs_c[1] - obs_s[1] / 2.0 - 0.150), start[2]], dtype=np.float32)
    p2 = np.array([target[0], float(obs_c[1] - obs_s[1] / 2.0 - 0.150), target[2]], dtype=np.float32)
    return [p1, p2]


def _get_hard_workspace_bad(scene: dict) -> list[np.ndarray]:
    """Go outside workspace on y+ side."""
    start = np.array(scene["start"], dtype=np.float32)
    target = np.array(scene["target"], dtype=np.float32)
    p1 = np.array([start[0], 0.100, start[2]], dtype=np.float32)
    p2 = np.array([target[0], 0.100, target[2]], dtype=np.float32)
    return [p1, p2]


# --------------------------------------------------------------------------- #
# Trajectory registry
# --------------------------------------------------------------------------- #

TRAJS = {
    "easy": {
        "run_direct": {
            "waypoints_fn": lambda s: [],
            "desc": "Straight line to target",
        },
        "run_correct": {
            "waypoints_fn": _get_waypoints_correct,
            "desc": "High over obstacle",
        },
        "run_low_through": {
            "waypoints_fn": _get_waypoints_low_through,
            "desc": "Through obstacle at low height",
        },
        "run_around": {
            "waypoints_fn": _get_waypoints_around,
            "desc": "Go around obstacle",
        },
    },
    "hard": {
        "direct": {
            "waypoints_fn": _get_hard_direct,
            "desc": "Straight to target (crosses obstacle)",
        },
        "over": {
            "waypoints_fn": _get_hard_over,
            "desc": "High over obstacle",
        },
        "around_y": {
            "waypoints_fn": _get_hard_around_y,
            "desc": "Around obstacle on y- side",
        },
        "workspace_bad": {
            "waypoints_fn": _get_hard_workspace_bad,
            "desc": "Go outside workspace on y+",
        },
    },
}


# --------------------------------------------------------------------------- #
# Trajectory runner
# --------------------------------------------------------------------------- #

def _run_trajectory(
    env,
    waypoints: list[np.ndarray],
    label: str,
    max_steps: int = 500,
    sleep_time: float = 0.0,
) -> dict:
    """Run through waypoints, accumulating reward. Returns final info."""
    obs, info = env.reset(seed=None)
    target_arr = np.array(env._target_pos, dtype=np.float32)
    start_arr = env.current_pos.copy()

    print(f"\n{'=' * 60}")
    print(f"  TRAJECTORY: {label}")
    print(f"  Start: {_fmt_vec(start_arr)}  Target: {_fmt_vec(target_arr)}")
    print(f"{'=' * 60}")

    total_reward = 0.0
    last_info = info
    terminated = False
    truncated = False
    global_step = 0
    waypoint_idx = 0

    COMPONENTS = ["success", "collision", "distance", "workspace", "episode", "time", "shake"]
    comp_sums = {k: 0.0 for k in COMPONENTS}
    step_logs = []

    while not terminated and not truncated and global_step < max_steps:
        global_step += 1

        if waypoint_idx < len(waypoints):
            goal = waypoints[waypoint_idx]
        else:
            goal = target_arr

        pos = env.current_pos.copy()
        delta = goal - pos
        dist_goal = float(np.linalg.norm(delta))

        if dist_goal <= MANUAL_WAYPOINT_EPS:
            waypoint_idx += 1
            if waypoint_idx >= len(waypoints) + 1:
                break
            continue

        action_step = float(env._cfg.environment.action_step)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

        if dist_goal <= action_step:
            action = np.clip(delta / action_step, -action_high, action_high).astype(np.float32)
        else:
            unit_dir = delta / dist_goal
            action = (unit_dir * action_high).astype(np.float32)

        obs_next, reward, terminated, truncated, info_next = env.step(action)
        total_reward += float(reward)
        last_info = info_next

        rc = info_next.get("reward_components", {})
        for k in COMPONENTS:
            comp_sums[k] += _scalar(rc.get(k, 0.0))

        dist_to_target = float(np.linalg.norm(env.current_pos - target_arr))
        step_logs.append({
            "step": global_step,
            "pos": env.current_pos.copy(),
            "dist_t": dist_to_target,
            "reward": float(reward),
            "rc": {k: _scalar(rc.get(k, 0.0)) for k in COMPONENTS},
        })

        if global_step <= 10:
            print(
                f"  s={global_step:3d}  pos={_fmt_vec(env.current_pos)}  "
                f"dist_t={dist_to_target:+.4f}  r={float(reward):+.4f}  "
                f"succ={int(info_next.get('is_success', False))}  "
                f"coll={int(info_next.get('is_collision', False))}"
            )

        if sleep_time > 0:
            time.sleep(sleep_time)

    # Last 5 steps audit
    print(f"\n  [AUDIT] Last 5 steps:")
    audit_steps = step_logs[-5:] if len(step_logs) >= 5 else step_logs
    print(f"  {'Step':>5}  {'dist_t':>8}  {'reward':>8}  {'succ':>5}  {'coll':>5}  {'success':>8}  {'collision':>9}  {'distance':>9}  {'shake':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
    for log in audit_steps:
        r = log["rc"]
        print(
            f"  {log['step']:>5d}  {log['dist_t']:>+8.4f}  {log['reward']:>+8.4f}  "
            f"{int(last_info.get('is_success', False) and log['step'] == global_step):>5}  "
            f"{int(last_info.get('is_collision', False) and log['step'] == global_step):>5}  "
            f"{r.get('success', 0.0):>+8.4f}  {r.get('collision', 0.0):>+9.4f}  "
            f"{r.get('distance', 0.0):>+9.4f}  {r.get('shake', 0.0):>+8.4f}"
        )

    sum_all = sum(comp_sums[k] for k in COMPONENTS)
    diff = total_reward - sum_all

    print(f"\n  Result:")
    print(f"    steps                   : {global_step}")
    print(f"    total_reward (env.sum): {total_reward:>+.4f}")
    print(f"    sum_components_total   : {sum_all:>+.4f}")
    print(f"    diff                   : {diff:>+.6f}  (should ~= 0)")
    print(f"    success                : {last_info.get('is_success', False)}")
    print(f"    collision              : {last_info.get('is_collision', False)}")
    print(f"    workspace              : {last_info.get('out_of_workspace', False)}")
    print(f"    final_pos              : {_fmt_vec(env.current_pos)}")
    print(f"  Component sums:")
    for k in COMPONENTS:
        print(f"    {k:20s}: {comp_sums[k]:>+.4f}")

    return {
        "label": label,
        "steps": global_step,
        "total_reward": total_reward,
        "sum_all": sum_all,
        "diff": diff,
        "components": comp_sums,
        "is_success": last_info.get("is_success", False),
        "is_collision": last_info.get("is_collision", False),
        "is_workspace": last_info.get("out_of_workspace", False),
        "final_pos": env.current_pos.copy(),
    }


# --------------------------------------------------------------------------- #
# Summary printer
# --------------------------------------------------------------------------- #

def _print_comparison(all_results: dict):
    print(f"\n{'=' * 110}")
    print(f"  TRAJECTORY COMPARISON")
    print(f"{'=' * 110}")
    print(f"  {'Trajectory':<18}  {'Steps':>5}  {'total_env':>9}  {'diff':>8}  {'Succ':>5}  {'Coll':>5}  {'WS':>3}  {'dist_sum':>9}  {'shake_sum':>9}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*9}  {'-'*9}")
    for name, r in all_results.items():
        c = r["components"]
        print(
            f"  {name:<18}  {r['steps']:>5}  {r['total_reward']:>+9.4f}  {r['diff']:>+8.4f}  "
            f"{int(r['is_success']):>5}  {int(r['is_collision']):>5}  {int(r['is_workspace']):>3}  "
            f"{c.get('distance', 0.0):>+9.4f}  {c.get('shake', 0.0):>+9.4f}"
        )
    print(f"\n  Component breakdown:")
    print(f"  {'Trajectory':<18}  {'success_sum':>11}  {'collision_sum':>13}  {'distance_sum':>12}  {'workspace_sum':>13}  {'episode_sum':>12}  {'time_sum':>9}")
    print(f"  {'-'*18}  {'-'*11}  {'-'*13}  {'-'*12}  {'-'*13}  {'-'*12}  {'-'*9}")
    for name, r in all_results.items():
        c = r["components"]
        print(
            f"  {name:<18}  {c.get('success', 0.0):>+11.4f}  {c.get('collision', 0.0):>+13.4f}  "
            f"{c.get('distance', 0.0):>+12.4f}  {c.get('workspace', 0.0):>+13.4f}  "
            f"{c.get('episode', 0.0):>+12.4f}  {c.get('time', 0.0):>+9.4f}"
        )
    print(f"\n  diff = total_env - sum_components (should ~= 0)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = _build_parser().parse_args()
    _apply_cli_overrides(args)

    # Determine scene
    scene_name = args.scene
    scene = EASY_SCENE if scene_name == "easy" else HARD_SCENE

    print(f"\n{'=' * 60}")
    print(f"  SIMPLE REWARD MANUAL TESTING")
    print(f"  r_t = r_success + r_collision + r_distance + r_workspace + r_episode + r_shake")
    print(f"  Scene: {scene_name}")
    print(f"{'=' * 60}")
    print(f"  GUI       : {GUI}")
    print(f"  max_steps : {MAX_STEPS}")
    print(f"  sleep     : {SLEEP}")

    cfg = _load_and_override_config(scene)
    env = CartesianPathPlanningEnv(
        env_cfg=cfg,
        start_mode="fixed",
        start_pos=tuple(scene["start"]),
    )

    viewer = None
    if GUI and HAVE_PYBULLET:
        viewer = _create_viewer(cfg)

    _print_scene_info(scene)

    # Non-interactive hard trajs mode
    if not (args.interactive.lower() == "true") and args.run_hard_trajs.lower() == "true":
        print(f"\n{'=' * 60}")
        print(f"  Running hard scene trajectories...")
        print(f"{'=' * 60}")
        traj_list = TRAJS["hard"]
        all_results = {}
        for name, tdef in traj_list.items():
            waypoints = tdef["waypoints_fn"](scene)
            result = _run_trajectory(env, waypoints, f"{name} ({tdef['desc']})", MAX_STEPS, SLEEP)
            all_results[name] = result
        _print_comparison(all_results)
        env.close()
        return

    # Interactive mode
    print(f"\nCommands: scene_easy | scene_hard | run_all | <traj_name> | reset | q")
    print(f"  Available trajs for '{scene_name}': {' | '.join(TRAJS[scene_name].keys())}")

    while True:
        try:
            raw = input("\ncmd> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("q", "quit", "exit"):
            print("Quitting...")
            break

        if cmd == "scene_easy":
            scene_name = "easy"
            scene = EASY_SCENE
            cfg = _load_and_override_config(scene)
            env.close()
            env = CartesianPathPlanningEnv(env_cfg=cfg, start_mode="fixed", start_pos=tuple(scene["start"]))
            if viewer is not None:
                viewer = _create_viewer(cfg)
            _print_scene_info(scene)
            print(f"  Available trajs: {' | '.join(TRAJS[scene_name].keys())}")
            continue

        if cmd == "scene_hard":
            scene_name = "hard"
            scene = HARD_SCENE
            cfg = _load_and_override_config(scene)
            env.close()
            env = CartesianPathPlanningEnv(env_cfg=cfg, start_mode="fixed", start_pos=tuple(scene["start"]))
            if viewer is not None:
                viewer = _create_viewer(cfg)
            _print_scene_info(scene)
            print(f"  Available trajs: {' | '.join(TRAJS[scene_name].keys())}")
            continue

        if cmd == "reset":
            obs, info = env.reset(seed=None)
            print(f"[RESET] pos={_fmt_vec(env.current_pos)}")
            continue

        if cmd == "run_all":
            all_results = {}
            for name, tdef in TRAJS[scene_name].items():
                waypoints = tdef["waypoints_fn"](scene)
                obs, info = env.reset(seed=None)
                result = _run_trajectory(env, waypoints, f"{name} ({tdef['desc']})", MAX_STEPS, SLEEP)
                all_results[name] = result
            _print_comparison(all_results)
            continue

        if cmd in TRAJS[scene_name]:
            tdef = TRAJS[scene_name][cmd]
            waypoints = tdef["waypoints_fn"](scene)
            obs, info = env.reset(seed=None)
            _run_trajectory(env, waypoints, f"{cmd} ({tdef['desc']})", MAX_STEPS, SLEEP)
            continue

        print(f"Unknown: {cmd!r}")
        print(f"  scene_easy | scene_hard | run_all | reset | q")
        print(f"  trajs: {' | '.join(TRAJS[scene_name].keys())}")

    env.close()


if __name__ == "__main__":
    main()
