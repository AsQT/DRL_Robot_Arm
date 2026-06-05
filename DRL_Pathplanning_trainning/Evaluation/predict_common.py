"""
Shared evaluation helpers for all algorithm predict scripts.

Provides:
  - Config loading from YAML (no inline config)
  - Model resolution with priority
  - Environment creation
  - Episode running (random & static modes)
  - Episode/CSV summary printing

Each algorithm-specific script (TD3, DDPG, SAC, PPO) imports from here
and only supplies the algorithm name and SB3 model class.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Protocol, TYPE_CHECKING

import numpy as np

import gymnasium as gym

from drl_pathplanning.gymnasium.config import Config
from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
from drl_pathplanning.training.trajectory import Trajectory
from drl_pathplanning.training.logger import print_prediction_summary

if TYPE_CHECKING:
    from stable_baselines3 import BaseAlgorithm


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #

def resolve_model_path(
    run_dir: Path,
    custom_zip: str = "",
) -> Path | None:
    """
    Resolve model path with priority:

    1. custom_zip  (if non-empty)
    2. model/best_model.zip
    3. model/final_model.zip
    4. best_model.zip
    5. final_model.zip

    Returns None if no model found.
    """
    candidates: List[Path] = []
    if custom_zip:
        candidates.append(run_dir / "model" / custom_zip)
    candidates.extend([
        run_dir / "model" / "best_model.zip",
        run_dir / "model" / "final_model.zip",
        run_dir / "best_model.zip",
        run_dir / "final_model.zip",
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# Environment creation
# --------------------------------------------------------------------------- #

def create_eval_env(
    config_path: Path,
    start_mode: str = "config",
    start_pos: tuple[float, float, float] | None = None,
    target_pos: tuple[float, float, float] | None = None,
) -> tuple[gym.Env, Config]:
    """
    Create a CartesianPathPlanning environment from a YAML config file.

    Parameters
    ----------
    config_path
        Path to the YAML config (e.g. config/environment.yaml).
    start_mode
        One of "config", "fixed", "random".
    start_pos
        Override start position (only used when start_mode="fixed").
    target_pos
        Override target position (fixed target for all episodes).

    Returns
    -------
    Tuple of (gym.Env, Config)
    """
    cfg: Config = Config.from_yaml(config_path)

    env: gym.Env = gym.make(
        "CartesianPathPlanning-Default-v0",
        env_cfg=cfg,
        start_mode=start_mode,
        start_pos=start_pos,
    )

    if target_pos is not None:
        env.set_target(np.array(target_pos, dtype=np.float32))

    return env, cfg


# --------------------------------------------------------------------------- #
# Episode running
# --------------------------------------------------------------------------- #

def run_random_episodes(
    model: "BaseAlgorithm",
    env: gym.Env,
    cfg: Config,
    num_episodes: int,
    deterministic: bool = True,
    sleep: float = 0.0,
    viewer=None,
    viewer_update_fn=None,
) -> List[dict]:
    """Run random-target episodes and return per-episode metrics."""
    results: List[dict] = []
    output_dir = get_project_root() / "Data" / "Prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"metrics_random_N_{num_episodes}.csv"

    fieldnames = [
        "episode", "start_mode",
        "start_x", "start_y", "start_z",
        "target_x", "target_y", "target_z",
        "is_success", "cumulative_reward", "episode_length",
        "final_distance", "expected_path_length", "actual_path_length",
        "path_efficiency_percent",
        "stop_reason",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for ep_idx in range(1, num_episodes + 1):
            obs, info = env.reset(seed=42 + ep_idx)
            unwrapped = env.unwrapped
            start_pos = unwrapped.current_pos.copy()
            target_pos = unwrapped.target_pos.copy()
            eff_mode = getattr(unwrapped, "_start_mode", "?")

            if viewer is not None and viewer_update_fn is not None:
                viewer.reset_episode(start_pos, target_pos)
                viewer_update_fn(viewer, cfg, info)

            ep_reward = 0.0
            ep_length = 0
            prev_pos = start_pos.copy()
            stop_reason = "max_steps"

            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                if sleep > 0:
                    time.sleep(sleep)
                ep_reward += float(reward)
                ep_length += 1

                curr_pos = env.unwrapped.current_pos.copy()
                if viewer is not None:
                    viewer.update_agent(curr_pos)
                    viewer.draw_path_segment(prev_pos, curr_pos)
                prev_pos = curr_pos

                if terminated or truncated:
                    stop_reason = _get_stop_reason(info, terminated, truncated)
                    final_dist = float(info["distance"])
                    final_success = bool(info["is_success"])
                    exp_pl = float(info.get("expected_path_length", float("nan")))
                    act_pl = float(info.get("actual_path_length", float("nan")))
                    eff_pct = float(info.get("path_efficiency_percent", float("nan")))

                    writer.writerow({
                        "episode": ep_idx,
                        "start_mode": eff_mode,
                        "start_x": float(start_pos[0]), "start_y": float(start_pos[1]), "start_z": float(start_pos[2]),
                        "target_x": float(target_pos[0]), "target_y": float(target_pos[1]), "target_z": float(target_pos[2]),
                        "is_success": int(final_success),
                        "cumulative_reward": ep_reward,
                        "episode_length": ep_length,
                        "final_distance": final_dist,
                        "expected_path_length": exp_pl,
                        "actual_path_length": act_pl,
                        "path_efficiency_percent": eff_pct,
                        "stop_reason": stop_reason,
                    })
                    results.append({
                        "episode": ep_idx,
                        "is_success": final_success,
                        "cumulative_reward": ep_reward,
                        "episode_length": ep_length,
                        "final_distance": final_dist,
                        "stop_reason": stop_reason,
                        "start_pos": start_pos.copy(),
                        "expected_path_length": exp_pl,
                        "actual_path_length": act_pl,
                        "path_efficiency_percent": eff_pct,
                    })
                    break

    return results


def run_static_episodes(
    model: "BaseAlgorithm",
    env: gym.Env,
    cfg: Config,
    num_episodes: int,
    deterministic: bool = True,
    sleep: float = 0.0,
    viewer=None,
    viewer_update_fn=None,
    output_dir: Path | None = None,
) -> List[dict]:
    """Run static-target episodes with per-step trajectory logging."""
    results: List[dict] = []
    if output_dir is None:
        output_dir = get_project_root() / "Data" / "Prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(1, num_episodes + 1):
        obs, info = env.reset(seed=42 + ep_idx)
        unwrapped = env.unwrapped
        start_pos = unwrapped.current_pos.copy()
        target_pos = unwrapped.target_pos.copy()
        eff_mode = getattr(unwrapped, "_start_mode", "?")

        if viewer is not None and viewer_update_fn is not None:
            viewer.reset_episode(start_pos, target_pos)
            viewer_update_fn(viewer, cfg, info)

        traj = Trajectory()
        step_count = 0
        terminated = False
        truncated = False
        prev_pos = start_pos.copy()
        stop_reason = "max_steps"

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if sleep > 0:
                time.sleep(sleep)
            step_count += 1

            curr_pos = env.unwrapped.current_pos.copy()
            if viewer is not None:
                viewer.update_agent(curr_pos)
                viewer.draw_path_segment(prev_pos, curr_pos)
            prev_pos = curr_pos

            traj.add(
                step=step_count,
                position=obs[:3],
                distance_to_target=float(info["distance"]),
                action=action,
                reward=reward,
                is_success=bool(info["is_success"]),
                is_collision=bool(info.get("obstacle_collision", False)),
                is_out_of_workspace=bool(info.get("out_of_workspace", False)),
            )

        stop_reason = _get_stop_reason(info, terminated, truncated)

        traj_csv = output_dir / f"trajectory_static_ep_{ep_idx:03d}.csv"
        traj.to_csv(traj_csv)
        print(f"  [INFO] Trajectory saved: {traj_csv}")

        nan = float("nan")
        results.append({
            "episode": ep_idx,
            "is_success": bool(info["is_success"]),
            "cumulative_reward": sum(w.reward for w in traj.waypoints),
            "episode_length": step_count,
            "final_distance": float(info["distance"]),
            "start_pos": start_pos.copy(),
            "expected_path_length": float(info.get("expected_path_length", nan)),
            "actual_path_length": float(info.get("actual_path_length", nan)),
            "stop_reason": stop_reason,
        })

        metadata = {
            "episode": ep_idx,
            "final_distance": float(info["distance"]),
            "final_success": bool(info["is_success"]),
            "step_count": step_count,
            "expected_path_length": float(info.get("expected_path_length", nan)),
            "actual_path_length": float(info.get("actual_path_length", nan)),
            "stop_reason": stop_reason,
        }
        traj_json = output_dir / f"trajectory_static_ep_{ep_idx:03d}.json"
        traj.to_json(traj_json, metadata=metadata)

    return results


def _get_stop_reason(info: dict, terminated: bool, truncated: bool) -> str:
    if terminated:
        if bool(info.get("is_success", False)):
            return "success"
        if bool(info.get("obstacle_collision", False)):
            return "collision"
        if bool(info.get("out_of_workspace", False)):
            return "workspace_violation"
        return "terminated"
    if truncated:
        return "max_steps"
    return "unknown"


# --------------------------------------------------------------------------- #
# Summary printing
# --------------------------------------------------------------------------- #

def print_episode_summary(results: List[dict], num_episodes: int) -> None:
    """Print a compact table of episode results."""
    print()
    print(f"{'Ep':>4} | {'R':>9} | {'Steps':>5} | {'Succ':>4} | {'Stop':>20} | {'Dist':>8}")
    print("-" * 70)
    for r in results:
        stop = r.get("stop_reason", "?")[:20]
        print(
            f"{r['episode']:>4} | "
            f"{r['cumulative_reward']:>9.1f} | "
            f"{r['episode_length']:>5} | "
            f"{'Y' if r['is_success'] else 'N':>4} | "
            f"{stop:>20} | "
            f"{r['final_distance']:>8.4f}"
        )
    print()

    successes = sum(1 for r in results if r["is_success"])
    print(f"  Success rate: {successes}/{num_episodes} = {successes/num_episodes*100:.0f}%")


def print_full_summary(
    results: List[dict],
    num_episodes: int,
    output_dir: Path | None = None,
) -> None:
    episode_rewards = [r["cumulative_reward"] for r in results]
    episode_lengths = [r["episode_length"] for r in results]
    episode_successes = [r["is_success"] for r in results]
    episode_distances = [r["final_distance"] for r in results]
    episode_exp_pl = [r.get("expected_path_length", float("nan")) for r in results]
    episode_act_pl = [r.get("actual_path_length", float("nan")) for r in results]

    print_episode_summary(results, num_episodes)

    if output_dir is not None:
        csv_path = output_dir / f"summary_N_{num_episodes}.csv"
        print_prediction_summary(
            episode_rewards, episode_lengths, episode_successes, episode_distances,
            csv_path, episode_exp_pl, episode_act_pl,
        )


# --------------------------------------------------------------------------- #
# Viewer helpers (optional PyBullet)
# --------------------------------------------------------------------------- #

def build_viewer(cfg: Config, gui: bool):
    """Create a FrameViewer. Returns None if PyBullet unavailable or gui=False."""
    try:
        from drl_pathplanning.pybullet import FrameViewer, FrameViewerSceneSpec, build_viz_config, HAVE_PYBULLET
    except ImportError:
        return None

    if not gui or not HAVE_PYBULLET:
        return None

    viz_cfg = build_viz_config(cfg, gui=gui)
    show_table = cfg.table.enabled

    scene = FrameViewerSceneSpec(
        workspace_min=cfg.workspace.min_np.tolist(),
        workspace_max=cfg.workspace.max_np.tolist(),
        target_region_min=cfg.target_region.min_np.tolist(),
        target_region_max=cfg.target_region.max_np.tolist(),
        table_center=cfg.table.center,
        table_half_extent=cfg.table.half_extent_np.tolist(),
        table_color=cfg.table.color,
        box_center=None,
        box_half_extent=None,
        box_color=(
            cfg.obstacle.visual.color
            if (cfg.obstacle.enabled and getattr(cfg.obstacle, "visual", None))
            else [0.1, 0.1, 0.1, 1.0]
        ),
        _obstacle_cfg=cfg.obstacle,
        gui=True,
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
    return FrameViewer.from_scene(scene)


def update_viewer_obstacle(viewer, cfg: Config, info: dict) -> None:
    """Update viewer obstacle to match current episode geometry."""
    try:
        from drl_pathplanning.pybullet import FrameViewer
    except ImportError:
        return

    obs_center = info.get("obstacle_center")
    obs_size = info.get("obstacle_size")
    if obs_center is None or obs_size is None:
        return
    viewer.update_obstacle(
        box_center=obs_center,
        box_half_extent=np.asarray(obs_size, dtype=np.float32) / 2.0,
        box_color=(
            cfg.obstacle.visual.color
            if (cfg.obstacle.enabled and getattr(cfg.obstacle, "visual", None))
            else None
        ),
        obstacle_cfg=cfg.obstacle,
    )


# --------------------------------------------------------------------------- #
# Shared argument parser
# --------------------------------------------------------------------------- #

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared across all predict scripts."""
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config [default: algorithm-specific default]",
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Training run directory path",
    )
    parser.add_argument(
        "--model", type=str, default="",
        help="Custom model filename inside run/model/ (e.g. checkpoint_t5000000.zip)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes [default: 10]",
    )
    parser.add_argument(
        "--mode", type=str, default="static",
        choices=["random", "static"],
        help="Evaluation mode [default: static]",
    )
    parser.add_argument(
        "--gui", type=str, default="true",
        choices=["true", "false"],
        help="Enable PyBullet GUI [default: true]",
    )
    parser.add_argument(
        "--show", type=str, default="true",
        choices=["true", "false"],
        help="Print per-episode summary [default: true]",
    )
    parser.add_argument(
        "--deterministic", type=str, default="true",
        choices=["true", "false"],
        help="Use deterministic policy [default: true]",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.1,
        help="Sleep between steps in GUI [default: 0.1]",
    )
    parser.add_argument(
        "--start-mode", type=str, default="config",
        choices=["config", "fixed", "random"],
        help="Start mode [default: config]",
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed target position",
    )


def parse_common_args(args: argparse.Namespace) -> dict:
    """Parse common args and resolve effective values."""
    return {
        "gui": args.gui.lower() == "true",
        "show": args.show.lower() == "true",
        "deterministic": args.deterministic.lower() == "true",
        "sleep": args.sleep,
        "mode": args.mode,
        "num_episodes": args.episodes,
        "start_mode": args.start_mode,
        "target_pos": tuple(args.target) if args.target else None,
        "custom_model": args.model or "",
        "run_path": Path(args.run).expanduser().resolve() if args.run else None,
        "config_path": args.config,
    }
