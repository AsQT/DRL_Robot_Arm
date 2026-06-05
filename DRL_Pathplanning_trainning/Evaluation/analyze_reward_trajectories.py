"""
Analyze reward trajectories — sanity-check the corridor-based reward function.

Loads config/environment.yaml, creates a RewardCalculator, and simulates five
contrasting trajectories to verify the reward landscape:

  A. low_straight_through_box      — toward target, through corridor at low z
  B. corridor_climb_cross_target   — climb at padding zone, cross at required_z, descend
  C. high_too_early_then_target   — climbs very high early, then to target
  D. far_detour_to_target         — goes around box at safe height, succeeds
  E. high_but_not_to_target       — climbs high, drifts away from target

Expected ordering (cumulative reward):
  B > D (+20 gap) > C > E > A

Usage
-----
    python Evaluation/analyze_reward_trajectories.py --config config/environment.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Resolve project root so we can import drl_pathplanning
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drl_pathplanning.gymnasium.config import load_config
from drl_pathplanning.gymnasium.reward import RewardCalculator
from drl_pathplanning.geometry.collision_geometry import GeometryCollisionChecker


# --------------------------------------------------------------------------- #
# Geometry from environment.yaml
# --------------------------------------------------------------------------- #

START_POS  = np.array([0.350, -0.330, 0.100], dtype=np.float32)
TARGET_POS = np.array([0.030, -0.535, 0.110], dtype=np.float32)

BOX_CENTER = np.array([0.1450, -0.550, 0.080], dtype=np.float32)
BOX_SIZE   = np.array([0.100,  0.100,  0.100], dtype=np.float32)  # overwritten after load_config

ACTION_STEP = 0.01
GOAL_THRESH = 0.03

# Pre-compute corridor geometry constants
_PATH_DELTA = TARGET_POS[:2] - START_POS[:2]
_PATH_LEN   = float(np.linalg.norm(_PATH_DELTA))
_PATH_DIR   = (_PATH_DELTA / _PATH_LEN).astype(np.float32)
_PATH_PERP  = np.array([-_PATH_DIR[1], _PATH_DIR[0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Trajectory definitions
# --------------------------------------------------------------------------- #

class TrajectoryPoint(NamedTuple):
    pos: np.ndarray
    action: np.ndarray


def _linear_interp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    return [np.array(a + tt * (b - a), dtype=np.float32) for tt in t]


def _compute_actions(positions: list[np.ndarray]) -> list[np.ndarray]:
    actions = []
    for i in range(len(positions) - 1):
        delta = positions[i + 1] - positions[i]
        action = delta / ACTION_STEP
        actions.append(action.astype(np.float32))
    return actions


# --------------------------------------------------------------------------- #
# A. low_straight_through_box
# --------------------------------------------------------------------------- #

def make_trajectory_A() -> list[TrajectoryPoint]:
    """
    Straight from start to target at low z (same as start z).
    Passes through the obstacle corridor at low z → low_clearance penalty, possible collision.
    """
    end = TARGET_POS.copy()
    end[2] = START_POS[2]   # keep low z
    points = _linear_interp(START_POS, end, 60)
    actions = _compute_actions(points)
    return [TrajectoryPoint(pos=p, action=a) for p, a in zip(points, actions)]


# --------------------------------------------------------------------------- #
# B. corridor_climb_cross_target
# --------------------------------------------------------------------------- #

def _safe_z() -> float:
    return float(BOX_CENTER[2] + BOX_SIZE[2] / 2.0 + 0.05)   # required_clearance_z


def make_trajectory_B() -> list[TrajectoryPoint]:
    """
    Optimal path:
      1. Move toward box at low z.
      2. Climb to required_z at approach_padding zone.
      3. Cross box at required_z.
      4. After cross_end, descend to target.
    """
    safe = _safe_z()

    # Find a point at the corridor entrance (approach zone), same XY as start
    # We simply go straight toward target and climb at the approach zone.
    # Approximate the approach zone: go 80% of the way in XY, then climb
    bc_xy = BOX_CENTER[:2]

    # Climb-start point: just before approach zone, at safe z
    # We climb at a point slightly before the box in XY
    climb_xy = bc_xy + (-0.10) * _PATH_DIR

    # Pre-climb: go toward climb_xy at low z
    leg1 = _linear_interp(START_POS, np.array([climb_xy[0], climb_xy[1], START_POS[2]], dtype=np.float32), 25)
    # Climb: go up at climb_xy
    leg2 = _linear_interp(
        leg1[-1],
        np.array([climb_xy[0], climb_xy[1], safe], dtype=np.float32),
        15,
    )
    # Cross: move through corridor at safe z, extending well past cross_end (0.17)
    # From climb_xy to a point just past the box
    cross_end_xy = bc_xy + (0.22) * _PATH_DIR   # well past cross_end
    leg3 = _linear_interp(
        leg2[-1],
        np.array([cross_end_xy[0], cross_end_xy[1], safe], dtype=np.float32),
        30,
    )
    # Descend to target
    leg4 = _linear_interp(leg3[-1], TARGET_POS, 30)

    all_positions = leg1 + leg2[1:] + leg3[1:] + leg4[1:]
    actions = _compute_actions(all_positions)
    return [TrajectoryPoint(pos=p, action=a) for p, a in zip(all_positions, actions)]


# --------------------------------------------------------------------------- #
# C. high_too_early_then_target
# --------------------------------------------------------------------------- #

def make_trajectory_C() -> list[TrajectoryPoint]:
    """
    Climbs very high (0.28) near the start, then goes to target.
    Penalised by path_length, extra action, extra time.
    """
    high_z = 0.28
    mid_xy = (START_POS[:2] + TARGET_POS[:2]) / 2.0

    # Go up at start position
    leg1 = _linear_interp(START_POS, np.array([START_POS[0], START_POS[1], high_z], dtype=np.float32), 20)
    # Move at high altitude toward mid-point
    leg2 = _linear_interp(leg1[-1], np.array([mid_xy[0], mid_xy[1], high_z], dtype=np.float32), 25)
    # Descend toward target
    leg3 = _linear_interp(leg2[-1], TARGET_POS, 30)

    all_positions = leg1 + leg2[1:] + leg3[1:]
    actions = _compute_actions(all_positions)
    return [TrajectoryPoint(pos=p, action=a) for p, a in zip(all_positions, actions)]


# --------------------------------------------------------------------------- #
# D. far_detour_to_target
# --------------------------------------------------------------------------- #

def make_trajectory_D() -> list[TrajectoryPoint]:
    """
    Goes around the box to the left at safe height, then to target.
    Longer path than B but collision-free.
    """
    safe_z = _safe_z()
    clear_x = -0.18   # to the left of table (table x_min = -0.135)

    p0 = START_POS.copy()
    p0[2] = safe_z

    step1 = _linear_interp(START_POS, p0, 15)
    p_left = np.array([clear_x, START_POS[1], safe_z], dtype=np.float32)
    step2 = _linear_interp(p0, p_left, 20)
    p_targ_y = np.array([clear_x, TARGET_POS[1], safe_z], dtype=np.float32)
    step3 = _linear_interp(p_left, p_targ_y, 20)
    step4 = _linear_interp(p_targ_y, TARGET_POS, 25)

    all_positions = step1 + step2[1:] + step3[1:] + step4[1:]
    actions = _compute_actions(all_positions)
    return [TrajectoryPoint(pos=p, action=a) for p, a in zip(all_positions, actions)]


# --------------------------------------------------------------------------- #
# E. high_but_not_to_target
# --------------------------------------------------------------------------- #

def make_trajectory_E() -> list[TrajectoryPoint]:
    """
    Goes high but drifts away from the target.
    Should have the lowest non-collided reward.
    """
    high_z = 0.28
    # Go upward first
    leg1 = _linear_interp(START_POS, np.array([START_POS[0], START_POS[1], high_z], dtype=np.float32), 20)
    # Then move laterally away from target (positive y direction)
    far_lateral = np.array([0.42, 0.0, high_z], dtype=np.float32)
    leg2 = _linear_interp(leg1[-1], far_lateral, 40)

    all_positions = leg1 + leg2[1:]
    actions = _compute_actions(all_positions)
    return [TrajectoryPoint(pos=p, action=a) for p, a in zip(all_positions, actions)]


TRAJECTORY_BUILDERS = {
    "low_straight_through_box": make_trajectory_A,
    "corridor_climb_cross_target": make_trajectory_B,
    "high_too_early_then_target": make_trajectory_C,
    "far_detour_to_target":       make_trajectory_D,
    "high_but_not_to_target":    make_trajectory_E,
}


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def build_collision_checker(cfg) -> GeometryCollisionChecker:
    return GeometryCollisionChecker(
        check_box=cfg.collision.enabled and cfg.obstacle.enabled,
        obstacle_center=cfg.obstacle.center_np if cfg.obstacle.enabled else None,
        obstacle_half_extent=cfg.obstacle.half_extent_np if cfg.obstacle.enabled else None,
        collision_half_extent=cfg.obstacle.collision_half_extent_np if cfg.obstacle.enabled else None,
        obstacle_name="box",
        ws_min=cfg.workspace.min_np,
        ws_max=cfg.workspace.max_np,
        table_center=cfg.table.center_np,
        table_half_extent=cfg.table.half_extent_np,
    )


def simulate_trajectory(
    name: str,
    trajectory: list[TrajectoryPoint],
    calc: RewardCalculator,
    collision_checker: GeometryCollisionChecker,
    target_pos: np.ndarray,
    box_center: np.ndarray,
    box_size: np.ndarray,
) -> dict:
    """Feed every point through RewardCalculator and accumulate components."""
    cumulative = 0.0
    total_target_progress   = 0.0
    total_forward_progress = 0.0
    total_z_climb          = 0.0
    total_z_hold           = 0.0
    max_low_clearance      = 0.0
    total_clearance_bonus  = 0.0
    total_corridor_clear   = 0.0
    total_path_length      = 0.0
    total_detour           = 0.0
    total_action_penalty   = 0.0
    total_smoothness       = 0.0
    total_time             = 0.0

    success   = False
    collision = False
    obstacle_cleared = False
    terminated = False

    prev_pos = trajectory[0].pos.copy()
    step = 0

    for point in trajectory:
        if terminated:
            break

        curr_pos = point.pos
        action   = point.action

        dist_to_target = float(np.linalg.norm(target_pos - curr_pos))

        cr = collision_checker.check(prev_pos, curr_pos)
        is_collision = (
            cr.collides
            and cr.collision_type in ("box_point", "box_segment", "table_point", "table_segment")
        )
        is_out_of_workspace = cr.collision_type == "workspace"
        is_success = dist_to_target < GOAL_THRESH

        if is_success or is_collision or is_out_of_workspace:
            terminated = True
            if is_success:
                success = True
            if is_collision:
                collision = True

        reward, info = calc.compute(
            prev_pos=prev_pos,
            action=action,
            curr_pos=curr_pos,
            target_pos=target_pos,
            obstacle_center=box_center,
            obstacle_size=box_size,
            is_success=is_success,
            is_collision=is_collision,
            is_out_of_workspace=is_out_of_workspace,
        )

        cumulative            += info["total"]
        total_target_progress += info["target_progress"]
        total_forward_progress += info["forward_progress"]
        total_z_climb         += info["z_climb"]
        total_z_hold          += info["z_hold"]
        max_low_clearance      = min(max_low_clearance, info["low_clearance"])
        total_clearance_bonus += info["clearance_bonus"]
        total_corridor_clear  += info["corridor_clear_bonus"]
        total_path_length     += info["path_length"]
        total_detour          += info["detour"]
        total_action_penalty  += info["action"]
        total_smoothness      += info["smoothness"]
        total_time            += info["time"]

        obstacle_cleared = info["obstacle_cleared"]

        prev_pos = curr_pos.copy()
        step += 1

    return {
        "trajectory_name":        name,
        "cumulative_reward":      cumulative,
        "success":              success,
        "collision":             collision,
        "obstacle_cleared":     obstacle_cleared,
        "total_target_progress":  total_target_progress,
        "total_forward_progress": total_forward_progress,
        "total_z_climb":          total_z_climb,
        "total_z_hold":           total_z_hold,
        "max_low_clearance_penalty": max_low_clearance,
        "total_clearance_bonus":  total_clearance_bonus,
        "total_corridor_clear_bonus": total_corridor_clear,
        "total_path_length_penalty": total_path_length,
        "total_detour_penalty":   total_detour,
        "total_action_penalty":   total_action_penalty,
        "total_smoothness_penalty": total_smoothness,
        "steps":                  step,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _sep() -> None:
    print("-" * 62)


def print_result(r: dict) -> None:
    _sep()
    print(f"  {r['trajectory_name']}")
    _sep()
    for k, v in [
        ("cumulative_reward",          r["cumulative_reward"]),
        ("success",                   r["success"]),
        ("collision",                 r["collision"]),
        ("obstacle_cleared",          r["obstacle_cleared"]),
        ("total_target_progress",      r["total_target_progress"]),
        ("total_forward_progress",     r["total_forward_progress"]),
        ("total_z_climb",             r["total_z_climb"]),
        ("total_z_hold",              r["total_z_hold"]),
        ("max_low_clearance_penalty", r["max_low_clearance_penalty"]),
        ("total_clearance_bonus",     r["total_clearance_bonus"]),
        ("total_corridor_clear_bonus",r["total_corridor_clear_bonus"]),
        ("total_path_length_penalty", r["total_path_length_penalty"]),
        ("total_detour_penalty",      r["total_detour_penalty"]),
        ("total_action_penalty",      r["total_action_penalty"]),
        ("total_smoothness_penalty",  r["total_smoothness_penalty"]),
        ("steps",                     r["steps"]),
    ]:
        if isinstance(v, bool):
            val = "True" if v else "False"
        elif isinstance(v, float):
            val = f"{v:.4f}" if abs(v) >= 1e-4 else "0.0000"
        else:
            val = str(v)
        print(f"  {k:<30} {val:>16}")
    print()


def print_summary_table(results: list[dict]) -> None:
    headers = [
        "trajectory_name",
        "cumulative_reward",
        "success",
        "collision",
        "obstacle_cleared",
        "total_target_progress",
        "total_forward_progress",
        "total_z_climb",
        "total_z_hold",
        "max_low_clearance_penalty",
        "total_clearance_bonus",
        "total_corridor_clear_bonus",
        "total_path_length_penalty",
        "total_detour_penalty",
        "total_action_penalty",
        "total_smoothness_penalty",
    ]
    cw = 14
    sep = "  " + "=" * (30 + cw * len(headers))
    print(sep)
    hline = "  " + f"{'trajectory':<28}"
    for h in headers[1:]:
        hline += f" {h:>{cw}}"
    print(hline)
    print(sep)

    for r in results:
        row = f"  {r['trajectory_name']:<28}"
        for h in headers[1:]:
            v = r[h]
            if isinstance(v, bool):
                s = "True" if v else "False"
            elif isinstance(v, float):
                s = f"{v:.2f}"
            else:
                s = str(v)
            row += f" {s:>{cw}}"
        print(row)
    print(sep)
    print()


def check_ordering_and_gaps(results: list[dict]) -> bool:
    """Verify: B > D+20, D > E, B > A, A < 0."""
    print("\n=== EXPECTATION CHECK ===")
    by_name = {r["trajectory_name"]: r for r in results}

    a = by_name.get("low_straight_through_box")
    b = by_name.get("corridor_climb_cross_target")
    c = by_name.get("high_too_early_then_target")
    d = by_name.get("far_detour_to_target")
    e = by_name.get("high_but_not_to_target")

    checks = []

    if b is not None and d is not None:
        ok = b["cumulative_reward"] > d["cumulative_reward"] + 20.0
        gap = b["cumulative_reward"] - d["cumulative_reward"]
        checks.append((f"corridor_climb (B) > far_detour (D) + 20  [gap={gap:.1f}]", ok))

    if d is not None and e is not None:
        ok = d["cumulative_reward"] > e["cumulative_reward"]
        checks.append((f"far_detour (D) > high_not_target (E)", ok))

    if c is not None and d is not None:
        ok = d["cumulative_reward"] > c["cumulative_reward"]
        checks.append((f"far_detour (D) > high_early (C)", ok))

    if b is not None and a is not None:
        ok = b["cumulative_reward"] > a["cumulative_reward"]
        checks.append((f"corridor_climb (B) > low_straight (A)", ok))

    if a is not None:
        ok = a["cumulative_reward"] < 0.0 or a["collision"]
        checks.append((f"low_straight (A) < 0 or collided", ok))

    if b is not None and c is not None:
        ok = b["cumulative_reward"] > c["cumulative_reward"]
        checks.append((f"corridor_climb (B) > high_early (C)", ok))

    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {label}")

    if all_passed:
        print("\n  All ordering checks PASSED.")
    else:
        print("\n  Some ordering checks FAILED — review reward coefficients.")

    return all_passed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check corridor-based reward with 5 hand-crafted trajectories."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to environment.yaml (default: auto-detect).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    # Override BOX_SIZE / BOX_CENTER from config when obstacle is enabled
    if cfg.obstacle.enabled:
        BOX_SIZE[0] = cfg.obstacle.size[0]
        BOX_SIZE[1] = cfg.obstacle.size[1]
        BOX_SIZE[2] = cfg.obstacle.size[2]
        BOX_CENTER[0] = cfg.obstacle.center[0]
        BOX_CENTER[1] = cfg.obstacle.center[1]
        BOX_CENTER[2] = cfg.obstacle.center[2]

    safe_z = float(BOX_CENTER[2] + BOX_SIZE[2] / 2.0 + cfg.reward.obstacle_clearance)
    print("=" * 62)
    print("  Corridor Reward Trajectory Analysis")
    print("=" * 62)
    print(f"  Config:  {args.config or 'default'}")
    print(f"  Start:   {START_POS.tolist()}")
    print(f"  Target:  {TARGET_POS.tolist()}")
    print(f"  Box:     center={BOX_CENTER.tolist()}, size={BOX_SIZE.tolist()}")
    print(f"  Box top z: {float(BOX_CENTER[2] + BOX_SIZE[2] / 2.0):.4f}")
    print(f"  Required clearance z: {safe_z:.4f}")
    print("=" * 62)
    print()

    calc = RewardCalculator(reward_cfg=cfg.reward)
    collision_checker = build_collision_checker(cfg)

    results = []
    for name, builder in TRAJECTORY_BUILDERS.items():
        traj = builder()
        calc.reset(
            obstacle_center=BOX_CENTER,
            obstacle_size=BOX_SIZE,
            target_pos=TARGET_POS,
            start_pos=traj[0].pos,
        )
        r = simulate_trajectory(
            name=name,
            trajectory=traj,
            calc=calc,
            collision_checker=collision_checker,
            target_pos=TARGET_POS,
            box_center=BOX_CENTER,
            box_size=BOX_SIZE,
        )
        results.append(r)
        print_result(r)

    print_summary_table(results)
    check_ordering_and_gaps(results)


if __name__ == "__main__":
    main()
