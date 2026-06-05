"""
Trajectory comparison test for the strategy-based reward function.

Uses real Gymnasium env (CartesianPathPlanningEnv) with fixed scene geometry,
then runs scripted trajectories through env.reset() + env.step() to verify:
  1. Correct trajectory has collision=False, success=True
  2. Incorrect trajectories are properly penalised
  3. Reward magnitude is reasonable (per-step reward ~0.01-1.0 scale)

Usage::

    python Evaluation/test_reward_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv
from drl_pathplanning.gymnasium.config import Config


# =============================================================================
# Fixed test scene geometry
# =============================================================================
ACTION_STEP = 0.01

START_POS = np.array([0.350, -0.550, 0.100], dtype=np.float32)
TARGET_POS = np.array([-0.050, -0.550, 0.100], dtype=np.float32)

OB_CENTER = np.array([0.145, -0.550, 0.130], dtype=np.float32)
OB_SIZE = np.array([0.100, 0.100, 0.100], dtype=np.float32)

# Expected collision AABB (half_extent + safety_margin=0.01):
#   x=[0.085,0.205], y=[-0.610,-0.490], z=[0.070,0.190]
COL_LB = OB_CENTER - (OB_SIZE / 2.0 + 0.01)
COL_UB = OB_CENTER + (OB_SIZE / 2.0 + 0.01)

# Expected safe zone (half_extent + collision_visual.padding):
#   padding = [0.05, 0.05, 0.03]
#   x=[0.045,0.245], y=[-0.650,-0.450], z_top=0.160
SAFE_PAD = np.array([0.05, 0.05, 0.03], dtype=np.float32)
SAFE_LB_XY = OB_CENTER[:2] - (OB_SIZE[:2] / 2.0 + SAFE_PAD[:2])
SAFE_UB_XY = OB_CENTER[:2] + (OB_SIZE[:2] / 2.0 + SAFE_PAD[:2])
SAFE_TOP_Z = 0.030 + OB_SIZE[2] + SAFE_PAD[2]  # = 0.160


# =============================================================================
# Config factory
# =============================================================================

def make_config() -> Config:
    """Create a Config with fixed scene geometry for testing."""
    cfg = Config.from_yaml(_SCRIPT_DIR.parent / "config" / "environment.yaml")

    cfg.start.mode = "fixed"
    cfg.start.fixed_position = START_POS.tolist()

    cfg.target_region.mode = "fixed"
    cfg.target_region.fixed_position = TARGET_POS.tolist()

    cfg.obstacle.enabled = True
    cfg.obstacle.mode = "fixed"
    cfg.obstacle.center = OB_CENTER.tolist()
    cfg.obstacle.size = OB_SIZE.tolist()
    cfg.obstacle.size_random.enabled = False

    return cfg


# =============================================================================
# Trajectory stepper using real env
# =============================================================================

def _scalar(v):
    """Convert numpy/Python scalar to plain float."""
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "item"):
        return float(v.item())
    if hasattr(v, "__float__"):
        return float(v)
    return 0.0


def run_trajectory(
    env: CartesianPathPlanningEnv,
    goals: list[np.ndarray],
    max_steps: int = 500,
    debug: bool = False,
) -> dict:
    """
    Run a scripted trajectory through env.reset() + env.step().

    Returns a dict with:
      - steps, success, collision, out_of_workspace
      - accumulated reward components (from info["reward_components"])
      - per-step stats (min/max/mean) for key components
      - phase_sequence
    """
    obs, info = env.reset()
    env.unwrapped.set_target(TARGET_POS)

    goal_idx = 0
    pos = env.unwrapped._current_pos.copy()

    acc = {k: 0.0 for k in [
        "subgoal_progress", "target_progress", "strategy",
        "low_clearance", "detour", "time", "action", "smoothness",
        "success", "collision", "workspace", "total"
    ]}
    acc["steps"] = 0
    acc["success"] = 0
    acc["collision"] = 0
    acc["out_of_workspace"] = 0

    # Per-step tracking for magnitude audit
    step_rewards = []
    step_subgoal = []
    step_strategy = []

    phases = []

    terminated = False
    truncated = False

    while acc["steps"] < max_steps and not terminated and not truncated:
        # Advance goal when within threshold
        if goal_idx < len(goals):
            subgoal = goals[goal_idx]
            if np.linalg.norm(pos - subgoal) < 0.003:
                goal_idx += 1
                subgoal = goals[goal_idx] if goal_idx < len(goals) else TARGET_POS
        else:
            subgoal = TARGET_POS

        # Compute action toward subgoal — pass NORMALISED action (the env scales by action_step)
        delta = subgoal - pos
        dist = float(np.linalg.norm(delta))
        action = (delta / dist) if dist > 1e-6 else np.zeros(3, dtype=np.float32)

        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        pos = env.unwrapped._current_pos.copy()

        # Extract terminal flags from real env info
        is_success = bool(info.get("is_success", False))
        is_collision = bool(info.get("is_collision", False))
        is_workspace = bool(info.get("out_of_workspace", False))

        # Phase from info["phase"] (set in cartesian_frame_env.py step())
        phase = info.get("phase", "unknown")

        # Get reward components from info["reward_components"] dict
        reward_components = info.get("reward_components", {})

        # Debug: show first few steps + phase transitions
        prev_phase = phases[-1] if phases else None
        if debug and (acc["steps"] < 5 or phase != prev_phase):
            inside_safe = (
                pos[0] >= SAFE_LB_XY[0] and pos[0] <= SAFE_UB_XY[0]
                and pos[1] >= SAFE_LB_XY[1] and pos[1] <= SAFE_UB_XY[1]
            )
            print(
                f"  [step={acc['steps']:3d}] pos=[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}]  "
                f"phase={phase:<10s}  inside_safe_xy={inside_safe}  "
                f"goal=[{subgoal[0]:+.4f},{subgoal[1]:+.4f},{subgoal[2]:+.4f}]  "
                f"reward={_scalar(reward):+.4f}"
            )

        # Accumulate
        acc["steps"] += 1
        for k in acc:
            if k in reward_components:
                acc[k] += _scalar(reward_components[k])

        if is_success:
            acc["success"] = 1
        if is_collision:
            acc["collision"] = 1
        if is_workspace:
            acc["out_of_workspace"] = 1

        phases.append(phase)
        step_rewards.append(_scalar(reward))
        step_subgoal.append(_scalar(reward_components.get("subgoal_progress", 0.0)))
        step_strategy.append(_scalar(reward_components.get("strategy", 0.0)))

    acc["phase_sequence"] = phases

    # Per-step magnitude audit
    if step_rewards:
        acc["step_reward_min"] = min(step_rewards)
        acc["step_reward_max"] = max(step_rewards)
        acc["step_reward_mean"] = sum(step_rewards) / len(step_rewards)
    else:
        acc["step_reward_min"] = 0.0
        acc["step_reward_max"] = 0.0
        acc["step_reward_mean"] = 0.0

    if step_subgoal:
        acc["step_subgoal_max_abs"] = max(abs(x) for x in step_subgoal)
    else:
        acc["step_subgoal_max_abs"] = 0.0

    if step_strategy:
        acc["step_strategy_max_abs"] = max(abs(x) for x in step_strategy)
    else:
        acc["step_strategy_max_abs"] = 0.0

    return acc


# =============================================================================
# Trajectory definitions
# =============================================================================

def build_trajectories(approach_xy, exit_xy, safe_z):
    """Build goal lists for each trajectory using computed subgoals."""
    goals = {
        "A. correct (approach->climb->traverse->target)": [
            np.array([approach_xy[0], approach_xy[1], START_POS[2]], dtype=np.float32),
            np.array([approach_xy[0], approach_xy[1], safe_z], dtype=np.float32),
            np.array([exit_xy[0], exit_xy[1], safe_z], dtype=np.float32),
        ],
        "B. straight_low": [],  # Go straight to target at low z
        "C. climb_early": [
            np.array([START_POS[0], START_POS[1], safe_z], dtype=np.float32),
        ],
        "D. go_around": [
            np.array([START_POS[0], -0.350, START_POS[2]], dtype=np.float32),
            np.array([TARGET_POS[0], -0.350, TARGET_POS[2]], dtype=np.float32),
        ],
        "E. enter_low_through_safe_zone": [
            np.array([OB_CENTER[0], OB_CENTER[1], START_POS[2]], dtype=np.float32),
            np.array([TARGET_POS[0], TARGET_POS[1], START_POS[2]], dtype=np.float32),
        ],
    }
    return goals


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("  STRATEGY REWARD TEST  (using real env.step())")
    print("=" * 72)
    print(f"  Collision AABB  : x=[{COL_LB[0]:.3f},{COL_UB[0]:.3f}] "
          f"y=[{COL_LB[1]:.3f},{COL_UB[1]:.3f}] z=[{COL_LB[2]:.3f},{COL_UB[2]:.3f}]")
    print(f"  Safe XY zone   : x=[{SAFE_LB_XY[0]:.3f},{SAFE_UB_XY[0]:.3f}] "
          f"y=[{SAFE_LB_XY[1]:.3f},{SAFE_UB_XY[1]:.3f}]")
    print(f"  Safe top z     : {SAFE_TOP_Z:.3f}")
    print(f"  Start          : {START_POS}")
    print(f"  Target         : {TARGET_POS}")
    print()

    # Create env with fixed config
    cfg = make_config()
    env = CartesianPathPlanningEnv(env_cfg=cfg, start_mode="fixed", start_pos=tuple(START_POS))

    # Reset once to get reward calculator's computed safe zone geometry
    obs, info = env.reset()
    env.unwrapped.set_target(TARGET_POS)

    rc = env.unwrapped._reward_calculator
    sz = rc.get_safe_zone()
    if sz is not None:
        approach_xy = rc._approach_goal_xy
        exit_xy = rc._exit_goal_xy
        safe_z = rc._subgoal_z
        print(f"  Computed approach goal : [{approach_xy[0]:.4f}, {approach_xy[1]:.4f}]")
        print(f"  Computed exit goal    : [{exit_xy[0]:.4f}, {exit_xy[1]:.4f}]")
        print(f"  Computed safe_z       : {safe_z:.4f}")
        print(f"  t_entry={sz['t_entry']:.3f}  t_exit={sz['t_exit']:.3f}")
    else:
        approach_xy = np.array([SAFE_UB_XY[0], START_POS[1]], dtype=np.float32)
        exit_xy = np.array([SAFE_LB_XY[0], START_POS[1]], dtype=np.float32)
        safe_z = SAFE_TOP_Z
        print("  [WARNING] Obstacle not on path — using fallback geometry")
    print()

    trajectories = build_trajectories(approach_xy, exit_xy, safe_z)

    # -------------------------------------------------------------------------
    # Trajectory A: detailed debug output
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("  TRAJECTORY A — DETAILED DEBUG (first 5 steps + phase changes)")
    print("=" * 72)
    print(f"  Goals: approach_xy -> climb_z -> traverse -> target")
    print()

    acc_a = run_trajectory(
        env, trajectories["A. correct (approach->climb->traverse->target)"], debug=True
    )
    print()
    print(f"  A result: steps={acc_a['steps']}  success={acc_a['success']}  "
          f"collision={acc_a['collision']}  "
          f"phases={dict.fromkeys(acc_a['phase_sequence'])}")
    print()

    # -------------------------------------------------------------------------
    # All trajectories
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("  ALL TRAJECTORIES")
    print("=" * 72)
    print(f"  {'Trajectory':<52} {'Total':>9} {'sg_prog':>9} {'str':>9} "
          f"{'low_clr':>9} {'detour':>9} {'succ':>4} {'coll':>4} {'steps':>5}")
    print(f"  {'-'*52} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*4} {'-'*4} {'-'*5}")

    results = {}
    for name, goals in trajectories.items():
        acc = run_trajectory(env, goals, debug=False)
        results[name] = acc

    sorted_res = sorted(results.items(), key=lambda x: x[1]["total"], reverse=True)
    for name, res in sorted_res:
        phases_sample = ",".join(dict.fromkeys(res["phase_sequence"][:4]))
        if len(res["phase_sequence"]) > 4:
            phases_sample += "..."
        print(
            f"  {name:<52} {res['total']:>+9.2f} "
            f"{res['subgoal_progress']:>+9.2f} "
            f"{res['strategy']:>+9.2f} "
            f"{res['low_clearance']:>+9.2f} "
            f"{res['detour']:>+9.2f} "
            f"{res['success']:>4} "
            f"{res['collision']:>4} "
            f"{res['steps']:>5}"
        )

    # -------------------------------------------------------------------------
    # Magnitude audit
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  REWARD MAGNITUDE AUDIT  (Trajectory A)")
    print("=" * 72)
    a = results["A. correct (approach->climb->traverse->target)"]
    print(f"  Per-step reward : min={a['step_reward_min']:+.4f}  "
          f"max={a['step_reward_max']:+.4f}  mean={a['step_reward_mean']:+.4f}")
    print(f"  Subgoal component max abs : {a['step_subgoal_max_abs']:+.4f}")
    print(f"  Strategy component max abs: {a['step_strategy_max_abs']:+.4f}")
    print(f"  Total reward accumulated  : {a['total']:+.4f}")
    print(f"  Steps                    : {a['steps']}")
    print(f"  Phase sequence           : {a['phase_sequence'][:8]}")
    print()

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("  SANITY CHECKS")
    print("=" * 72)

    best_name, best = sorted_res[0]
    worst_name, worst = sorted_res[-1]
    print(f"  Best : {best_name}")
    print(f"         total={best['total']:+.2f}  collision={best['collision']}  "
          f"success={best['success']}")
    print(f"  Worst: {worst_name}")
    print(f"         total={worst['total']:+.2f}  collision={worst['collision']}  "
          f"success={worst['success']}")

    a_res = results["A. correct (approach->climb->traverse->target)"]
    b_res = results["B. straight_low"]
    e_res = results["E. enter_low_through_safe_zone"]
    d_res = results["D. go_around"]

    a_rank = next(i for i, (n, _) in enumerate(sorted_res) if "correct" in n)
    e_rank = next(i for i, (n, _) in enumerate(sorted_res) if "safe_zone" in n)

    print(f"  A rank : #{a_rank+1}/{len(results)}  (target: #1)")
    print(f"  E rank : #{e_rank+1}/{len(results)}  (target: last)")
    print()

    print("  [CHECK] Trajectory A (correct strategy):")
    print(f"    collision={a_res['collision']}  (should be 0)")
    print(f"    success={a_res['success']}  (should be 1)")
    print(f"    low_clearance={a_res['low_clearance']:+.2f}  (should be ~0)")
    print(f"    phases={dict.fromkeys(a_res['phase_sequence'])}")

    print("  [CHECK] Trajectory E (low through safe zone):")
    print(f"    collision={e_res['collision']}  (should be 1)")
    print(f"    low_clearance={e_res['low_clearance']:+.2f}  (should be negative)")

    print("  [CHECK] Trajectory B (straight low):")
    print(f"    collision={b_res['collision']}  (should be 1)")
    print(f"    low_clearance={b_res['low_clearance']:+.2f}  (should be negative)")

    print("  [CHECK] Trajectory D (go around):")
    print(f"    collision={d_res['collision']}  (should be 0)")
    print(f"    detour={d_res['detour']:+.2f}  (should be negative)")

    print()
    print("  Done.")

    env.close()


if __name__ == "__main__":
    main()
