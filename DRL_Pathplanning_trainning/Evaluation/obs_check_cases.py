"""
Audit: Check whether observation obstacle segment correctly reflects
current obstacle geometry per episode across 4 configuration cases.

Case A: obstacle.mode=fixed,  size_random.enabled=false
Case B: obstacle.mode=random, size_random.enabled=false
Case C: obstacle.mode=fixed,  size_random.enabled=true
Case D: obstacle.mode=random, size_random.enabled=true

Usage:
    python Evaluation/obs_check_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from drl_pathplanning.gymnasium.config import Config
from drl_pathplanning.gymnasium.cartesian_frame_env import CartesianPathPlanningEnv

# Observation layout constants (from spaces.py)
OBS_REL_START = 9
OBS_REL_END = 12
OBS_SIZE_START = 12
OBS_SIZE_END = 15
OBS_SEG_START = OBS_REL_START
OBS_SEG_END = OBS_SIZE_END  # 9:15 = 6 values


def clone_cfg_for_case(cfg: Config, case: str) -> Config:
    """Create a modified config for each test case."""
    import copy
    new_cfg = copy.deepcopy(cfg)

    if case == "A":
        new_cfg.obstacle.mode = "fixed"
        new_cfg.obstacle.size_random.enabled = False
    elif case == "B":
        new_cfg.obstacle.mode = "random"
        new_cfg.obstacle.size_random.enabled = False
    elif case == "C":
        new_cfg.obstacle.mode = "fixed"
        new_cfg.obstacle.size_random.enabled = True
    elif case == "D":
        new_cfg.obstacle.mode = "random"
        new_cfg.obstacle.size_random.enabled = True

    return new_cfg


def compute_expected_segment(
    env_obstacle_center: np.ndarray,
    env_obstacle_size: np.ndarray,
    current_pos: np.ndarray,
    ws_range: np.ndarray,
) -> np.ndarray:
    """Replicate the exact formula from normalize_obstacle_info()."""
    half_extent = np.asarray(env_obstacle_size, dtype=np.float32) / 2.0
    rel_obs = (
        (np.asarray(env_obstacle_center, dtype=np.float32)
         - np.asarray(current_pos, dtype=np.float32))
        / np.asarray(ws_range, dtype=np.float32)
    )
    obs_size = (
        np.asarray(half_extent, dtype=np.float32)
        / np.asarray(ws_range, dtype=np.float32)
    )
    return np.concatenate([rel_obs, obs_size]).astype(np.float32)


def run_case(case: str, cfg: Config, n_episodes: int = 3) -> dict:
    """Run n_episodes and collect per-episode observation data."""
    test_cfg = clone_cfg_for_case(cfg, case)

    env = CartesianPathPlanningEnv(env_cfg=test_cfg)
    ws_range = test_cfg.workspace.range_np

    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()

        env_obs_center = info.get("obstacle_center")
        env_obs_size = info.get("obstacle_size")
        cfg_fixed_size = test_cfg.obstacle.size
        cfg_fixed_half = np.array(cfg_fixed_size, dtype=np.float32) / 2.0

        current_pos = obs[:3]
        expected_seg = compute_expected_segment(
            env_obs_center, env_obs_size, current_pos, ws_range
        )

        actual_seg = obs[OBS_SEG_START:OBS_SEG_END].copy()
        seg_match = bool(np.allclose(actual_seg, expected_seg, atol=1e-5))

        seg_diff = actual_seg - expected_seg

        results.append({
            "episode": ep + 1,
            "env_obs_center": (
                env_obs_center.copy() if env_obs_center is not None else None
            ),
            "env_obs_size": (
                env_obs_size.copy() if env_obs_size is not None else None
            ),
            "current_pos": current_pos.copy(),
            "cfg_fixed_size": cfg_fixed_size,
            "expected_seg": expected_seg.copy(),
            "actual_seg": actual_seg.copy(),
            "seg_diff": seg_diff.copy(),
            "seg_match": seg_match,
        })

        # Check per-step stability (steps 0,1,2)
        for _step in range(3):
            action = np.zeros(3, dtype=np.float32)
            obs_next, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        env.close()
        break  # only need first episode for step stability check

    # Re-run for full episode data
    test_cfg2 = clone_cfg_for_case(cfg, case)
    env2 = CartesianPathPlanningEnv(env_cfg=test_cfg2)

    ep_results = []
    prev_center = None
    prev_size = None
    prev_obs_seg = None

    for ep in range(n_episodes):
        obs, info = env2.reset()

        env_obs_center = info.get("obstacle_center")
        env_obs_size = info.get("obstacle_size")
        cfg_fixed_size = test_cfg.obstacle.size
        cfg_fixed_half = np.array(cfg_fixed_size, dtype=np.float32) / 2.0

        current_pos = obs[:3]
        expected_seg = compute_expected_segment(
            env_obs_center, env_obs_size, current_pos, ws_range
        )
        actual_seg = obs[OBS_SEG_START:OBS_SEG_END].copy()
        seg_match = bool(np.allclose(actual_seg, expected_seg, atol=1e-5))
        seg_diff = actual_seg - expected_seg

        center_changed = None
        size_changed = None
        obs_changed = None
        if prev_center is not None and env_obs_center is not None:
            center_changed = bool(not np.allclose(env_obs_center, prev_center, atol=1e-6))
        elif prev_center is None and env_obs_center is not None:
            center_changed = True  # first episode: initial vs actual
        else:
            center_changed = False

        if prev_size is not None and env_obs_size is not None:
            size_changed = bool(not np.allclose(env_obs_size, prev_size, atol=1e-6))
        elif prev_size is None and env_obs_size is not None:
            size_changed = True  # first episode: initial vs actual
        else:
            size_changed = False

        if prev_obs_seg is not None:
            obs_changed = bool(not np.allclose(actual_seg, prev_obs_seg, atol=1e-6))
        else:
            obs_changed = True  # first episode: initial vs actual

        # size_random bug check
        sr_enabled = (
            test_cfg2.obstacle.size_random.enabled
            if hasattr(test_cfg2.obstacle, "size_random")
            else False
        )
        uses_cfg_fixed_in_obs = False
        if sr_enabled and env_obs_size is not None:
            cfg_fixed_obs_size = cfg_fixed_half / ws_range
            uses_cfg_fixed_in_obs = bool(
                np.allclose(obs[OBS_SIZE_START:OBS_SIZE_END], cfg_fixed_obs_size, atol=1e-5)
            )

        ep_results.append({
            "episode": ep + 1,
            "env_obs_center": env_obs_center.copy() if env_obs_center is not None else None,
            "env_obs_size": env_obs_size.copy() if env_obs_size is not None else None,
            "cfg_fixed_size": cfg_fixed_size,
            "expected_seg": expected_seg.copy(),
            "actual_seg": actual_seg.copy(),
            "seg_diff": seg_diff.copy(),
            "seg_match": seg_match,
            "center_changed": center_changed,
            "size_changed": size_changed,
            "obs_changed": obs_changed,
            "uses_cfg_fixed_in_obs": uses_cfg_fixed_in_obs,
        })

        prev_center = env_obs_center.copy() if env_obs_center is not None else None
        prev_size = env_obs_size.copy() if env_obs_size is not None else None
        prev_obs_seg = actual_seg.copy()

    env2.close()

    return {
        "case": case,
        "mode": {"A": "fixed", "B": "random", "C": "fixed", "D": "random"}[case],
        "size_random": {"A": False, "B": False, "C": True, "D": True}[case],
        "episodes": ep_results,
    }


def print_summary(all_results: list):
    """Print the audit table."""
    print()
    print("=" * 110)
    print("  OBS-CHECK SUMMARY TABLE")
    print("=" * 110)
    header = (
        f"  {'Case':<5} {'Mode':<8} {'SR':<4} {'Ep':<3} "
        f"{'env_center':>20} {'env_size':>16} "
        f"{'obs_segment':>30} "
        f"{'seg_match':<10} {'center_chg':<12} {'size_chg':<10} {'obs_chg':<9} {'bug':<5}"
    )
    print(header)
    print("-" * 110)

    case_labels = {"A": "FIXED+noSR", "B": "RAND+noSR", "C": "FIXED+SR", "D": "RAND+SR"}

    for res in all_results:
        c = res["case"]
        mode = res["mode"]
        sr = res["size_random"]
        for ep_res in res["episodes"]:
            ep = ep_res["episode"]
            ec = ep_res["env_obs_center"]
            es = ep_res["env_obs_size"]
            seg = ep_res["actual_seg"]

            ec_str = (
                f"[{float(ec[0]):+.4f},{float(ec[1]):+.4f},{float(ec[2]):+.4f}]"
                if ec is not None else "None"
            )
            es_str = (
                f"[{float(es[0]):.4f},{float(es[1]):.4f},{float(es[2]):.4f}]"
                if es is not None else "None"
            )
            seg_str = (
                f"[{float(seg[0]):+.4f},{float(seg[1]):+.4f},{float(seg[2]):+.4f},"
                f"{float(seg[3]):.4f},{float(seg[4]):.4f},{float(seg[5]):.4f}]"
            )

            bug_str = "BUG!" if ep_res["uses_cfg_fixed_in_obs"] else ""

            print(
                f"  {c:<5} {mode:<8} {str(sr):<4} {ep:<3} "
                f"{ec_str:>20} {es_str:>16} "
                f"{seg_str:>30} "
                f"{str(ep_res['seg_match']):<10} "
                f"{str(ep_res['center_changed']):<12} "
                f"{str(ep_res['size_changed']):<10} "
                f"{str(ep_res['obs_changed']):<9} "
                f"{bug_str:<5}"
            )

    print("=" * 110)

    # Pass/fail per case
    print()
    print("  PASS/FAIL AUDIT")
    print("  " + "-" * 60)

    # Expectation: for episode 1 (prev=None), any "changed" is meaningless — skip check.
    # For episodes 2+, changed should match the case expectation.
    # We report per-episode but use Ep2+ for pass/fail.
    case_expectations = {
        "A": {"center_changed": False, "size_changed": False, "obs_changed": False},
        "B": {"center_changed": True,  "size_changed": False, "obs_changed": True},
        # Case C: center_z changes with size even though x,y are fixed
        "C": {"center_changed": True,  "size_changed": True,  "obs_changed": True},
        "D": {"center_changed": True,  "size_changed": True,  "obs_changed": True},
    }

    all_passed = True
    for res in all_results:
        c = res["case"]
        exp = case_expectations[c]
        for ep_res in res["episodes"]:
            ep = ep_res["episode"]

            # For episode 1, prev=None, so center_changed/size_changed/obs_changed
            # will always be True. Skip these checks for ep1 (expected vs actual
            # match only matters for ep2+ where we have a real prev value).
            if ep == 1:
                print(f"  Case {c} Ep 1: skipped (prev=None; first episode, not comparable)")
                # Still check seg_match and size_random bug for ep1
                ok = ep_res["seg_match"]
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_passed = False
                print(f"  Case {c} Ep 1: seg_match = {ok} => {status}")
                if ep_res["uses_cfg_fixed_in_obs"]:
                    all_passed = False
                    print(f"  Case {c} Ep 1: [BUG] obs_size uses cfg.fixed_size despite size_random.enabled=True")
                else:
                    print(f"  Case {c} Ep 1: size_random_check => PASS")
                continue

            for key, expected in exp.items():
                actual = ep_res[key]
                ok = (actual == expected)
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_passed = False
                print(
                    f"  Case {c} Ep {ep}: {key} = {actual} (expected {expected}) => {status}"
                )

            # seg_match
            ok = ep_res["seg_match"]
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False
            print(
                f"  Case {c} Ep {ep}: seg_match = {ok} => {status}"
            )

            # size_random bug
            if ep_res["uses_cfg_fixed_in_obs"]:
                all_passed = False
                print(
                    f"  Case {c} Ep {ep}: [BUG] obs_size still uses cfg.fixed_size "
                    f"despite size_random.enabled=True"
                )
            else:
                print(
                    f"  Case {c} Ep {ep}: size_random_check => PASS"
                )

    print("  " + "-" * 60)
    if all_passed:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS FAILED — see details above")

    return all_passed


def main():
    default_config = _SCRIPT_DIR.parent / "config" / "environment.yaml"
    cfg = Config.from_yaml(default_config)

    print()
    print("=" * 110)
    print("  OBSERVATION OBSTACLE AUDIT — 4 CASES")
    print("=" * 110)
    print(f"  Config: {default_config}")
    print(f"  cfg.obstacle.mode         = {cfg.obstacle.mode}")
    print(f"  cfg.obstacle.center       = {cfg.obstacle.center}")
    print(f"  cfg.obstacle.size         = {cfg.obstacle.size}")
    print(f"  cfg.obstacle.size_random  = enabled={cfg.obstacle.size_random.enabled}")
    print(f"    length: [{cfg.obstacle.size_random.length_min}, {cfg.obstacle.size_random.length_max}]")
    print(f"    width:  [{cfg.obstacle.size_random.width_min},  {cfg.obstacle.size_random.width_max}]")
    print(f"    height: [{cfg.obstacle.size_random.height_min}, {cfg.obstacle.size_random.height_max}]")
    print(f"  workspace_range           = {cfg.workspace.range_np.tolist()}")
    print(f"  table_top_z               = {float(cfg.table.center_np[2] + cfg.table.size_np[2]/2.0):.6f}")
    print("=" * 110)

    all_results = []
    for case in ["A", "B", "C", "D"]:
        print(f"\n  Running Case {case}...")
        res = run_case(case, cfg, n_episodes=3)
        all_results.append(res)
        print(f"  Case {case} done.")

    passed = print_summary(all_results)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
