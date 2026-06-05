"""
Training and evaluation logger utilities.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_config_json(run_dir: Path, config: Dict[str, Any]) -> None:
    """
    Save an experiment configuration dict as ``run_dir/config.json``.
    """
    with open(run_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=4)


def save_time_log(log_dir: Path, elapsed_seconds: float) -> None:
    """Write the training elapsed time to ``log_dir/time.txt``."""
    (log_dir / "time.txt").write_text(f"Training time: {elapsed_seconds:.5f}s\n")


def print_training_summary(
    episode_rewards: list[float],
    episode_lengths: list[int],
    episode_successes: list[bool],
    episode_distances: list[float],
    elapsed: float,
) -> None:
    """Print a formatted summary of training results."""
    n = len(episode_rewards)
    if n < 1:
        return

    early = np.mean(episode_rewards[: max(1, n // 3)])
    late = np.mean(episode_rewards[-max(1, n // 3) :])
    success_count = sum(1 for s in episode_successes if s)
    finite_dists = [d for d in episode_distances if np.isfinite(d)]

    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Episodes           : {n}")
    print(f"  Early avg reward   : {early:.3f}  (first 1/3)")
    print(f"  Late  avg reward   : {late:.3f}  (last  1/3)")
    print(f"  Improvement        : {late - early:+.3f}")
    print(f"  Success rate       : {success_count}/{n} = {success_count / n * 100:.1f}%")
    if finite_dists:
        print(
            f"  Final distance     : mean={np.mean(finite_dists):.4f}  "
            f"min={np.min(finite_dists):.4f}  max={np.max(finite_dists):.4f}"
        )
    print(f"  Training time      : {elapsed:.1f}s")
    print("=" * 60)


def print_prediction_summary(
    episode_rewards: list[float],
    episode_lengths: list[int],
    episode_successes: list[bool],
    episode_distances: list[float],
    output_path: Path,
    episode_expected_path_lengths: list[float] | None = None,
    episode_actual_path_lengths: list[float] | None = None,
) -> None:
    """Print a formatted summary of prediction / evaluation results."""
    n = len(episode_rewards)
    if n < 1:
        return

    finite_dists = [d for d in episode_distances if np.isfinite(d)]
    mean_error = float(np.nanmean(episode_distances))
    max_error = float(np.nanmax(episode_distances))
    min_error = float(np.nanmin(episode_distances))
    mean_steps = float(np.mean(episode_lengths))
    success_rate = sum(1 for s in episode_successes if s) / n

    print("=" * 60)
    print("  PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Episodes          : {n}")
    print(f"  Success rate     : {success_rate:.2%}  ({sum(1 for s in episode_successes if s)}/{n})")
    print(f"  Mean reward      : {np.mean(episode_rewards):.3f}")
    print(f"  Mean steps       : {mean_steps:.1f}")
    print(f"  Mean error       : {mean_error:.4f}  m")
    print(f"  Max error        : {max_error:.4f}  m")
    print(f"  Min error        : {min_error:.4f}  m")

    if episode_expected_path_lengths and episode_actual_path_lengths:
        finite_s_exp = [
            x for x, s in zip(episode_expected_path_lengths, episode_successes)
            if s and np.isfinite(x)
        ]
        finite_s_act = [
            x for x, s in zip(episode_actual_path_lengths, episode_successes)
            if s and np.isfinite(x)
        ]
        if finite_s_exp and finite_s_act:
            effs = [
                exp / act * 100.0
                for exp, act in zip(finite_s_exp, finite_s_act)
                if act > 1e-8
            ]
            if effs:
                print(
                    f"  Mean success path eff %: {np.mean(effs):.2f}%  "
                    f"(n={len(effs)})"
                )
            print(
                f"  Mean success exp path: {np.mean(finite_s_exp):.4f} m  "
                f"(n={len(finite_s_exp)})"
            )
            print(
                f"  Mean success act path: {np.mean(finite_s_act):.4f} m  "
                f"(n={len(finite_s_act)})"
            )
        elif success_rate > 0:
            print("  Path efficiency   : N/A (no valid path data)")

    print(f"  Output file      : {output_path}")
    print("=" * 60)
