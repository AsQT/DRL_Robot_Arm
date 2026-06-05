"""
Plot training and evaluation results from a training run.

Reads progress.csv (from training callbacks) and evaluation CSV files and saves
PNG plots to Data/Plots/<run_name>/.

Usage:
    # Plot from a training run directory
    python Evaluation/plot_results.py \
        --run Data/Training/Environment_Default/TD3/FRAME_ONLY/run_20260518_023828 \
        --show false

    # Plot from explicit CSV paths
    python Evaluation/plot_results.py \
        --progress-csv Data/Training/.../logs/progress.csv \
        --eval-csv Data/Evaluation/.../evaluation_episodes.csv \
        --show false

    # Plot from monitor.csv (SB3 Monitor wrapper format)
    python Evaluation/plot_results.py \
        --monitor Data/Training/.../logs/monitor.csv \
        --show false
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# =============================================================================
# USER CONFIG - Change only this run path
# =============================================================================
USE_DEFAULT_RUN_DIR = True

DEFAULT_RUN_DIR = (
    _PROJECT_ROOT
    / "Data"
    / "Training"
    / "TD3"
    / "FRAME_ONLY"
    / "run_20260501_random_obs_start_size"
)
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_output_dir(run_dir: Path) -> Path:
    run_name = run_dir.name
    out_dir = Path("Data/Plots") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def infer_run_info(run_dir: Path) -> dict:
    """Infer algorithm, mode, and run name from the run directory path."""
    parts = run_dir.parts
    info = {
        "algo": "unknown",
        "mode": "unknown",
        "run_name": run_dir.name,
    }
    if "Training" in parts:
        idx = parts.index("Training")
        if len(parts) > idx + 1:
            info["algo"] = parts[idx + 1]
        if len(parts) > idx + 2:
            info["mode"] = parts[idx + 2]
    return info


def _load_progress_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")
    return df


def _load_eval_csv(csv_path: Path) -> pd.DataFrame:
    """Load evaluation_episodes.csv which stores rows as JSON on individual lines."""
    rows = []
    with open(csv_path) as fh:
        next(fh, None)  # skip CSV header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _load_monitor_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path, comment="#")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Plot: training progress (progress.csv)
# --------------------------------------------------------------------------- #

def _plot_reward_curve(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "rollout/ep_rew_mean"
    if col not in df.columns:
        return None
    timesteps = df["time/total_timesteps"].values.astype(float)
    reward = df[col].values.astype(float)
    mask = ~np.isnan(reward)
    timesteps = timesteps[mask]
    reward = reward[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, reward, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    ax.set_title("Training: Episode Reward vs. Timesteps")
    fig.tight_layout()

    out_path = out_dir / "reward_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_success_rate(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "rollout/success_rate"
    if col not in df.columns:
        return None
    timesteps = df["time/total_timesteps"].values.astype(float)
    success = df[col].values.astype(float)
    mask = ~np.isnan(success)
    timesteps = timesteps[mask]
    success = success[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, success * 100, color="tab:green", linewidth=1.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate (%)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title("Training: Success Rate vs. Timesteps")
    fig.tight_layout()

    out_path = out_dir / "success_rate_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_distance_curve(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "rollout/distance_improvement_mean"
    if col not in df.columns:
        return None
    timesteps = df["time/total_timesteps"].values.astype(float)
    dist = df[col].values.astype(float)
    mask = ~np.isnan(dist)
    timesteps = timesteps[mask]
    dist = dist[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, dist, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Distance Improvement (m/step)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(True, alpha=0.3)
    ax.set_title("Training: Distance Improvement vs. Timesteps")
    fig.tight_layout()

    out_path = out_dir / "distance_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Plot: monitor.csv (SB3 Monitor format)
# --------------------------------------------------------------------------- #

def _plot_monitor_csv(monitor_df: pd.DataFrame, out_dir: Path) -> list:
    saved = []
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_len_col = "l" if "l" in monitor_df.columns else ("episode_length" if "episode_length" in monitor_df.columns else None)
    rew_col = "r" if "r" in monitor_df.columns else ("reward" if "reward" in monitor_df.columns else None)

    if rew_col:
        rewards = monitor_df[rew_col].astype(float).values
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, color="tab:blue", linewidth=1.0, alpha=0.7, label="Episode Reward")
        # Rolling mean
        window = min(50, len(rewards) // 4 + 1)
        if window > 1:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(rewards) + 1), smooth, color="tab:blue", linewidth=2, label=f"Rolling mean (w={window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Episode Reward (Monitor CSV)")
        fig.tight_layout()
        out_path = out_dir / "monitor_reward.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    if ep_len_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        lengths = monitor_df[ep_len_col].astype(float).values
        episodes = np.arange(1, len(lengths) + 1)
        ax.plot(episodes, lengths, color="tab:orange", linewidth=1.0, alpha=0.7, label="Episode Length")
        window = min(50, len(lengths) // 4 + 1)
        if window > 1:
            smooth = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(lengths) + 1), smooth, color="tab:orange", linewidth=2, label=f"Rolling mean (w={window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Episode Length (Monitor CSV)")
        fig.tight_layout()
        out_path = out_dir / "monitor_length.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved


# --------------------------------------------------------------------------- #
# Plot: evaluation CSV
# --------------------------------------------------------------------------- #

def _plot_eval_success_rate(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "success"
    if col not in df.columns:
        return None
    success_vals = df[col].astype(float).values
    episodes = np.arange(1, len(success_vals) + 1)
    cum_success = np.cumsum(success_vals) / episodes * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, cum_success, color="tab:green", linewidth=1.5, label="Cumulative Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Success Rate (%)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title("Evaluation: Cumulative Success Rate vs. Episode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "eval_success_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_eval_path_efficiency(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "path_efficiency"
    if col not in df.columns:
        return None
    eff = df[col].astype(float).values
    episodes = np.arange(1, len(eff) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, eff * 100, color="tab:blue", linewidth=1.5, label="Path Efficiency (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Path Efficiency (%)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    ax.set_title("Evaluation: Path Efficiency vs. Episode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "eval_path_efficiency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_eval_action_delta(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "mean_action_delta"
    if col not in df.columns:
        return None
    delta = df[col].astype(float).values
    episodes = np.arange(1, len(delta) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, delta, color="tab:orange", linewidth=1.5, label="Mean Action Delta")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Action Delta", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax.grid(True, alpha=0.3)
    ax.set_title("Evaluation: Mean Action Delta vs. Episode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "eval_action_delta.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_eval_cosine_alignment(df: pd.DataFrame, out_dir: Path) -> Path | None:
    col = "mean_cosine_alignment"
    if col not in df.columns:
        return None
    align = df[col].astype(float).values
    episodes = np.arange(1, len(align) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, align, color="tab:purple", linewidth=1.5, label="Mean Cosine Alignment")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Cosine Alignment", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title("Evaluation: Mean Cosine Alignment vs. Episode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "eval_cosine_alignment.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training and evaluation results")
    parser.add_argument(
        "--run",
        type=str,
        default=str(DEFAULT_RUN_DIR) if USE_DEFAULT_RUN_DIR else None,
        help=f"Path to training run directory. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--progress-csv",
        type=str,
        default=None,
        help="Explicit path to logs/progress.csv",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Explicit path to logs/monitor.csv (SB3 Monitor format)",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        default=None,
        help="Explicit path to evaluation_episodes.csv",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Display plots interactively",
    )
    args = parser.parse_args()

    # Resolve paths
    progress_path: Path | None = None
    monitor_path: Path | None = None
    eval_path: Path | None = None
    out_dir: Path | None = None

    if args.progress_csv:
        progress_path = Path(args.progress_csv)
    if args.monitor:
        monitor_path = Path(args.monitor)
    if args.eval_csv:
        eval_path = Path(args.eval_csv)
    if args.run:
        run_path = Path(args.run)
        if not run_path.exists():
            print(f"[ERROR] Run directory not found:")
            print(f"        {run_path}")
            print()
            print(f"        Please update DEFAULT_RUN_DIR in USER CONFIG or pass --run manually.")
            return 1
        out_dir = _make_output_dir(run_path)
        run_info = infer_run_info(run_path)
        print("[INFO] Run dir : " + str(run_path))
        print("[INFO] Algo    : " + run_info["algo"])
        print("[INFO] Mode    : " + run_info["mode"])
        print("[INFO] Run name: " + run_info["run_name"])
        if progress_path is None:
            p = run_path / "logs" / "progress.csv"
            if p.exists():
                progress_path = p
        if monitor_path is None:
            m = run_path / "logs" / "monitor.csv"
            if m.exists():
                monitor_path = m
        if eval_path is None:
            candidates = sorted((run_path.parent.parent).glob("eval_*"))
            if candidates:
                eval_path = candidates[-1] / "evaluation_episodes.csv"
                if not eval_path.exists():
                    eval_path = None

    if out_dir is None:
        if progress_path:
            out_dir = _make_output_dir(progress_path.parent.parent)
        elif monitor_path:
            out_dir = _make_output_dir(monitor_path.parent.parent)
        else:
            print("[ERROR] Cannot determine output directory. Pass --run or explicit CSV paths.")
            return 1

    show = args.show.lower() == "true"

    print("=" * 60)
    print("  Plot Results")
    print("=" * 60)
    print(f"  progress.csv : {progress_path or 'not provided'}")
    print(f"  monitor.csv : {monitor_path or 'not provided'}")
    print(f"  eval CSV    : {eval_path or 'not provided'}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Show       : {show}")
    print("=" * 60)

    saved = []

    # Progress CSV
    if progress_path and progress_path.exists():
        df = _load_progress_csv(progress_path)
        print(f"[INFO] Loaded {len(df)} rows from progress.csv")
        print(f"[INFO] Columns: {list(df.columns)[:10]}...")

        p = _plot_reward_curve(df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")

        p = _plot_success_rate(df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")

        p = _plot_distance_curve(df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")
    elif progress_path:
        print(f"[WARN] progress.csv not found: {progress_path}")

    # Monitor CSV
    if monitor_path:
        df = _load_monitor_csv(monitor_path)
        if df is not None and len(df) > 0:
            print(f"[INFO] Loaded {len(df)} rows from monitor.csv")
            plots = _plot_monitor_csv(df, out_dir)
            for p in plots:
                saved.append(p)
                print(f"[INFO] Saved: {p}")
        elif monitor_path:
            print(f"[WARN] monitor.csv not found or empty: {monitor_path}")
    elif not progress_path:
        print("[WARN] No progress.csv provided — nothing to plot from training logs")

    # Evaluation CSV
    if eval_path and eval_path.exists():
        print(f"[INFO] Loading evaluation CSV: {eval_path}")
        eval_df = _load_eval_csv(eval_path)
        print(f"[INFO] Loaded {len(eval_df)} episodes from evaluation CSV")

        p = _plot_eval_success_rate(eval_df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")
        else:
            print(f"[WARN] 'success' column not in eval CSV — skipping success rate plot")

        p = _plot_eval_path_efficiency(eval_df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")

        p = _plot_eval_action_delta(eval_df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")

        p = _plot_eval_cosine_alignment(eval_df, out_dir)
        if p:
            saved.append(p)
            print(f"[INFO] Saved: {p}")
    elif eval_path:
        print(f"[WARN] Evaluation CSV not found: {eval_path}")

    if not saved:
        print("[WARN] No plots were generated — check file paths above")
        if show:
            plt.close("all")
        return 1

    if show:
        plt.show()

    print(f"[INFO] Done. {len(saved)} plot(s) saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
