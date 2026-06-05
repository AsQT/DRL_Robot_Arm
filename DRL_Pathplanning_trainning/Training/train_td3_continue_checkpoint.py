"""
Continue training from a TD3 checkpoint WITHOUT loading the old replay buffer.

This script loads a trained TD3 model checkpoint and continues training with a
fresh replay buffer.  Use this when a checkpoint mid-training is better than the
final model, but the replay buffer from the end of training is mismatched.

Usage::

    # USER CONFIG at top of file — just edit the CHECKPOINT_PATH
    python Training/train_td3_continue_checkpoint.py

    # Override via CLI
    python Training/train_td3_continue_checkpoint.py \\
        --checkpoint Data/Training/TD3/FRAME_ONLY/run_xxx/model/checkpoint_t15000000.zip \\
        --timesteps 5000000 \\
        --run-name-suffix continue_from_best \\
        --gui false \\
        --show false
"""

import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import numpy as np

from drl_pathplanning.gymnasium.config import load_config, Config
from drl_pathplanning.training.env_factory import create_training_env, EnvOutput
from drl_pathplanning.training.sb3_factory import create_sb3_model
from drl_pathplanning.training.run_config import create_training_run, TrainingRun
from drl_pathplanning.training.io import (
    save_full_config_snapshots,
    save_final_model,
    save_replay_buffer,
    save_training_time,
    print_training_summary,
)
from drl_pathplanning.training.callbacks import (
    EpisodeCallback,
    TerminationStatsCallback,
    EarlyStopSafetyCallback,
    EvalBestModelCallback,
    SaveCheckpointCallback,
)
from drl_pathplanning.training.viewer_callback import create_training_viewer_callback

from stable_baselines3 import TD3
import stable_baselines3.common.callbacks as sb3_cb
import stable_baselines3.common.logger as _logger

# --------------------------------------------------------------------------- #
# USER CONFIG — edit these values
# --------------------------------------------------------------------------- #
PROJECT_ROOT = _SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "environment.yaml"

# Path to the checkpoint .zip to continue from
# EDIT THIS to your checkpoint path
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Training"
    / "TD3"
    / "FRAME_ONLY"
    / "run_20260528_221532"
    / "model"
    / "checkpoint_t25700000.zip"
)

# How many additional timesteps to train
CONTINUE_TIMESTEPS = 1_000_000

# Suffix appended to the new run directory name.
# "AUTO" = auto-detect from checkpoint filename (e.g. checkpoint_t25700000.zip → continue_from_t25700000_no_replay).
RUN_NAME_SUFFIX = "AUTO"

# Always False — this script intentionally does NOT load the old replay buffer
LOAD_REPLAY_BUFFER = False

# Load with the old episode counter (True) or reset to 0 (False)?
RESET_NUM_TIMESTEPS = False

# Save checkpoint every this many timesteps (0 = use config default)
SAVE_FREQ = 0

# --------------------------------------------------------------------------- #
# SAFE EARLY STOP — disabled by default for continue-without-replay
# --------------------------------------------------------------------------- #
# Disable SAFE EARLY STOP because a fresh replay buffer after 25M+ timesteps
# means the rolling success-rate window is meaningless.  The model needs time
# to fill the new buffer before its success-rate signal is trustworthy.
SAFE_EARLY_STOP_ENABLED = False
SAFE_MIN_SUCCESS_RATE = 0.90
SAFE_WARMUP_EPISODES = 200
SAFE_CHECK_INTERVAL = 20   # check every N episodes
SAFE_WINDOW_SIZE = 20      # rolling window size
SAFE_MAX_ACTION_NORM = 1.5

# GUI / rendering (keep False for long training runs)
GUI = False
SHOW = False
RENDER_FIRST_EPISODES = 0
RENDER_SLEEP = 0.0
# --------------------------------------------------------------------------- #


def _has_tensorboard() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except ImportError:
        return False


def _has_progress_bar() -> bool:
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue TD3 training from a checkpoint without loading the old replay buffer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to config YAML [default: {CONFIG_PATH}]",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Path to TD3 checkpoint .zip [default: {CHECKPOINT_PATH}]",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=CONTINUE_TIMESTEPS,
        help=f"Number of additional timesteps [default: {CONTINUE_TIMESTEPS}]",
    )
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default=RUN_NAME_SUFFIX,
        help=f"Run directory suffix [default: {RUN_NAME_SUFFIX}]",
    )
    parser.add_argument(
        "--gui",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable PyBullet GUI [default: false]",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Print per-episode results in GUI [default: false]",
    )
    parser.add_argument(
        "--render-first-episodes",
        type=int,
        default=RENDER_FIRST_EPISODES,
        help=f"Render only the first N episodes in GUI [default: {RENDER_FIRST_EPISODES}]",
    )
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=RENDER_SLEEP,
        help=f"Sleep between steps in GUI [default: {RENDER_SLEEP}]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device [default: auto]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config [default: use config value]",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=SAVE_FREQ,
        help=f"Checkpoint save frequency (0 = use config) [default: {SAVE_FREQ}]",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        type=str,
        default="false" if not RESET_NUM_TIMESTEPS else "true",
        choices=["true", "false"],
        help="Reset SB3 episode counter [default: false]",
    )
    parser.add_argument(
        "--safe-early-stop",
        type=str,
        default="false" if not SAFE_EARLY_STOP_ENABLED else "true",
        choices=["true", "false"],
        help="Enable EarlyStopSafetyCallback [default: false]",
    )
    parser.add_argument(
        "--safe-min-success-rate",
        type=float,
        default=SAFE_MIN_SUCCESS_RATE,
        help=f"Min rolling success rate threshold [default: {SAFE_MIN_SUCCESS_RATE}]",
    )
    parser.add_argument(
        "--safe-warmup-episodes",
        type=int,
        default=SAFE_WARMUP_EPISODES,
        help=f"Episode warmup before safety checks start [default: {SAFE_WARMUP_EPISODES}]",
    )
    parser.add_argument(
        "--safe-check-interval",
        type=int,
        default=SAFE_CHECK_INTERVAL,
        help=f"Check safety every N episodes [default: {SAFE_CHECK_INTERVAL}]",
    )
    parser.add_argument(
        "--safe-window-size",
        type=int,
        default=SAFE_WINDOW_SIZE,
        help=f"Rolling window size for success rate [default: {SAFE_WINDOW_SIZE}]",
    )
    parser.add_argument(
        "--safe-max-action-norm",
        type=float,
        default=SAFE_MAX_ACTION_NORM,
        help=f"Max action L2-norm before flagging [default: {SAFE_MAX_ACTION_NORM}]",
    )
    return parser.parse_args()


def _build_algo_cfg(cfg: Config) -> dict:
    td3_c = cfg.td3
    algo_cfg: dict = {
        "gamma": td3_c.gamma,
        "learning_rate": td3_c.learning_rate,
        "batch_size": td3_c.batch_size,
        "buffer_size": td3_c.buffer_size,
        "learning_starts": td3_c.learning_starts,
        "tau": td3_c.tau,
        "train_freq": td3_c.train_freq,
        "gradient_steps": td3_c.gradient_steps,
        "policy_kwargs": td3_c.policy_kwargs,
        "policy": td3_c.policy,
        "log_interval": cfg.training.log_interval,
        "episode_log_interval": cfg.training.episode_log_interval,
        "policy_delay": td3_c.policy_delay,
        "target_policy_noise": td3_c.target_policy_noise,
        "target_noise_clip": td3_c.target_noise_clip,
    }
    if cfg.action_noise.enabled:
        import copy
        action_noise_cfg = {
            "type": cfg.action_noise.type,
            "sigma": cfg.action_noise.sigma,
            "mean": cfg.action_noise.mean,
        }
        algo_cfg["action_noise"] = action_noise_cfg
    return algo_cfg


def _create_run_dir(algorithm: str, suffix: str) -> TrainingRun:
    """Create a new timestamped run directory."""
    base_dir = _SCRIPT_DIR.parent / "Data" / "Training"
    import time as _time
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + suffix
    run_dir = base_dir / algorithm.upper() / "FRAME_ONLY" / f"run_{run_id}"
    model_dir = run_dir / "model"
    log_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    trajectory_dir = run_dir / "trajectory"
    for d in [model_dir, log_dir, tensorboard_dir, trajectory_dir]:
        d.mkdir(parents=True, exist_ok=True)

    from drl_pathplanning.training.run_config import TrainingRun
    return TrainingRun(
        run_id=run_id,
        run_dir=run_dir,
        model_dir=model_dir,
        log_dir=log_dir,
        tensorboard_dir=tensorboard_dir,
        trajectory_dir=trajectory_dir,
        algorithm=algorithm.upper(),
    )


def _auto_suffix_from_checkpoint(checkpoint_path: Path) -> str:
    """
    Extract the timestep tag from a checkpoint filename.

    e.g.  checkpoint_t25700000.zip  →  continue_from_t25700000_no_replay
    """
    stem = checkpoint_path.stem          # e.g. "checkpoint_t25700000"
    suffix = stem.replace("checkpoint", "continue_from")
    if not LOAD_REPLAY_BUFFER:
        suffix += "_no_replay"
    return suffix


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------- #
    # 0. Resolve run name suffix (AUTO or explicit)
    # ------------------------------------------------------------------- #
    if args.run_name_suffix.upper() == "AUTO":
        args.run_name_suffix = _auto_suffix_from_checkpoint(args.checkpoint)
        print(f"[INFO] Auto suffix: {args.run_name_suffix}")

    # Parse safe-early-stop flag
    safe_enabled = args.safe_early_stop.lower() == "true"

    # ------------------------------------------------------------------- #
    # 1. Safety checks
    # ------------------------------------------------------------------- #
    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    print("=" * 60)
    print("  Continue Training — TD3")
    print("=" * 60)
    print(f"[INFO] Checkpoint    : {args.checkpoint}")
    print(f"[INFO] Config        : {args.config}")
    print(f"[INFO] Continue steps: {args.timesteps:,}")
    print(f"[INFO] Run suffix    : {args.run_name_suffix}")
    print(f"[INFO] Load replay   : {LOAD_REPLAY_BUFFER}  (always False in this script)")
    print(f"[INFO] reset_num_timesteps: {args.reset_num_timesteps}")
    print(f"[INFO] Safe early stop: {'enabled' if safe_enabled else 'disabled'}  (override with --safe-early-stop true)")
    print("=" * 60)

    # ------------------------------------------------------------------- #
    # 2. Load config
    # ------------------------------------------------------------------- #
    cfg: Config = load_config(args.config)
    effective_seed = args.seed if args.seed is not None else cfg.training.seed
    cfg.training.seed = effective_seed

    algorithm = cfg.training.algorithm.upper()
    if algorithm not in ("TD3", "DDPG", "SAC"):
        sys.exit(f"[ERROR] Unsupported algorithm: {algorithm}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gui_enabled = args.gui.lower() == "true"
    show = args.show.lower() == "true"

    save_freq = args.save_freq if args.save_freq > 0 else cfg.training.save_freq

    # ------------------------------------------------------------------- #
    # 3. Create new run directory
    # ------------------------------------------------------------------- #
    run = _create_run_dir(algorithm, args.run_name_suffix)
    print(f"[INFO] New run dir: {run.run_dir}")

    # ------------------------------------------------------------------- #
    # 4. Configure SB3 logger (new directory, no overwrite)
    # ------------------------------------------------------------------- #
    log_formats = ["stdout", "csv"]
    if _has_tensorboard():
        log_formats.append("tensorboard")
    _logger.configure(str(run.log_dir), log_formats)

    # ------------------------------------------------------------------- #
    # 5. Create environment
    # ------------------------------------------------------------------- #
    effective_n_envs = cfg.training.n_envs
    if gui_enabled and effective_n_envs > 1:
        print("[WARN] --gui requires n_envs=1. Forcing n_envs=1.")
        effective_n_envs = 1

    print(f"[INFO] Creating environment (n_envs={effective_n_envs})")
    curriculum_config_path: Path | None = None
    if cfg.curriculum.enabled and cfg.curriculum.stages:
        curriculum_config_path = None

    env_result: EnvOutput = create_training_env(
        env_cfg=cfg,
        seed=effective_seed,
        n_envs=effective_n_envs,
        vec_env_type=cfg.training.vec_env_type,
        log_dir=run.log_dir,
        curriculum_config_path=curriculum_config_path,
        curriculum_enabled=cfg.curriculum.enabled,
    )
    gym_env = env_result.env
    raw_env = env_result.raw_env

    print(f"[INFO] observation_space = {gym_env.observation_space}")
    print(f"[INFO] action_space      = {gym_env.action_space}")

    # ------------------------------------------------------------------- #
    # 6. Safety check: model predict from checkpoint
    # ------------------------------------------------------------------- #
    print("[INFO] Testing checkpoint prediction...")
    _reset_result = gym_env.reset()
    # gymnasium env.reset() returns 5 values, gym returns 2
    if len(_reset_result) == 2:
        _obs, _info = _reset_result
    else:
        _obs, _info = _reset_result[0], _reset_result[1]
    _model_for_test = TD3.load(str(args.checkpoint), env=gym_env, device=device)
    _test_action, _ = _model_for_test.predict(_obs, deterministic=True)
    print(f"[INFO]   obs shape     = {_obs.shape}")
    print(f"[INFO]   action shape = {_test_action.shape}")
    print(f"[INFO]   predict OK   = True")
    del _model_for_test

    # ------------------------------------------------------------------- #
    # 7. Build algo cfg and create model
    # ------------------------------------------------------------------- #
    print(f"[INFO] Building {algorithm} model (fresh replay buffer)")
    algo_cfg = _build_algo_cfg(cfg)
    model = create_sb3_model(
        algorithm=algorithm,
        env=gym_env,
        algo_cfg=algo_cfg,
        device=device,
    )

    # ------------------------------------------------------------------- #
    # 8. Load checkpoint weights only (NO replay buffer)
    # ------------------------------------------------------------------- #
    print(f"[INFO] Loading checkpoint weights: {args.checkpoint}")
    model.set_parameters(load_path_or_dict=str(args.checkpoint), exact_match=False)
    print(f"[INFO] Replay buffer  : NEW / EMPTY (not loaded)")
    print(f"[INFO] Load replay    : False")

    # ------------------------------------------------------------------- #
    # 9. GUI viewer callback
    # ------------------------------------------------------------------- #
    viewer_callback = None
    if gui_enabled:
        viewer_callback = create_training_viewer_callback(
            env_cfg=cfg,
            raw_env=raw_env,
            training_env=env_result.training_env,
            gui=True,
            show=show,
            render_sleep=args.render_sleep,
            render_first_episodes=args.render_first_episodes,
        )

    # ------------------------------------------------------------------- #
    # 10. Training callbacks
    # ------------------------------------------------------------------- #
    episode_callback = EpisodeCallback(
        log_interval=cfg.training.log_interval,
        episode_log_interval=cfg.training.episode_log_interval,
    )
    term_stats_callback = TerminationStatsCallback(log_interval=5)
    progress_callback = _TimestepProgressCallback(
        total_timesteps=args.timesteps,
        n_envs=effective_n_envs,
        log_interval=5000,
    )
    best_model_callback = EvalBestModelCallback(
        eval_freq=cfg.training.eval_freq,
        n_eval_episodes=cfg.evaluation.num_episodes,
        eval_env_cfg=cfg,
        eval_seed=cfg.training.seed,
        model_dir=run.model_dir,
        warmup_episodes=50,
        verbose=1,
    )
    checkpoint_callback = SaveCheckpointCallback(
        model_dir=run.model_dir,
        save_freq=save_freq,
        verbose=1,
    )

    callbacks: list[sb3_cb.BaseCallback] = [
        episode_callback,
        term_stats_callback,
        progress_callback,
        best_model_callback,
        checkpoint_callback,
    ]
    if viewer_callback is not None:
        callbacks.append(viewer_callback)

    # Safety callback — only when --safe-early-stop true
    if safe_enabled:
        safety_callback = EarlyStopSafetyCallback(
            min_success_rate=args.safe_min_success_rate,
            warmup_episodes=args.safe_warmup_episodes,
            check_every_n_episodes=args.safe_check_interval,
            check_window_size=args.safe_window_size,
            max_action_norm=args.safe_max_action_norm,
            verbose=1,
        )
        callbacks.append(safety_callback)

    # ------------------------------------------------------------------- #
    # 11. Print banner
    # ------------------------------------------------------------------- #
    print("=" * 60)
    print(f"  Continue Training — {algorithm}")
    print("=" * 60)
    print(f"  TRAIN_MODE       : continue_checkpoint")
    print(f"  ALGORITHM        : {algorithm}")
    print(f"  TOTAL_TIMESTEPS  : {args.timesteps:,}")
    print(f"  SEED             : {effective_seed}")
    print(f"  N_ENVS           : {effective_n_envs}")
    print(f"  DEVICE           : {device}")
    print(f"  LOG_INTERVAL     : {cfg.training.log_interval}")
    print(f"  EP_LOG_INTERVAL  : {cfg.training.episode_log_interval}")
    print(f"  SAVE_FREQ        : {save_freq:,}  (from {'CLI' if args.save_freq > 0 else 'config'})")
    print(f"  CHECKPOINT       : {args.checkpoint}")
    print(f"  REPLAY_BUFFER   : NEW / EMPTY")
    print(f"  RUN_ID           : {run.run_id}")
    print(f"  TENSORBOARD      : {'available' if _has_tensorboard() else 'NOT installed'}")
    print(f"  PROGRESS_BAR     : {cfg.training.progress_bar}")
    print(f"  SAFE_EARLY_STOP  : {'ENABLED' if safe_enabled else 'DISABLED'}")
    if safe_enabled:
        print(f"    min_success_rate   : {args.safe_min_success_rate}")
        print(f"    warmup_episodes   : {args.safe_warmup_episodes}")
        print(f"    check_interval    : {args.safe_check_interval}")
        print(f"    window_size       : {args.safe_window_size}")
        print(f"    max_action_norm   : {args.safe_max_action_norm}")
    print("=" * 60)
    print(f"  ENVIRONMENT:")
    print(f"    action_step       : {cfg.environment.action_step}")
    print(f"    max_episode_steps : {cfg.environment.max_episode_steps}")
    print(f"    goal_threshold    : {cfg.termination.goal_threshold}")
    print(f"    obstacle.enabled  : {cfg.obstacle.enabled}")
    print(f"    collision.enabled : {cfg.collision.enabled}")
    print(f"  REWARD (simple distance-based):")
    print(f"    success_bonus      : {cfg.reward.success_bonus}")
    print(f"    collision_penalty  : {cfg.reward.collision_penalty}")
    print(f"    workspace_penalty  : {cfg.reward.workspace_penalty}")
    print(f"    timeout_penalty   : {cfg.reward.timeout_penalty}")
    print(f"    distance_scale    : {cfg.reward.distance_scale}")
    print(f"    shake_penalty    : {cfg.reward.shake_penalty_scale}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------- #
    # 12. Training loop
    # ------------------------------------------------------------------- #
    print("[INFO] Training started.\n")
    t0 = time.time()
    reset_ts = args.reset_num_timesteps.lower() == "true"

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=cfg.training.progress_bar,
            log_interval=cfg.training.log_interval,
            reset_num_timesteps=reset_ts,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        raise

    elapsed = time.time() - t0

    # ------------------------------------------------------------------- #
    # 13. Save artifacts
    # ------------------------------------------------------------------- #
    print("\n[INFO] Training complete. Saving artifacts...")

    yaml_path, json_path, info_path = save_full_config_snapshots(
        run_dir=run.run_dir,
        env_cfg=cfg,
        algo_cfg=algo_cfg,
        algo_name=algorithm,
        device=device,
        total_timesteps=args.timesteps,
        config_path=args.config,
        mode="continue",
        checkpoint_path=args.checkpoint,
        load_replay_buffer=False,
        reset_num_timesteps=reset_ts,
        continue_timesteps=args.timesteps,
        seed=cfg.training.seed,
        extra={
            "train_mode": "continue_checkpoint",
            "safe_early_stop_enabled": safe_enabled,
            "training.n_envs": effective_n_envs,
            "training.vec_env_type": cfg.training.vec_env_type,
            "training.progress_bar": cfg.training.progress_bar,
        },
    )
    print(f"[INFO] Config YAML   : {yaml_path}")
    print(f"[INFO] Config JSON   : {json_path}")
    print(f"[INFO] Run info     : {info_path}")

    model_path = save_final_model(model, run.model_dir)
    print(f"[INFO] Final model    : {model_path}")

    replay_path = save_replay_buffer(model, run.run_dir)
    if replay_path is not None:
        print(f"[INFO] Replay buffer  : {replay_path}  (new buffer from this run)")
    else:
        print(f"[INFO] Replay buffer  : not available for this algorithm")

    time_path = save_training_time(run.log_dir, elapsed)
    print(f"[INFO] Training time   : {time_path}")

    # ------------------------------------------------------------------- #
    # 14. Summary
    # ------------------------------------------------------------------- #
    print_training_summary(
        episode_callback.episode_rewards,
        episode_callback.episode_lengths,
        episode_callback.episode_successes,
        episode_callback.episode_distances,
        elapsed,
        episode_callback.episode_expected_path_lengths,
        episode_callback.episode_actual_path_lengths,
    )

    # ------------------------------------------------------------------- #
    # 15. Cleanup
    # ------------------------------------------------------------------- #
    gym_env.close()
    if viewer_callback is not None:
        viewer_callback.close()

    print(f"[INFO] Done. Run: {run.run_dir}")


# --------------------------------------------------------------------------- #
# Internal: lightweight progress callback (not exported in training/__init__)
# --------------------------------------------------------------------------- #

class _TimestepProgressCallback(sb3_cb.BaseCallback):
    """Prints compact progress lines without needing tqdm."""

    def __init__(
        self,
        total_timesteps: int,
        n_envs: int = 1,
        log_interval: int = 5000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.log_interval = log_interval
        self._last_print_step = 0
        self._start_time: float | None = None

    def _on_step(self) -> bool:
        num_timesteps = self.num_timesteps
        if self._start_time is None:
            self._start_time = time.time()
        if num_timesteps - self._last_print_step < self.log_interval:
            return True

        elapsed = time.time() - self._start_time
        pct = num_timesteps / self.total_timesteps * 100
        fps = num_timesteps / elapsed if elapsed > 0 else 0
        eta = (elapsed / pct * 100 - elapsed) if pct > 0 else 0

        if self.verbose:
            print(
                f"[TRAIN] steps={num_timesteps}/{self.total_timesteps} "
                f"{pct:5.1f}% | fps={fps:6.0f} | "
                f"elapsed={elapsed:5.0f}s | eta={eta:5.0f}s | "
                f"n_envs={self.n_envs}"
            )

        self._last_print_step = num_timesteps
        return True


if __name__ == "__main__":
    main()
