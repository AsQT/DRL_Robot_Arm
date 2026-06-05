"""
Continue training PPO from a checkpoint.

PPO is on-policy — there is no replay buffer. This script loads model weights
and continues training with a fresh buffer (PPO's rollout buffer, not replay).

Usage::

    python Training/train_ppo_continue_checkpoint.py \\
        --checkpoint Data/Training/PPO/.../checkpoint_t5000000.zip \\
        --timesteps 5000000
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
from stable_baselines3 import PPO
import stable_baselines3.common.callbacks as sb3_cb
import stable_baselines3.common.logger as _logger

from drl_pathplanning.gymnasium.config import load_config, Config
from drl_pathplanning.training.env_factory import create_training_env, EnvOutput
from drl_pathplanning.training.run_config import TrainingRun
from drl_pathplanning.training.io import (
    save_full_config_snapshots,
    save_final_model,
    save_training_time,
    print_training_summary,
)
from drl_pathplanning.training.callbacks import (
    EpisodeCallback,
    TerminationStatsCallback,
    EvalBestModelCallback,
    SaveCheckpointCallback,
)
from drl_pathplanning.training.viewer_callback import create_training_viewer_callback

# --------------------------------------------------------------------------- #
# USER CONFIG — edit these values
# --------------------------------------------------------------------------- #
PROJECT_ROOT = _SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "environment.yaml"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Training"
    / "PPO"
    / "FRAME_ONLY"
    / "run_YYYYMMDD_HHMMSS"
    / "model"
    / "checkpoint_tXXXXXXX.zip"
)
CONTINUE_TIMESTEPS = 5_000_000

# Suffix appended to the new run directory name.
# "AUTO" = auto-detect from checkpoint filename.
RUN_NAME_SUFFIX = "AUTO"

RESET_NUM_TIMESTEPS = False
SAVE_FREQ = 0
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue PPO training from checkpoint")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--timesteps", type=int, default=CONTINUE_TIMESTEPS)
    parser.add_argument("--run-name-suffix", type=str, default=RUN_NAME_SUFFIX)
    parser.add_argument("--gui", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--show", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--render-first-episodes", type=int, default=RENDER_FIRST_EPISODES)
    parser.add_argument("--render-sleep", type=float, default=RENDER_SLEEP)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    parser.add_argument(
        "--reset-num-timesteps", type=str,
        default="false" if not RESET_NUM_TIMESTEPS else "true",
        choices=["true", "false"],
    )
    return parser.parse_args()


def _build_ppo_kwargs(cfg: Config) -> dict:
    ppo_c = cfg.ppo
    return {
        "gamma": ppo_c.gamma,
        "learning_rate": ppo_c.learning_rate,
        "n_steps": ppo_c.n_steps,
        "batch_size": ppo_c.batch_size,
        "n_epochs": ppo_c.n_epochs,
        "gae_lambda": ppo_c.gae_lambda,
        "clip_range": ppo_c.clip_range,
        "ent_coef": ppo_c.ent_coef,
        "vf_coef": ppo_c.vf_coef,
        "max_grad_norm": ppo_c.max_grad_norm,
        "policy_kwargs": ppo_c.policy_kwargs,
        "policy": ppo_c.policy,
    }


def _create_run_dir(suffix: str) -> TrainingRun:
    base_dir = _SCRIPT_DIR.parent / "Data" / "Training"
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + suffix
    run_dir = base_dir / "PPO" / "FRAME_ONLY" / f"run_{run_id}"
    for d in [run_dir / "model", run_dir / "logs", run_dir / "tensorboard", run_dir / "trajectory"]:
        d.mkdir(parents=True, exist_ok=True)
    return TrainingRun(
        run_id=run_id, run_dir=run_dir,
        model_dir=run_dir / "model", log_dir=run_dir / "logs",
        tensorboard_dir=run_dir / "tensorboard", trajectory_dir=run_dir / "trajectory",
        algorithm="PPO",
    )


class _ProgressCallback(sb3_cb.BaseCallback):
    def __init__(self, total_timesteps: int, log_interval: int = 5000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._last = 0
        self._start: float | None = None

    def _on_step(self) -> bool:
        if self._start is None:
            self._start = time.time()
        n = self.num_timesteps
        if n - self._last < self.log_interval:
            return True
        self._last = n
        elapsed = time.time() - self._start
        pct = n / self.total_timesteps * 100
        fps = n / elapsed if elapsed > 0 else 0
        eta = (elapsed / pct * 100 - elapsed) if pct > 0 else 0
        if self.verbose:
            print(f"[TRAIN] steps={n}/{self.total_timesteps} {pct:5.1f}%% | fps={fps:6.0f} | eta={eta:5.0f}s")
        return True


def _auto_suffix_from_checkpoint(checkpoint_path: Path) -> str:
    """Extract timestep tag from checkpoint filename."""
    import re
    stem = checkpoint_path.stem
    match = re.search(r"t(\d+)", stem)
    if match:
        return f"continue_from_t{match.group(1)}"
    return "continue_from_checkpoint"


def main() -> None:
    args = _parse_args()

    if args.run_name_suffix.upper() == "AUTO":
        args.run_name_suffix = _auto_suffix_from_checkpoint(args.checkpoint)
        print(f"[INFO] Auto suffix: {args.run_name_suffix}")

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    cfg: Config = load_config(args.config)
    cfg.training.seed = args.seed if args.seed is not None else cfg.training.seed
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    gui_enabled = args.gui.lower() == "true"
    save_freq = args.save_freq if args.save_freq > 0 else cfg.training.save_freq

    print("=" * 60)
    print("  Continue Training — PPO")
    print("=" * 60)
    print(f"[INFO] Checkpoint    : {args.checkpoint}")
    print(f"[INFO] Config       : {args.config}")
    print(f"[INFO] Continue steps: {args.timesteps:,}")
    print(f"[INFO] Run suffix  : {args.run_name_suffix}")
    print(f"[INFO] Replay buf : N/A (PPO is on-policy)")
    print(f"[INFO] Save freq     : {save_freq:,}  (from {'CLI' if args.save_freq > 0 else 'config'})")
    print("=" * 60)

    run = _create_run_dir(args.run_name_suffix)
    print(f"[INFO] New run dir: {run.run_dir}")

    # Save config snapshots immediately so run_info.json exists even if the
    # run crashes or is interrupted.  A complete overwrite happens after training.
    yaml_path, json_path, info_path = save_full_config_snapshots(
        run_dir=run.run_dir,
        env_cfg=cfg,
        algo_cfg={},
        algo_name="PPO",
        device=device,
        total_timesteps=args.timesteps,
        config_path=args.config,
        mode="continue",
        checkpoint_path=args.checkpoint,
        load_replay_buffer=False,
        reset_num_timesteps=args.reset_num_timesteps.lower() == "true",
        continue_timesteps=args.timesteps,
        seed=cfg.training.seed,
        extra={
            "train_mode": "continue_checkpoint",
            "replay_buffer": "N/A (PPO is on-policy)",
        },
    )
    print(f"[INFO] Config YAML   : {yaml_path}")
    print(f"[INFO] Config JSON   : {json_path}")
    print(f"[INFO] Run info     : {info_path}")

    log_formats = ["stdout", "csv"]
    if _has_tensorboard():
        log_formats.append("tensorboard")
    _logger.configure(str(run.log_dir), log_formats)

    effective_n_envs = cfg.training.n_envs
    if gui_enabled and effective_n_envs > 1:
        effective_n_envs = 1

    print(f"[INFO] Creating environment (n_envs={effective_n_envs})")
    env_result: EnvOutput = create_training_env(
        env_cfg=cfg, seed=cfg.training.seed, n_envs=effective_n_envs,
        vec_env_type=cfg.training.vec_env_type, log_dir=run.log_dir,
        curriculum_config_path=None, curriculum_enabled=cfg.curriculum.enabled,
    )
    gym_env = env_result.env
    raw_env = env_result.raw_env

    print(f"[INFO] observation_space = {gym_env.observation_space}")
    print(f"[INFO] action_space      = {gym_env.action_space}")

    # Safety check
    reset_result = gym_env.reset()
    _obs = reset_result[0] if len(reset_result) == 5 else reset_result[1]
    _test_model = PPO.load(str(args.checkpoint), env=gym_env, device=device)
    _act, _ = _test_model.predict(_obs, deterministic=True)
    print(f"[INFO]   action shape = {_act.shape}, predict OK")
    del _test_model

    # Build PPO model
    ppo_kwargs = _build_ppo_kwargs(cfg)
    print(f"[INFO] Building PPO model (fresh rollout buffer)")
    model = PPO(env=gym_env, device=device, verbose=1, **ppo_kwargs)

    print(f"[INFO] Loading checkpoint weights: {args.checkpoint}")
    model.set_parameters(load_path_or_dict=str(args.checkpoint), exact_match=False)
    print(f"[INFO] Replay buffer  : N/A (PPO is on-policy)")

    viewer_cb = None
    if gui_enabled:
        viewer_cb = create_training_viewer_callback(
            env_cfg=cfg, raw_env=raw_env, training_env=env_result.training_env,
            gui=True, show=args.show.lower() == "true",
            render_sleep=args.render_sleep,
            render_first_episodes=args.render_first_episodes,
        )

    callbacks: list[sb3_cb.BaseCallback] = [
        EpisodeCallback(log_interval=cfg.training.log_interval, episode_log_interval=cfg.training.episode_log_interval),
        TerminationStatsCallback(log_interval=5),
        _ProgressCallback(args.timesteps),
        SaveCheckpointCallback(model_dir=run.model_dir, save_freq=save_freq),
        EvalBestModelCallback(
            eval_freq=cfg.training.eval_freq,
            n_eval_episodes=cfg.evaluation.num_episodes,
            eval_env_cfg=cfg,
            eval_seed=cfg.training.seed,
            model_dir=run.model_dir,
            warmup_episodes=50,
            verbose=1,
        ),
    ]
    if viewer_cb:
        callbacks.append(viewer_cb)

    print("=" * 60)
    print(f"  PPO — Continue from checkpoint")
    print("=" * 60)
    print(f"  CHECKPOINT    : {args.checkpoint.name}")
    print(f"  REPLAY_BUFFER : N/A (PPO is on-policy)")
    print(f"  SAVE_FREQ     : {save_freq:,}  (from {'CLI' if args.save_freq > 0 else 'config'})")
    print("=" * 60)
    print()

    print("[INFO] Training started.\n")
    t0 = time.time()
    reset_ts = args.reset_num_timesteps.lower() == "true"

    try:
        model.learn(
            total_timesteps=args.timesteps, callback=callbacks,
            progress_bar=cfg.training.progress_bar,
            log_interval=cfg.training.log_interval,
            reset_num_timesteps=reset_ts,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

    elapsed = time.time() - t0
    print("\n[INFO] Training complete. Saving final model...")

    save_final_model(model, run.model_dir)
    print(f"[INFO] Final model  : {run.model_dir / 'final_model.zip'}")
    print(f"[INFO] Replay buffer: N/A (PPO is on-policy)")
    save_training_time(run.log_dir, elapsed)

    print_training_summary(
        callbacks[0].episode_rewards, callbacks[0].episode_lengths,
        callbacks[0].episode_successes, callbacks[0].episode_distances,
        elapsed, callbacks[0].episode_expected_path_lengths,
        callbacks[0].episode_actual_path_lengths,
    )

    gym_env.close()
    if viewer_cb:
        viewer_cb.close()
    print(f"[INFO] Done. Run: {run.run_dir}")


if __name__ == "__main__":
    main()
