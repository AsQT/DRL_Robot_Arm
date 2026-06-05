"""
Proximal Policy Optimization (PPO) training entry point.

Loads config/environment.yaml and trains a PPO model on the Cartesian path planning task.
PPO is on-policy — no replay buffer is used.

Run from scratch::

    python Training/train_ppo.py --config config/environment.yaml

For GPU training, set ``training.device: cuda`` in the config YAML.
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
from drl_pathplanning.training.run_config import create_training_run, TrainingRun
from drl_pathplanning.training.io import (
    save_full_config_snapshots,
    save_final_model,
    save_training_time,
    save_run_info,
    print_training_summary,
)
from drl_pathplanning.training.callbacks import (
    EpisodeCallback,
    TerminationStatsCallback,
    SaveCheckpointCallback,
    EvalBestModelCallback,
)
from drl_pathplanning.training.viewer_callback import create_training_viewer_callback


def _has_tensorboard() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Cartesian Path Planning")
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR.parent / "config" / "environment.yaml",
        help="Path to the unified config YAML [default: config/environment.yaml]",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config [default: use config value]",
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
        default=0,
        help="Render only the first N episodes in GUI [default: 0]",
    )
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.0,
        help="Sleep between steps in GUI [default: 0.0]",
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


def main() -> None:
    args = _parse_args()

    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    cfg: Config = load_config(args.config)
    effective_seed = args.seed if args.seed is not None else cfg.training.seed
    effective_timesteps = args.timesteps if args.timesteps is not None else cfg.training.total_timesteps
    cfg.training.seed = effective_seed

    # Script dictates algorithm, overriding whatever is in the YAML
    script_algo = "PPO"
    cfg.training.algorithm = script_algo

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Algorithm requested by script: {script_algo}")
    print(f"[INFO] Training algorithm: {script_algo}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gui_enabled = args.gui.lower() == "true"
    show = args.show.lower() == "true"

    # Create run directory
    run = create_training_run(algorithm="PPO")
    print(f"[INFO] Run directory: {run.run_dir}")

    # Save config snapshots immediately so run_info.json exists even if the
    # run crashes or is interrupted.  A complete overwrite happens after training.
    yaml_path, json_path, info_path = save_full_config_snapshots(
        run_dir=run.run_dir,
        env_cfg=cfg,
        algo_cfg={},
        algo_name="PPO",
        device=device,
        total_timesteps=effective_timesteps,
        config_path=args.config,
        mode="from_scratch",
        checkpoint_path=None,
        load_replay_buffer=False,
        reset_num_timesteps=True,
        continue_timesteps=None,
        seed=effective_seed,
        extra={
            "train_mode": "from_scratch",
            "replay_buffer": "N/A (PPO is on-policy)",
        },
    )
    print(f"[INFO] Config YAML   : {yaml_path}")
    print(f"[INFO] Config JSON   : {json_path}")
    print(f"[INFO] Run info     : {info_path}")

    # SB3 logger
    log_formats = ["stdout", "csv"]
    if _has_tensorboard():
        log_formats.append("tensorboard")
    _logger.configure(str(run.log_dir), log_formats)

    # Create environment
    effective_n_envs = cfg.training.n_envs
    if gui_enabled and effective_n_envs > 1:
        print("[WARN] --gui requires n_envs=1. Forcing n_envs=1.")
        effective_n_envs = 1

    print(f"[INFO] Creating environment (n_envs={effective_n_envs})")
    env_result: EnvOutput = create_training_env(
        env_cfg=cfg,
        seed=effective_seed,
        n_envs=effective_n_envs,
        vec_env_type=cfg.training.vec_env_type,
        log_dir=run.log_dir,
        curriculum_config_path=None,
        curriculum_enabled=cfg.curriculum.enabled,
    )
    gym_env = env_result.env
    raw_env = env_result.raw_env

    print(f"[INFO] observation_space = {gym_env.observation_space}")
    print(f"[INFO] action_space      = {gym_env.action_space}")

    # Create PPO model
    ppo_kwargs = _build_ppo_kwargs(cfg)
    print(f"[INFO] Building PPO model (n_steps={ppo_kwargs['n_steps']}, n_envs={effective_n_envs})")
    model = PPO(
        env=gym_env,
        device=device,
        verbose=1,
        **ppo_kwargs,
    )

    # Callbacks
    episode_callback = EpisodeCallback(
        log_interval=cfg.training.log_interval,
        episode_log_interval=cfg.training.episode_log_interval,
    )
    term_stats_callback = TerminationStatsCallback(log_interval=5)
    checkpoint_callback = SaveCheckpointCallback(
        model_dir=run.model_dir,
        save_freq=cfg.training.save_freq,
        verbose=1,
    )
    callbacks: list[sb3_cb.BaseCallback] = [
        episode_callback,
        term_stats_callback,
        checkpoint_callback,
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

    # GUI viewer
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
        callbacks.append(viewer_callback)

    # Banner
    print("=" * 60)
    print(f"  Cartesian Path Planning — PPO Training")
    print("=" * 60)
    print(f"  ALGORITHM         : PPO")
    print(f"  TOTAL_TIMESTEPS  : {effective_timesteps:,}")
    print(f"  SEED             : {effective_seed}")
    print(f"  N_ENVS           : {effective_n_envs}")
    print(f"  DEVICE           : {device}")
    print(f"  LOG_INTERVAL     : {cfg.training.log_interval}")
    print(f"  SAVE_FREQ        : {cfg.training.save_freq:,} timesteps")
    print(f"  REPLAY_BUFFER    : not applicable (PPO is on-policy)")
    print("=" * 60)
    print()

    # Training
    print("[INFO] Training started.\n")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=effective_timesteps,
            callback=callbacks,
            progress_bar=cfg.training.progress_bar,
            log_interval=cfg.training.log_interval,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        raise

    elapsed = time.time() - t0

    # Save final model
    print("\n[INFO] Training complete. Saving final model...")
    model_path = save_final_model(model, run.model_dir)
    print(f"[INFO] Final model saved: {model_path}")
    print(f"[INFO] Replay buffer: not applicable (PPO is on-policy)")

    time_path = save_training_time(run.log_dir, elapsed)
    print(f"[INFO] Training time: {time_path}")

    print_training_summary(
        episode_callback.episode_rewards,
        episode_callback.episode_lengths,
        episode_callback.episode_successes,
        episode_callback.episode_distances,
        elapsed,
        episode_callback.episode_expected_path_lengths,
        episode_callback.episode_actual_path_lengths,
    )

    gym_env.close()
    if viewer_callback is not None:
        viewer_callback.close()

    print(f"[INFO] Done. Run: {run.run_dir}")


if __name__ == "__main__":
    main()
