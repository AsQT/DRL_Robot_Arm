"""
Main training entry-point for SB3 algorithms.

Provides ``train_sb3_model()`` — a single high-level function that:
  1. Loads the unified Config object
  2. Creates run directories (lazily, after configs are loaded)
  3. Seeds everything
  4. Builds the environment with wrappers
  5. Builds the SB3 model
  6. Trains it
  7. Saves all artifacts
  8. Prints the summary

The unified ``Config`` object replaces separate environment and algorithm YAML files.
Algorithm-specific scripts (``train_td3.py``) are thin wrappers that parse CLI args and call this.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import stable_baselines3.common.callbacks as sb3_cb

from drl_pathplanning.gymnasium.config import Config
from drl_pathplanning.training.env_factory import create_training_env, EnvOutput
from drl_pathplanning.training.sb3_factory import create_sb3_model
from drl_pathplanning.training.run_config import TrainingRun, create_training_run
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


def train_sb3_model(
    algorithm: str,
    env_config: Config,
    total_timesteps: int | None = None,
    seed: int | None = None,
    n_envs: int | None = None,
    vec_env_type: str = "auto",
    progress_bar: bool | None = None,
    progress_log_interval: int = 5000,
    load_model: Path | None = None,
    load_replay_buffer: Path | None = None,
    gui: bool = False,
    show: bool = False,
    render_sleep: float = 0.0,
    render_first_episodes: int = 0,
    safety_warmup_timesteps: int | None = None,
    safety_warmup_episodes: int | None = None,
    safety_min_success_rate: float | None = None,
    config_path: Path | None = None,
) -> TrainingRun:
    """
    Train a Stable-Baselines3 model on the Cartesian path planning task.

    All parameters are read from the unified ``Config`` object passed as
    ``env_config``. No separate algo-config or curriculum-config YAML files are needed.

    Parameters
    ----------
    algorithm
        One of ``"DDPG"``, ``"SAC"``, ``"TD3"``.
    env_config
        The unified ``Config`` object loaded from ``config/environment.yaml``.
    total_timesteps
        Override total training timesteps.  If ``None``, uses
        ``env_config.training.total_timesteps``.
    seed
        Override random seed.  If ``None``, uses ``env_config.training.seed``.
    n_envs
        Override number of parallel environments.  If ``None``, uses
        ``env_config.training.n_envs``.
    vec_env_type
        Vectorized env backend: ``"auto"`` (default), ``"dummy"``, or ``"subproc"``.
    progress_bar
        Enable tqdm/rich progress bar.  If ``None``, uses
        ``env_config.training.progress_bar``.
    load_model
        Path to a ``.zip`` model file to fine-tune.
    load_replay_buffer
        Path to a ``replay_buffer.pkl`` to load before training.
    safety_warmup_timesteps
        Minimum global timesteps before EarlyStopSafetyCallback checks success rate.
        Default: 50000 for fine-tune/resume, 0 for from-scratch.
    safety_warmup_episodes
        Minimum completed episodes before EarlyStopSafetyCallback checks success rate.
        Default: 200 for fine-tune/resume, 0 for from-scratch.
    safety_min_success_rate
        Override minimum success rate threshold for EarlyStopSafetyCallback.
        Default: 0.90.

    Returns
    -------
    TrainingRun
        The created run with all directory paths.
    """
    cfg = env_config

    # ---- Resolve effective values ----
    effective_seed = seed if seed is not None else cfg.training.seed
    effective_timesteps = (
        total_timesteps if total_timesteps is not None else cfg.training.total_timesteps
    )
    effective_n_envs = n_envs if n_envs is not None else cfg.training.n_envs
    effective_vec_env_type = (
        vec_env_type if vec_env_type != "auto" else cfg.training.vec_env_type
    )
    effective_progress_bar = (
        progress_bar if progress_bar is not None else cfg.training.progress_bar
    )

    # Build algo_cfg dict from the algorithm-specific config section
    algo_upper = algorithm.upper()
    if algo_upper == "TD3":
        algo_c = cfg.td3
    elif algo_upper == "DDPG":
        algo_c = cfg.ddpg
    elif algo_upper == "SAC":
        algo_c = cfg.sac
    elif algo_upper == "PPO":
        algo_c = cfg.ppo
    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Supported: TD3, DDPG, SAC, PPO"
        )

    # PPO uses a completely different config path — build its dict here and
    # skip the off-policy common kwargs below.
    if algo_upper == "PPO":
        algo_cfg: dict[str, Any] = {
            "gamma": algo_c.gamma,
            "learning_rate": algo_c.learning_rate,
            "batch_size": algo_c.batch_size,
            "n_epochs": algo_c.n_epochs,
            "n_steps": algo_c.n_steps,
            "gae_lambda": algo_c.gae_lambda,
            "clip_range": algo_c.clip_range,
            "ent_coef": algo_c.ent_coef,
            "vf_coef": algo_c.vf_coef,
            "max_grad_norm": algo_c.max_grad_norm,
            "policy_kwargs": algo_c.policy_kwargs,
            "policy": algo_c.policy,
            "log_interval": cfg.training.log_interval,
            "episode_log_interval": cfg.training.episode_log_interval,
        }
    else:
        algo_cfg = {
            "gamma": algo_c.gamma,
            "learning_rate": algo_c.learning_rate,
            "batch_size": algo_c.batch_size,
            "buffer_size": algo_c.buffer_size,
            "learning_starts": algo_c.learning_starts,
            "tau": algo_c.tau,
            "train_freq": algo_c.train_freq,
            "gradient_steps": algo_c.gradient_steps,
            "policy_kwargs": algo_c.policy_kwargs,
            "policy": algo_c.policy,
            "log_interval": cfg.training.log_interval,
            "episode_log_interval": cfg.training.episode_log_interval,
        }
    if algo_upper == "TD3":
        algo_cfg["policy_delay"] = cfg.td3.policy_delay
        algo_cfg["target_policy_noise"] = cfg.td3.target_policy_noise
        algo_cfg["target_noise_clip"] = cfg.td3.target_noise_clip
    if algo_upper == "SAC":
        algo_cfg["ent_coef"] = algo_c.ent_coef
    if cfg.action_noise.enabled and algo_upper in ("TD3", "DDPG"):
        algo_cfg["action_noise"] = {
            "type": cfg.action_noise.type,
            "sigma": cfg.action_noise.sigma,
            "mean": cfg.action_noise.mean,
        }

    # Override seed in config for consistency
    cfg.training.seed = effective_seed

    # ---- GUI debug mode ----
    gui_enabled = gui
    if gui_enabled and effective_n_envs > 1:
        print(
            "[WARN] --gui true requires n_envs=1. "
            f"Got n_envs={effective_n_envs}. Forcing n_envs=1."
        )
        effective_n_envs = 1

    if gui_enabled:
        print("[INFO] GUI mode: using single env (n_envs=1).")

    # ---- Progress bar ----
    if effective_progress_bar and not _has_progress_bar():
        print(
            "[WARN] progress_bar=true but tqdm/rich is missing.\n"
            "      Install with: pip install tqdm rich\n"
            "      Continuing with progress_bar=False."
        )
        effective_progress_bar = False

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Train mode ----
    train_mode = "fine_tune" if load_model is not None else "from_scratch"
    print(f"[INFO] TRAIN_MODE: {train_mode}")
    if load_model:
        print(f"[INFO]   fine-tuning from: {load_model}")
    if load_replay_buffer:
        print(f"[INFO]   loading replay buffer from: {load_replay_buffer}")

    # ---- Print config summary ----
    print(f"[INFO] ACTION_STEP: {cfg.environment.action_step}")
    print(f"[INFO] MAX_STEPS: {cfg.environment.max_steps}")
    print(f"[INFO] Training device: {device}")
    print(f"[INFO] Algorithm: {algorithm.upper()}")
    print(f"[INFO] Seed: {effective_seed}")
    print(f"[INFO] Total timesteps: {effective_timesteps:,}")
    print(f"[INFO] n_envs: {effective_n_envs}")
    print(f"[INFO] vec_env_type: {effective_vec_env_type}")
    if gui_enabled:
        print(f"[INFO] Visualization: GUI ENABLED "
              f"(render_first={render_first_episodes}, sleep={render_sleep}s, show={show})")
    else:
        print(f"[INFO] Visualization: DISABLED")

    # ---- Create run directories ----
    run = create_training_run(
        algorithm=algorithm,
    )
    print(f"[INFO] Run directory: {run.run_dir}")

    # ---- SB3 logger ----
    import stable_baselines3.common.logger as _logger

    log_formats = ["stdout", "csv"]
    if _has_tensorboard():
        log_formats.append("tensorboard")
    new_logger = _logger.configure(str(run.log_dir), log_formats)

    # ---- Create environment ----
    print(f"[INFO] Creating environment "
          f"(n_envs={effective_n_envs}, vec_env_type={effective_vec_env_type})")

    # Curriculum: read from the unified config's curriculum section
    curriculum_config_path: Path | None = None
    if cfg.curriculum.enabled and cfg.curriculum.stages:
        curriculum_config_path = None  # Signals create_training_env to use the samplers from cfg

    env_result: EnvOutput = create_training_env(
        env_cfg=cfg,
        seed=effective_seed,
        n_envs=effective_n_envs,
        vec_env_type=effective_vec_env_type,
        log_dir=run.log_dir,
        curriculum_config_path=curriculum_config_path,
        curriculum_enabled=cfg.curriculum.enabled,
    )
    gym_env = env_result.env
    raw_env = env_result.raw_env

    # ---- Create debug GUI viewer (GUI mode only) ----
    viewer_callback = None
    if gui_enabled:
        from drl_pathplanning.training.viewer_callback import create_training_viewer_callback
        viewer_callback = create_training_viewer_callback(
            env_cfg=cfg,
            raw_env=raw_env,
            training_env=env_result.training_env,
            gui=gui_enabled,
            show=show,
            render_sleep=render_sleep,
            render_first_episodes=render_first_episodes,
        )

    # ---- Create model ----
    print(f"[INFO] Building {algorithm.upper()} model")
    model = create_sb3_model(
        algorithm=algorithm,
        env=gym_env,
        algo_cfg=algo_cfg,
        device=device,
    )
    model.set_logger(new_logger)

    # ---- Load pre-trained model for fine-tuning ----
    if load_model is not None:
        if not Path(load_model).exists():
            raise FileNotFoundError(f"--load-model: {load_model} does not exist")
        model.set_parameters(load_path_or_dict=str(load_model), exact_match=False)
        print(f"[INFO] Loaded model weights from: {load_model}")

    if load_replay_buffer is not None:
        from drl_pathplanning.training.io import load_replay_buffer as _do_load_rb
        if _do_load_rb(model, load_replay_buffer):
            print(f"[INFO] Loaded replay buffer from: {load_replay_buffer}")

    # ---- Print banner ----
    _print_banner(cfg=cfg, algorithm=algorithm, device=device,
                  total_timesteps=effective_timesteps, run=run,
                  n_envs=effective_n_envs, vec_env_type=effective_vec_env_type,
                  progress_bar=effective_progress_bar,
                  log_interval=cfg.training.log_interval,
                  train_mode=train_mode,
                  save_freq=cfg.training.save_freq)

    # ---- Training callbacks ----
    episode_callback = EpisodeCallback(
        log_interval=cfg.training.log_interval,
        episode_log_interval=cfg.training.episode_log_interval,
    )
    term_stats_callback = TerminationStatsCallback(log_interval=5)
    progress_callback = TimestepProgressCallback(
        total_timesteps=effective_timesteps,
        n_envs=effective_n_envs,
        log_interval=progress_log_interval,
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
        save_freq=cfg.training.save_freq,
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

    # ---- Safety callback for fine-tuning ----
    is_resume = load_model is not None
    _def_warmup_ts = 50000 if is_resume else 0
    _def_warmup_ep = 200 if is_resume else 0
    _def_min_sr = 0.90

    safety_callback: EarlyStopSafetyCallback | None = None
    if load_model is not None:
        _w_ts = (
            safety_warmup_timesteps if safety_warmup_timesteps is not None else _def_warmup_ts
        )
        _w_ep = (
            safety_warmup_episodes if safety_warmup_episodes is not None else _def_warmup_ep
        )
        _sr = safety_min_success_rate if safety_min_success_rate is not None else _def_min_sr

        safety_callback = EarlyStopSafetyCallback(
            min_success_rate=_sr,
            warmup_timesteps=_w_ts,
            warmup_episodes=_w_ep,
            check_every_n_episodes=20,
            check_window_size=20,
            max_action_norm=1.5,
            verbose=1,
        )
        callbacks.append(safety_callback)
        print(
            f"[INFO] Safety callback ENABLED\n"
            f"  warmup: timesteps={_w_ts:,}  episodes={_w_ep}\n"
            f"  min_success_rate={_sr:.2f}  max_action_norm=1.5"
        )

    # ---- Curriculum stage tracking ----
    if curriculum_config_path is None and env_result.master_curriculum_sampler is not None:
        from drl_pathplanning.training.callbacks import CurriculumGlobalStepCallback

        curriculum_callback = CurriculumGlobalStepCallback(
            master_sampler=env_result.master_curriculum_sampler,
            worker_samplers=env_result.worker_samplers,
            verbose=1,
        )
        callbacks.append(curriculum_callback)
        if effective_n_envs == 1:
            raw_env.add_reset_callback(curriculum_callback.on_env_reset)
        else:
            for w_env in raw_env:
                w_env.add_reset_callback(curriculum_callback.on_env_reset)
        print("[INFO] Curriculum stage callback ENABLED")

    # ---- Training loop ----
    print("[INFO] Training started.\n")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=effective_timesteps,
            callback=callbacks,
            progress_bar=effective_progress_bar,
            log_interval=cfg.training.log_interval,
            reset_num_timesteps=(load_model is None),
        )
    except Exception as e:
        print(f"\n[ERROR] Training interrupted: {e}")

    elapsed = time.time() - t0

    # ---- Save artifacts ----
    print("\n[INFO] Training complete. Saving artifacts...")

    mode = "continue" if load_model else "from_scratch"
    yaml_path, json_path, info_path = save_full_config_snapshots(
        run_dir=run.run_dir,
        env_cfg=cfg,
        algo_cfg=algo_cfg,
        algo_name=algorithm,
        device=device,
        total_timesteps=effective_timesteps,
        config_path=config_path,
        mode=mode,
        checkpoint_path=load_model,
        load_replay_buffer=bool(load_replay_buffer),
        reset_num_timesteps=(load_model is None),
        continue_timesteps=effective_timesteps if load_model else None,
        seed=cfg.training.seed,
        extra={
            "training.total_timesteps": effective_timesteps,
            "training.n_envs": effective_n_envs,
            "training.vec_env_type": effective_vec_env_type,
            "training.progress_bar": effective_progress_bar,
        },
    )
    print(f"[INFO] Config YAML   : {yaml_path}")
    print(f"[INFO] Config JSON   : {json_path}")
    print(f"[INFO] Run info     : {info_path}")

    model_path = save_final_model(model, run.model_dir)
    print(f"[INFO] Final model saved: {model_path}")

    replay_path = save_replay_buffer(model, run.run_dir)
    if replay_path is not None:
        print(f"[INFO] Replay buffer saved: {replay_path}")

    time_path = save_training_time(run.log_dir, elapsed)
    print(f"[INFO] Training time: {time_path}")

    # ---- Summary ----
    print_training_summary(
        episode_callback.episode_rewards,
        episode_callback.episode_lengths,
        episode_callback.episode_successes,
        episode_callback.episode_distances,
        elapsed,
        episode_callback.episode_expected_path_lengths,
        episode_callback.episode_actual_path_lengths,
    )

    # ---- Cleanup ----
    gym_env.close()
    if viewer_callback is not None:
        viewer_callback.close()

    print(f"[INFO] Done. Run: {run.run_dir}")
    return run


# --------------------------------------------------------------------------- #
# Internal helpers
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


def _print_banner(
    cfg: Config,
    algorithm: str,
    device: str,
    total_timesteps: int,
    run: TrainingRun,
    n_envs: int,
    vec_env_type: str,
    progress_bar: bool,
    log_interval: int,
    train_mode: str,
    save_freq: int | None = None,
) -> None:
    """Print the training banner with full configuration details."""
    sep = "=" * 60
    print(sep)
    print(f"  Cartesian Path Planning — {algorithm.upper()} Training")
    print(sep)
    print(f"  TRAIN_MODE        : {train_mode}")
    print(f"  ALGORITHM         : {algorithm.upper()}")
    print(f"  TOTAL_TIMESTEPS   : {total_timesteps:,}")
    print(f"  SEED              : {cfg.training.seed}")
    print(f"  N_ENVS            : {n_envs}")
    print(f"  VEC_ENV_TYPE      : {vec_env_type}")
    print(f"  DEVICE            : {device}")
    print(f"  PROGRESS_BAR     : {progress_bar}")
    print(f"  LOG_INTERVAL     : {log_interval}")
    if save_freq is not None:
        print(f"  SAVE_FREQ         : {save_freq:,} timesteps")
    print(f"  RUN_ID           : {run.run_id}")
    print(f"  TENSORBOARD       : {'available' if _has_tensorboard() else 'NOT installed'}")
    print(sep)
    print("  ENVIRONMENT:")
    print(f"    action_step         : {cfg.environment.action_step}")
    print(f"    max_episode_steps   : {cfg.environment.max_episode_steps}")
    print(f"    target_threshold    : {cfg.termination.goal_threshold}")
    print(f"    workspace_min       : {cfg.workspace.min_np.tolist()}")
    print(f"    workspace_max       : {cfg.workspace.max_np.tolist()}")
    print(f"    start.mode          : {cfg.start.mode}")
    print(f"    start.fixed_pos    : {cfg.start.fixed_position}")
    print(f"    start.random_bounds.min: {cfg.start.random_bounds.min}")
    print(f"    start.random_bounds.max: {cfg.start.random_bounds.max}")
    print(f"    target.mode         : {cfg.target_region.mode}")
    print(f"    target.fixed_pos   : {cfg.target_region.fixed_position}")
    print(f"    target.random_bounds.min: {cfg.target_region.random_bounds.min}")
    print(f"    target.random_bounds.max: {cfg.target_region.random_bounds.max}")
    print(f"    obstacle.enabled   : {cfg.obstacle.enabled}")
    print(f"    collision.enabled  : {cfg.collision.enabled}")
    print(f"    termination.goal   : {cfg.termination.goal_threshold}")
    print(f"    table_enabled      : {cfg.table.enabled}")
    print(sep)
    print("  REWARD (simple distance-based):")
    print(f"    success_bonus      : {cfg.reward.success_bonus}")
    print(f"    collision_penalty  : {cfg.reward.collision_penalty}")
    print(f"    workspace_penalty  : {cfg.reward.workspace_penalty}")
    print(f"    timeout_penalty   : {cfg.reward.timeout_penalty}")
    print(f"    distance_scale     : {cfg.reward.distance_scale}")
    print(f"    time_penalty       : {cfg.reward.time_penalty}")
    print(f"    shake_penalty_scale: {cfg.reward.shake_penalty_scale}")
    print(f"    shake_window       : {cfg.reward.shake_window}")
    print(f"    shake_dot_threshold: {cfg.reward.shake_dot_threshold}")
    print(sep)
    print()


class TimestepProgressCallback(sb3_cb.BaseCallback):
    """
    Lightweight timestep-level progress logger.

    Prints a compact progress line every ``log_interval`` timesteps when the
    SB3 rich progress bar is disabled.
    """

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
