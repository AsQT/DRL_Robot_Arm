"""
Continue-training script for DDPG GP7 models.

Loads a previously saved DDPG policy (actor + critic weights) and continues
training from where it left off.  A new timestamped output directory is created
for every continued run so the original model is never overwritten.

================================================================================
USAGE - two modes are supported:
================================================================================

  MODE 1 - Direct-edit (no CLI args):
    Edit the SCRIPT_* constants in the USER CONFIG block at the top of this file,
    then simply run::

        python Training/continue_train_ddpg_gp7.py

    This is useful for repeated use or quick runs without remembering flags.

  MODE 2 - Command-line arguments (override script config):
    Pass CLI arguments to override any SCRIPT_* setting::

        python Training/continue_train_ddpg_gp7.py --run-dir <path> --timesteps 300000
        python Training/continue_train_ddpg_gp7.py --model-path <path> --enable-gui

    CLI arguments always take precedence over SCRIPT_* values when both are present.

  Show all CLI options::

        python Training/continue_train_ddpg_gp7.py --help

================================================================================
OUTPUT:
================================================================================

    Data/Training/Environment_{mode}/DDPG/YASKAWA_GP7/continue_YYYYMMDD_HHMMSS/
        config.json
        logs/
        model/
            continued_model.zip
        tensorboard/
        replay_buffer.pkl   (only if --load-replay-buffer was used)

================================================================================
NOTES:
================================================================================

- reset_num_timesteps=False is always used so training curves continue from the
  prior run (no learning-rate schedule reset).
- The original training script (train_ddpg_gp7.py) does NOT save a replay buffer
  by default.  The script prints a clear [WARN] when no replay buffer is found.
- The smoke check (reset + one predict + step) runs before training starts.
"""
import argparse
import random
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Resolve the src/ directory relative to this script file.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import time
import gymnasium as gym
import stable_baselines3
import stable_baselines3.common.noise
import stable_baselines3.common.logger
import stable_baselines3.common.monitor
import stable_baselines3.common.vec_env
import stable_baselines3.common.callbacks
import RoLE.Parameters.Robot as Parameters
import RoLE.Utilities.File_IO
import Industrial_Robotics_Gym

CONST_PROJECT_FOLDER = Path(_SCRIPT_DIR).parent.resolve()
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str

CONST_LOG_INTERVAL = 10  # episodes — match original training script

# =============================================================================
# USER CONFIG — edit here for direct-run mode (Mode 1)
# =============================================================================
# Set to True and fill in the values below, then run:
#     python Training/continue_train_ddpg_gp7.py
#
# Set to False to require CLI arguments (--run-dir or --model-path).
USE_SCRIPT_CONFIG = True

SCRIPT_RUN_DIR = SCRIPT_RUN_DIR = r"D:\01. Master\0000. Thesis report\DRL\DRL\DRL_Robot_Manipulator\Data\Training\Environment_Default\DDPG\YASKAWA_GP7\run_20260505_010923"
SCRIPT_MODEL_PATH = None

SCRIPT_TIMESTEPS = 300000
SCRIPT_ENABLE_GUI = True
SCRIPT_SAVE_NAME = "continued_model.zip"
SCRIPT_LOAD_REPLAY_BUFFER = False
SCRIPT_SEED = None          # None = auto-detect from original config.json
SCRIPT_ENV_MODE = None       # None = auto-detect from original config.json
SCRIPT_DEVICE = "auto"       # "auto", "cuda", or "cpu"
# =============================================================================

# progress_bar availability (requires tqdm AND rich).
try:
    import tqdm as _  # noqa: F401
    import rich  # noqa: F401
    _PROGRESS_BAR = True
except ImportError:
    _PROGRESS_BAR = False

# TensorBoard availability.
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False


def _resolve_model_path(model_path: str, run_dir: str | None) -> Path:
    """
    Resolve a model path from either an explicit --model-path or an auto-detected
    path inside --run-dir.
    """
    if run_dir is not None:
        run_dir_p = Path(run_dir).resolve()
        if not run_dir_p.exists():
            raise FileNotFoundError(f'[FATAL] Run directory not found: {run_dir_p}')
        model_candidate = run_dir_p / 'model' / 'final_model.zip'
        if model_candidate.exists():
            print(f'[INFO] Auto-detected model: {model_candidate}')
            return model_candidate
        raise FileNotFoundError(
            f'[FATAL] Could not find model at expected path: {model_candidate}\n'
            f'        The directory exists but does not contain model/final_model.zip.'
        )

    if model_path is None:
        raise ValueError('[FATAL] Either --model-path or --run-dir must be provided.')

    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'[FATAL] Model file not found: {path}')
    return path


def _load_config_json(run_dir: Path) -> dict:
    """Load config.json from a run directory if it exists."""
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            return json.load(f)
    return {}


def _detect_env_mode_from_config(run_dir: Path) -> str | None:
    """Read env_mode from the original run's config.json if available."""
    cfg = _load_config_json(run_dir)
    return cfg.get('env_mode')


def _build_output_dir(project_folder: Path, env_mode: str) -> Path:
    """Create and return a new timestamped continue_/ output directory."""
    base = project_folder / 'Data' / 'Training' / f'Environment_{env_mode}' / 'DDPG' / 'YASKAWA_GP7'
    cont_id = f'continue_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir = base / cont_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class EpisodeCallback(stable_baselines3.common.callbacks.BaseCallback):
    """
    Episode-level progress logger — mirrors EpisodeCallback from train_ddpg_gp7.py.

    Tracks cumulative reward, episode length, final distance, and success for each
    episode.  Prints a formatted progress line every `log_interval` episodes
    showing rolling averages and the success rate based on info["is_success"].

    The success rate is derived from the environment's terminated flag (via
    info["is_success"]) rather than a distance threshold, ensuring consistency
    with the SB3 Monitor rollout/success_rate metric.
    """

    def __init__(self, log_interval: int = CONST_LOG_INTERVAL, total_timesteps: int = 0):
        super().__init__()
        self.log_interval = log_interval
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.episode_successes = []
        self._current_rewards = []
        self._current_distances = []
        self._current_success_flags = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]
        self._current_rewards.append(float(reward))
        if 'distance' in info:
            self._current_distances.append(float(info['distance']))
        if 'is_success' in info:
            self._current_success_flags.append(bool(info['is_success']))

        if done:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_lengths.append(len(self._current_rewards))
            final_distance = (
                self._current_distances[-1]
                if self._current_distances else float('nan')
            )
            self.episode_distances.append(final_distance)

            if self._current_success_flags:
                final_success = bool(self._current_success_flags[-1])
            elif 'is_success' in info:
                final_success = bool(info['is_success'])
            else:
                final_success = bool(
                    np.isfinite(final_distance) and final_distance < 0.01
                )
            self.episode_successes.append(final_success)

            self._current_rewards = []
            self._current_distances = []
            self._current_success_flags = []

            ep_idx = len(self.episode_rewards)
            if ep_idx % self.log_interval == 0:
                recent_r = self.episode_rewards[-self.log_interval:]
                recent_l = self.episode_lengths[-self.log_interval:]
                recent_s = self.episode_successes[-self.log_interval:]
                recent_d = self.episode_distances[-self.log_interval:]

                avg_r = np.mean(recent_r)
                avg_l = np.mean(recent_l)
                avg_d = np.nanmean(recent_d)
                success_rate = sum(1 for s in recent_s if s) / len(recent_s)

                ts = getattr(self.model, 'num_timesteps', 0)

                print(
                    f'[EP {ep_idx:06d} | ts {ts:08d}] '
                    f'reward={avg_r:8.3f} | '
                    f'len={avg_l:5.0f} | '
                    f'success_rate={success_rate:.2f} | '
                    f'distance={avg_d:.4f}'
                )
        return True


def _print_banner(
    env_mode: str,
    env_id: str,
    enable_gui: bool,
    additional_timesteps: int,
    loaded_model_path: Path,
    output_dir: Path,
    device: str,
    reset_num_timesteps: bool,
) -> None:
    sep = '=' * 54
    print(sep)
    print('  GP7 DDPG — Continue Training')
    print(sep)
    print(f'  ENV_MODE             : {env_mode}')
    print(f'  ENV_ID               : {env_id}')
    print(f'  ENABLE_GUI           : {enable_gui}')
    print(f'  ADDITIONAL TIMESTEPS : {additional_timesteps:,}')
    print(f'  LOADED MODEL         : {loaded_model_path}')
    print(f'  OUTPUT DIR           : {output_dir}')
    print(f'  DEVICE               : {device}')
    print(f'  RESET NUM TIMESTEPS  : {reset_num_timesteps}')
    print(f'  TENSORBOARD          : {"available" if _HAS_TENSORBOARD else "NOT installed"}')
    print(f'  PROGRESS BAR         : {_PROGRESS_BAR}')
    print(sep)
    print()


def _smoke_check(model, vec_env, enable_gui: bool) -> None:
    """
    Run a minimal safety check after loading the model:
    - reset env
    - predict one action (deterministic)
    - step once
    - verify reward is finite
    """
    print('[SMOKE] Running safety check before training...')
    raw_env = vec_env.envs[0].env  # unwrap DummyVecEnv -> Monitor -> raw env
    obs, info = raw_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = raw_env.step(action)

    if not np.isfinite(reward):
        raise ValueError(f'[SMOKE FAILED] reward is not finite: {reward}')
    if 'distance' not in info:
        raise ValueError('[SMOKE FAILED] info missing "distance" key')

    print(f'[SMOKE] OK — action={action}, reward={reward:.4f}, '
          f'terminated={terminated}, truncated={truncated}, '
          f'distance={info["distance"]:.4f}')
    print('[SMOKE] Safety check passed.')
    print()


def _try_load_replay_buffer(model, search_dir: Path, load_flag: bool) -> bool:
    """
    Attempt to load replay buffer from a saved file.

    Searches in this order:
        search_dir / replay_buffer.pkl
        search_dir / model / replay_buffer.pkl

    Returns True if a buffer was loaded, False otherwise.
    """
    if not load_flag:
        print('[INFO] --load-replay-buffer not set; skipping replay buffer.')
        return False

    candidates = [
        search_dir / 'replay_buffer.pkl',
        search_dir / 'model' / 'replay_buffer.pkl',
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                model.load_replay_buffer(str(candidate))
                print(f'[INFO] Replay buffer loaded from: {candidate}')
                return True
            except Exception as e:
                print(f'[WARN] Found replay buffer at {candidate} but failed to load: {e}')
                return False

    print('[WARN] Replay buffer file not found in the loaded run directory.')
    print('      The original training script does not save replay_buffer.pkl by default.')
    print('      Continuing with an empty replay buffer — this is normal.')
    return False


def _save_replay_buffer(model, output_dir: Path) -> Path | None:
    """Save the replay buffer to the output directory. Returns the path, or None if empty."""
    try:
        buf = model.replay_buffer
        if buf is None or (hasattr(buf, 'buffer') and len(buf.buffer) == 0):
            print('[INFO] Replay buffer is empty; not saving.')
            return None
        out_path = output_dir / 'replay_buffer.pkl'
        model.save_replay_buffer(str(out_path))
        print(f'[INFO] Replay buffer saved: {out_path}')
        return out_path
    except Exception as e:
        print(f'[WARN] Failed to save replay buffer: {e}')
        return None


def main() -> int:
    # --- Detect whether CLI arguments were provided ---
    cli_args_provided = len(sys.argv) > 1

    parser = argparse.ArgumentParser(
        description='Continue training a DDPG GP7 model from a saved checkpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to a saved DDPG .zip model file.',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help=(
            'Path to a previous training run directory (e.g. Data/Training/.../run_20260505_010923). '
            'If provided, the script auto-detects model/final_model.zip inside it.'
        ),
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=200000,
        help='Number of additional timesteps to train. Default: 200000',
    )
    parser.add_argument(
        '--enable-gui',
        action='store_true',
        help='Enable PyBullet GUI window. Default: False (headless).',
    )
    parser.add_argument(
        '--save-name',
        type=str,
        default='continued_model.zip',
        help='Filename for the saved continued model. Default: continued_model.zip',
    )
    parser.add_argument(
        '--load-replay-buffer',
        action='store_true',
        help=(
            'Attempt to load replay_buffer.pkl from the loaded run directory. '
            'Only useful if the original training script saved it.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Random seed for continued training. '
            'Default: use seed from original run config.json if available, otherwise 42.'
        ),
    )
    parser.add_argument(
        '--env-mode',
        type=str,
        choices=['Default', 'Collision-Free'],
        default=None,
        help=(
            'Environment mode. Default: auto-detected from original run config.json. '
            'Pass this if the original config.json is not available.'
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device for the neural networks. Default: auto (uses CUDA if available).',
    )

    args = parser.parse_args()

    # --- Apply script config when no CLI args were given and USE_SCRIPT_CONFIG is True ---
    if USE_SCRIPT_CONFIG and not cli_args_provided:
        print('[INFO] Using direct-run script configuration block.')
        args.run_dir = SCRIPT_RUN_DIR
        args.model_path = SCRIPT_MODEL_PATH
        args.timesteps = SCRIPT_TIMESTEPS
        args.enable_gui = SCRIPT_ENABLE_GUI
        args.save_name = SCRIPT_SAVE_NAME
        args.load_replay_buffer = SCRIPT_LOAD_REPLAY_BUFFER
        args.seed = SCRIPT_SEED
        args.env_mode = SCRIPT_ENV_MODE
        args.device = SCRIPT_DEVICE
    else:
        print('[INFO] Using CLI arguments.')

    # --- Validate that a model source is configured ---
    if args.run_dir is None and args.model_path is None:
        print('[FATAL] No model source configured.')
        print('        Edit the USER CONFIG block at the top of this file, set USE_SCRIPT_CONFIG=True,')
        print('        and fill in SCRIPT_RUN_DIR (or SCRIPT_MODEL_PATH).')
        print('        Or provide --run-dir or --model-path via CLI.')
        return 1

    # --- Resolve model path ---
    print('[INFO] Resolving model path...')
    try:
        loaded_model_path = _resolve_model_path(args.model_path, args.run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return 1
    print(f'[INFO] Model resolved: {loaded_model_path}')

    # --- Determine the run directory of the original model ---
    original_run_dir = loaded_model_path.parent.parent  # .../run_YYYYMMDD_HHMMSS/model/final_model.zip

    # --- Detect env_mode ---
    env_mode = args.env_mode
    if env_mode is None:
        env_mode = _detect_env_mode_from_config(original_run_dir)
        if env_mode is None:
            print('[WARN] Could not detect env_mode from original config.json.')
            print('      Defaulting to "Default".')
            print('      Use --env-mode to override if needed.')
            env_mode = 'Default'

    env_id = f'YaskawaGP7ReachPyBullet-{env_mode}-v0'
    print(f'[INFO] Environment: {env_id}')

    # --- Detect seed ---
    seed = args.seed
    if seed is None:
        cfg = _load_config_json(original_run_dir)
        seed = cfg.get('seed', 42)
        if seed == 42 and 'seed' not in cfg:
            print('[INFO] No seed in original config; using seed=42.')

    # --- Build output directory ---
    output_dir = _build_output_dir(CONST_PROJECT_FOLDER, env_mode)
    model_dir = output_dir / 'model'
    log_dir = output_dir / 'logs'
    tb_dir = output_dir / 'tensorboard'
    for d in [model_dir, log_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Output directory: {output_dir}')

    # --- Create environment (same way as train_ddpg_gp7.py) ---
    print(f'[INFO] Creating environment: {env_id}  (enable_gui={args.enable_gui})')
    raw_env = gym.make(env_id, enable_gui=args.enable_gui)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.reset(seed=seed)

    gym_environment = stable_baselines3.common.monitor.Monitor(raw_env, str(log_dir))
    gym_environment = stable_baselines3.common.vec_env.DummyVecEnv([lambda: gym_environment])

    # --- Load model ---
    print(f'[INFO] Loading model from: {loaded_model_path}')
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = stable_baselines3.DDPG.load(
            str(loaded_model_path),
            env=gym_environment,
            device=device,
        )
    except Exception as e:
        print(f'[FATAL] Failed to load model: {e}')
        gym_environment.close()
        return 1

    print(f'[INFO] Model loaded. Current num_timesteps: {model.num_timesteps}')

    # --- Replay buffer ---
    rb_loaded = _try_load_replay_buffer(model, original_run_dir, args.load_replay_buffer)

    # --- Logger ---
    _log_formats = ['stdout', 'csv']
    if _HAS_TENSORBOARD:
        _log_formats.append('tensorboard')
    new_logger = stable_baselines3.common.logger.configure(str(log_dir), _log_formats)
    model.set_logger(new_logger)

    # --- Safety smoke check ---
    try:
        _smoke_check(model, gym_environment, args.enable_gui)
    except Exception as e:
        print(f'[SMOKE FAILED] {e}')
        print('[FATAL] Aborting. The loaded model or environment may be incompatible.')
        gym_environment.close()
        return 1

    # --- Save config.json ---
    config = {
        'mode': 'continued_training',
        'env_mode': env_mode,
        'env_id': env_id,
        'algorithm': 'DDPG',
        'loaded_model_path': str(loaded_model_path),
        'loaded_run_dir': str(original_run_dir),
        'replay_buffer_loaded': rb_loaded,
        'additional_timesteps': args.timesteps,
        'reset_num_timesteps': False,
        'enable_gui': args.enable_gui,
        'seed': int(seed),
        'device': device,
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat(),
        'reward_settings': {
            'type': 'dense_distance_with_optional_collision_soft_penalty',
            'default_reward': '-distance',
            'collision_free_reward': '-(distance + collision_penalty * collision_obj_penalty_threshold)',
            'collision_penalty_formula': '1 / (1 + distance_to_obstacle)',
            'collision_obj_penalty_threshold': 0.01,
            'hard_collision_penalty': -5.0,
            'hard_failure_penalty': -1.0,
            'success_bonus': 0.0,
        },
        'ik_settings': {
            'use_orientation': False,
            'ik_position_tolerance': 0.01,
            'note': 'position-only IK — matches GP7ReachPyBulletEnv.__ik_props',
        },
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # --- Banner ---
    _print_banner(
        env_mode=env_mode,
        env_id=env_id,
        enable_gui=args.enable_gui,
        additional_timesteps=args.timesteps,
        loaded_model_path=loaded_model_path,
        output_dir=output_dir,
        device=device,
        reset_num_timesteps=False,
    )

    # --- Train ---
    callback = EpisodeCallback(log_interval=CONST_LOG_INTERVAL)

    print(f'[INFO] Continuing training for {args.timesteps:,} additional timesteps.')
    print('[INFO] Note: reset_num_timesteps=False — training curves continue from prior run.')
    print()
    t_0 = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=callback,
        progress_bar=_PROGRESS_BAR,
        log_interval=CONST_LOG_INTERVAL,
    )

    elapsed = time.time() - t_0
    final_timesteps = model.num_timesteps

    # --- Save model ---
    model_path_out = model_dir / args.save_name
    model.save(str(model_path_out))
    print()
    print(f'[INFO] Continued model saved: {model_path_out}')

    # --- Save replay buffer ---
    replay_buffer_out = _save_replay_buffer(model, output_dir)

    # --- Summary ---
    n = len(callback.episode_rewards)
    print()
    print('=' * 54)
    print('  Training Summary')
    print('=' * 54)
    print(f'  Loaded model    : {loaded_model_path}')
    print(f'  Original run dir : {original_run_dir}')
    print(f'  Replay buf loaded: {rb_loaded}')
    print(f'  Additional steps: {args.timesteps:,}')
    print(f'  Final timesteps : {final_timesteps:,}')
    print(f'  Episodes logged : {n}')
    print(f'  Elapsed time    : {elapsed:.1f}s')
    print(f'  Output model    : {model_path_out}')
    if replay_buffer_out:
        print(f'  Replay buffer   : {replay_buffer_out}')
    print(f'  TensorBoard     : {tb_dir}')
    print(f'  Config          : {output_dir / "config.json"}')
    print('=' * 54)

    RoLE.Utilities.File_IO.Save(
        str(log_dir / 'time'),
        f'Training time: {elapsed:0.05f}s\nFinal timesteps: {final_timesteps}',
        'txt',
        '',
    )

    gym_environment.close()
    print('[INFO] Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
