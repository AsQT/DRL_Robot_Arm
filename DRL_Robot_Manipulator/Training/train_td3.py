"""
Twin Delayed DDPG (TD3) training script for the Yaskawa GP7 robot.

Trains a TD3 agent on the YaskawaGP7ReachPyBullet Collision-Free environment,
which includes a cube obstacle the robot must navigate around while reaching
randomly sampled targets in the configuration space.

TD3 addresses the overestimation bias of DDPG by using twin Q-networks and
delayed policy updates.  It requires an explicit action noise model for
exploration, which is provided by the NormalActionNoise wrapper.

Outputs are written to timestamped directories under Data/Training/, including:
    - progress.csv          (SB3 training metrics)
    - monitor.csv           (episode-level rewards and lengths)
    - final_model.zip       (saved policy)
    - config.json           (experiment configuration)
    - tensorboard/          (optional, if tensorboard is installed)

Each run is reproducible via the SEED constant.  See the top-level constants
below to configure the environment mode, algorithm, GUI, and hyperparameters.
"""
# System (Default)
import random
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Resolve the src/ directory relative to this script file, so that imports work
# from both "python Training/train_td3.py" and "cd Training && python train_td3.py".
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# PyTorch (needed for torch.manual_seed during environment setup)
import torch
# Time (Time access and conversions)
import time
# Gymnasium (Developing and comparing reinforcement learning algorithms) [pip3 install gymnasium]
import gymnasium as gym
# Stable-Baselines3 (A set of implementations of reinforcement learning algorithms in PyTorch) [pip3 install stable-baselines3]
import stable_baselines3
import stable_baselines3.common.noise
import stable_baselines3.common.logger
import stable_baselines3.common.monitor
import stable_baselines3.common.vec_env
import stable_baselines3.common.callbacks
# Custom Lib.:
import RoLE.Parameters.Robot as Parameters
import RoLE.Utilities.File_IO
import Industrial_Robotics_Gym
import Industrial_Robotics_Gym.Utilities

# Locate the path to the project folder (derive from script file, not cwd).
CONST_PROJECT_FOLDER = Path(_SCRIPT_DIR).parent.resolve()

# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str

# The name of the environment mode.
CONST_ENV_MODE = 'Collision-Free'

# The name of the reinforcement learning algorithm.
CONST_ALGORITHM = 'TD3'

# True for visual debugging, False for long training.
ENABLE_GUI = True

# The GP7 PyBullet environment ID (derived from mode to ensure consistency).
CONST_ENV_ID = f'YaskawaGP7ReachPyBullet-{CONST_ENV_MODE}-v0'

# Training hyperparameters.
CONST_TOTAL_TIMESTEPS = 100000
CONST_LOG_INTERVAL = 10  # episodes

# Reproducibility seed — change this value to re-run experiments deterministically.
SEED = 42

# --- Unique run ID: each training run gets its own timestamped directory ---
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Structured experiment directory ---
#   Data/Training/Environment_{ENV_MODE}/{ALGORITHM}/YASKAWA_GP7/run_{RUN_ID}/{logs,model,tensorboard}
_BASE_DIR = (
    Path(CONST_PROJECT_FOLDER)
    / 'Data' / 'Training'
    / f'Environment_{CONST_ENV_MODE}'
    / CONST_ALGORITHM
    / 'YASKAWA_GP7'
)
_RUN_DIR = _BASE_DIR / f'run_{RUN_ID}'

MODEL_DIR = _RUN_DIR / 'model'
LOG_DIR   = _RUN_DIR / 'logs'
TB_DIR    = _RUN_DIR / 'tensorboard'

# Whether progress_bar=True is available (requires tqdm AND rich).
try:
    import tqdm as _  # noqa: F401
    import rich  # noqa: F401
    _PROGRESS_BAR = True
except ImportError:
    _PROGRESS_BAR = False

# Whether TensorBoard is available for logging.
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

"""
Notes:
    A command to kill all Python processes within the GPU.
    $ ../> sudo killall -9 python

    Start training the model.
    $ ../> python Training/train_td3.py
    $ cd Training && python train_td3.py
"""


class EpisodeCallback(stable_baselines3.common.callbacks.BaseCallback):
    """
    Clean episode-level progress logger for SB3 training.

    Tracks cumulative reward, episode length, and final distance for each episode.
    Prints a formatted progress line every `log_interval` episodes showing
    rolling averages and the success rate (fraction of recent episodes where
    the robot reached within 0.01 m of the target).
    """

    def __init__(self, log_interval: int = CONST_LOG_INTERVAL):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self._current_rewards = []
        self._current_distances = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        self._current_rewards.append(reward)
        if 'distance' in info:
            self._current_distances.append(info['distance'])

        if self.locals['dones'][0]:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_lengths.append(len(self._current_rewards))
            self.episode_distances.append(
                self._current_distances[-1] if self._current_distances else float('nan'))
            self._current_rewards = []
            self._current_distances = []

            ep_idx = len(self.episode_rewards)
            if ep_idx % self.log_interval == 0:
                recent_r = self.episode_rewards[-self.log_interval:]
                recent_l = self.episode_lengths[-self.log_interval:]
                recent_d = self.episode_distances[-self.log_interval:]

                avg_r = np.mean(recent_r)
                avg_l = np.mean(recent_l)
                avg_d = np.nanmean(recent_d)
                success_rate = sum(1 for d in recent_d if d < 0.01) / len(recent_d)

                print(
                    f'[EP {ep_idx:06d}] '
                    f'reward={avg_r:8.3f} | '
                    f'len={avg_l:5.0f} | '
                    f'success_rate={success_rate:.2f} | '
                    f'distance={avg_d:.4f}'
                )
        return True


def _print_banner() -> None:
    """Print a formatted header with the current run configuration."""
    sep = '========================================'
    print(f'{sep}')
    print(f'  GP7 {CONST_ALGORITHM} Training')
    print(f'{sep}')
    print(f'  ENV_MODE     : {CONST_ENV_MODE}  [{"obstacle DISABLED" if CONST_ENV_MODE == "Default" else "obstacle ENABLED"}]')
    print(f'  ENV_ID       : {CONST_ENV_ID}')
    print(f'  ENABLE_GUI   : {ENABLE_GUI}')
    print(f'  TIMESTEPS    : {CONST_TOTAL_TIMESTEPS:,}')
    print(f'  RUN_ID       : {RUN_ID}')
    print(f'  RUN_DIR      : {_RUN_DIR}')
    print(f'  MODEL_DIR    : {MODEL_DIR}')
    print(f'  LOG_DIR      : {LOG_DIR}')
    print(f'  TENSORBOARD  : {TB_DIR}  [{(_HAS_TENSORBOARD and "available") or "NOT installed"}]')
    print(f'  PROGRESS_BAR : {_PROGRESS_BAR}')
    print(f'{sep}')
    print()


def main() -> None:
    """
    Run the full training pipeline end-to-end.

    Creates output directories, seeds the environment and libraries, builds the
    TD3 model, trains for CONST_TOTAL_TIMESTEPS, saves the final policy and
    a training summary, then cleans up the environment.
    """
    Robot_Str = CONST_ROBOT_TYPE

    # Create output directories (safe: never overwrites, creates fresh run dir).
    for d in [LOG_DIR, MODEL_DIR, TB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Save experiment config.
    config = {
        'env_mode': CONST_ENV_MODE,
        'algorithm': CONST_ALGORITHM,
        'timesteps': CONST_TOTAL_TIMESTEPS,
        'run_id': RUN_ID,
        'ik_settings': {
            'use_orientation': False,
            'ik_position_tolerance': 0.01,
            'note': 'position-only IK — matches GP7ReachPyBulletEnv.__ik_props',
        },
    }
    with open(_RUN_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # --- SB3 logger: stdout + CSV + TensorBoard ---
    _log_formats = ['stdout', 'csv']
    if _HAS_TENSORBOARD:
        _log_formats.append('tensorboard')
    new_logger = stable_baselines3.common.logger.configure(str(LOG_DIR), _log_formats)

    # Create environment.
    print(f'[INFO] Creating environment: {CONST_ENV_ID}  (enable_gui={ENABLE_GUI})')
    raw_env = gym.make(CONST_ENV_ID, enable_gui=ENABLE_GUI)

    # Seed all random sources for reproducible training runs.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    raw_env.reset(seed=SEED)

    # Monitor records episode rewards/lengths to monitor.csv.
    # DummyVecEnv wraps it as a vectorized env (required by SB3 algorithms).
    gym_environment = stable_baselines3.common.monitor.Monitor(raw_env, str(LOG_DIR))
    gym_environment = stable_baselines3.common.vec_env.DummyVecEnv([lambda: gym_environment])

    # Action noise.
    n_action = gym_environment.action_space.shape[-1]
    action_noise = stable_baselines3.common.noise.NormalActionNoise(
        mean=np.zeros(n_action), sigma=0.1 * np.ones(n_action)
    )

    # Build model.
    model = stable_baselines3.TD3(
        policy='MlpPolicy',
        env=gym_environment,
        gamma=0.95,
        learning_rate=0.001,
        action_noise=action_noise,
        device='cuda',
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
    )
    model.set_logger(new_logger)

    # Print banner.
    _print_banner()

    callback = EpisodeCallback(log_interval=CONST_LOG_INTERVAL)

    print('[INFO] Training started.')
    print()
    t_0 = time.time()

    # Train.
    model.learn(
        total_timesteps=CONST_TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=_PROGRESS_BAR,
        log_interval=CONST_LOG_INTERVAL,
    )

    elapsed = time.time() - t_0

    # Save model.
    model_path = MODEL_DIR / 'final_model'
    model.save(str(model_path))
    print()
    print(f'[INFO] Training complete. Elapsed: {elapsed:.1f}s')
    print(f'[INFO] Model saved: {model_path}.zip')

    # Summary.
    n = len(callback.episode_rewards)
    if n >= 2:
        early = np.mean(callback.episode_rewards[:max(1, n // 3)])
        late  = np.mean(callback.episode_rewards[-max(1, n // 3):])
        success_count = sum(1 for d in callback.episode_distances if d < 0.01)
        print(f'[SUMMARY] Episodes: {n}')
        print(f'[SUMMARY] Early avg reward (first 1/3): {early:.3f}')
        print(f'[SUMMARY] Late  avg reward (last  1/3): {late:.3f}')
        print(f'[SUMMARY] Improvement: {late - early:+.3f}')
        print(f'[SUMMARY] Success rate (< 1cm): {success_count}/{n} = {success_count / n * 100:.1f}%')

    print(f'[INFO] Training time: {elapsed:0.05f}s')
    RoLE.Utilities.File_IO.Save(str(LOG_DIR / 'time'), f'Training time: {elapsed:0.05f}s', 'txt', '')

    gym_environment.close()
    print('[INFO] Done.')


if __name__ == '__main__':
    sys.exit(main())
