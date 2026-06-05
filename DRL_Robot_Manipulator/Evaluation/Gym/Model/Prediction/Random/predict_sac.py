"""
Random-target prediction script for SAC.

Runs N episodes on the GP7 reach environment with randomly sampled targets,
logging per-episode results (episode index, success, reward, length, distance)
to a .txt file.

Edit USER CONFIG below to set the model path, then run:
    python Evaluation/Gym/Model/Prediction/Random/predict_sac.py
"""
# =============================================================================
# PATH RESOLUTION — must come before any project imports
# =============================================================================
from pathlib import Path
import sys
import os

_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "Training").exists():
            return parent
    raise RuntimeError(
        f"[FATAL] Could not locate project root from {start}.\n"
        "        Expected to find folders: src/ and Training/."
    )


_PROJECT_ROOT = _find_project_root(_SCRIPT_DIR)
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

print(f"[INFO] Project root : {_PROJECT_ROOT}")
print(f"[INFO] src path     : {_SRC_DIR}")

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import RoLE.Parameters.Robot as Parameters
import RoLE.Utilities.File_IO
import Industrial_Robotics_Gym

# =============================================================================
# USER CONFIG — edit here
# =============================================================================
CONST_ALGORITHM = "SAC"
CONST_ENV_MODE = "Default"

# --- Model source (pick one) ---
# Option A: direct path to .zip file
CONST_MODEL_PATH = r"Data/Training/Environment_Default/SAC/YASKAWA_GP7/run_20260505_010923/model/final_model.zip"

# Option B: run directory — auto-detects model inside it
CONST_RUN_DIR = None

# --- Prediction settings ---
CONST_ENABLE_GUI = True
CONST_DETERMINISTIC = True
CONST_ACTION_STEP = 0.01      # must match training (0.01 m)
CONST_DISTANCE_THRESH = 0.01   # must match training (0.01 m)
CONST_MAX_STEPS = 200
CONST_NUM_EPISODES = 10
CONST_SEED = 42
CONST_VERBOSE_EPISODE = True

# --- Output ---
CONST_OUTPUT_PATH = None
# =============================================================================

# --- Device ---
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'


def _resolve_model_path(model_path, run_dir, algorithm):
    """
    Resolve a model .zip path from a direct path or a run directory.
    Relative paths are resolved against _PROJECT_ROOT.

    Search order for run directories:
        1. run_dir / "model" / "final_model.zip"
        2. run_dir / "final_model.zip"
        3. run_dir / "model" / "continued_model.zip"
        4. run_dir / "continued_model.zip"
    """
    def _resolve(p: str, base: Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        return (base / p).resolve()

    if model_path is not None:
        p = _resolve(model_path, _PROJECT_ROOT)
        if not p.exists():
            raise FileNotFoundError(f'[FATAL] Model file not found: {p}\n'
                                    f'        (CONST_MODEL_PATH = {model_path!r})')
        return p

    if run_dir is not None:
        rd = _resolve(run_dir, _PROJECT_ROOT)
        if not rd.exists():
            raise FileNotFoundError(f'[FATAL] Run directory not found: {rd}\n'
                                    f'        (CONST_RUN_DIR = {run_dir!r})')
        candidates = [
            rd / 'model' / 'final_model.zip',
            rd / 'final_model.zip',
            rd / 'model' / 'continued_model.zip',
            rd / 'continued_model.zip',
        ]
        for c in candidates:
            if c.exists():
                print(f'[INFO] Auto-detected model: {c}')
                return c
        raise FileNotFoundError(
            f'[FATAL] No model file found in: {rd}\n'
            f'        Expected one of: final_model.zip, continued_model.zip\n'
            f'        in {rd}/ or {rd}/model/'
        )

    raise FileNotFoundError(
        '[FATAL] No model source configured.\n'
        '        Set CONST_MODEL_PATH or CONST_RUN_DIR in the USER CONFIG block above.'
    )


def _resolve_output_path(project_root, algorithm, env_mode, n_episodes):
    """Resolve or auto-generate the output .txt path."""
    if CONST_OUTPUT_PATH is not None:
        return Path(CONST_OUTPUT_PATH)
    pred_dir = project_root / 'Data' / 'Prediction' / f'Environment_{env_mode}' / algorithm / Parameters.YASKAWA_GP7_Str.Name
    pred_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir / f'random_N_{n_episodes}.txt'


def main():
    model_zip = _resolve_model_path(CONST_MODEL_PATH, CONST_RUN_DIR, CONST_ALGORITHM)
    print(f'[INFO] Model: {model_zip}')
    print(f'[INFO] Env mode: {CONST_ENV_MODE}')
    print(f'[INFO] GUI: {CONST_ENABLE_GUI}')
    print(f'[INFO] Deterministic: {CONST_DETERMINISTIC}')
    print(f'[INFO] Device: {DEVICE}')
    print(f'[INFO] Episodes: {CONST_NUM_EPISODES}')

    env_id = f'YaskawaGP7ReachPyBullet-{CONST_ENV_MODE}-v0'
    env = gym.make(
        env_id,
        enable_gui=CONST_ENABLE_GUI,
        action_step=CONST_ACTION_STEP,
        distance_thresh=CONST_DISTANCE_THRESH,
        max_episode_steps=CONST_MAX_STEPS,
    )

    model = SAC.load(str(model_zip), device=DEVICE)

    output_path = _resolve_output_path(_PROJECT_ROOT, CONST_ALGORITHM, CONST_ENV_MODE, CONST_NUM_EPISODES)
    if output_path.exists():
        try:
            os.remove(output_path)
        except PermissionError:
            pass

    print(f'[INFO] Output: {output_path}')
    print()

    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_successes = []

    for ep_idx in range(1, CONST_NUM_EPISODES + 1):
        obs, info = env.reset(seed=CONST_SEED + ep_idx)
        episode_reward = 0.0
        episode_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=CONST_DETERMINISTIC)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1

            if terminated or truncated:
                final_distance = float(info.get('distance', float('nan')))
                final_success = bool(info.get('is_success', False))

                RoLE.Utilities.File_IO.Save(
                    str(output_path),
                    np.array([ep_idx, int(final_success), episode_reward, episode_length, final_distance]),
                    'txt', ','
                )

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_distances.append(final_distance)
                episode_successes.append(final_success)

                if CONST_VERBOSE_EPISODE:
                    print(
                        f'[EP {ep_idx:03d}] '
                        f'reward={episode_reward:9.3f} | '
                        f'steps={episode_length:4d} | '
                        f'success={final_success} | '
                        f'distance={final_distance:.4f}'
                    )
                break

    env.close()

    n = len(episode_rewards)
    if n > 0:
        mean_reward = float(np.mean(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        mean_dist = float(np.nanmean(episode_distances))
        success_rate = sum(episode_successes) / n
        print()
        print('=' * 56)
        print('  PREDICTION SUMMARY')
        print('=' * 56)
        print(f'  Model         : {model_zip}')
        print(f'  Env ID        : {env_id}')
        print(f'  Episodes      : {n}')
        print(f'  Success rate  : {success_rate:.2%}  ({sum(episode_successes)}/{n})')
        print(f'  Mean reward   : {mean_reward:.3f}')
        print(f'  Mean length   : {mean_length:.1f}')
        print(f'  Mean dist     : {mean_dist:.4f}  m')
        print(f'  Deterministic : {CONST_DETERMINISTIC}')
        print(f'  GUI           : {CONST_ENABLE_GUI}')
        print(f'  Output file   : {output_path}')
        print('=' * 56)
    print('[INFO] Done.')


if __name__ == '__main__':
    sys.exit(main())
