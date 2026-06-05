"""
Twin Delayed DDPG (DDPG) training entry point.

Loads config/environment.yaml and trains a DDPG model on the Cartesian path planning task.

Run from scratch::

    python Training/train_ddpg.py --config config/environment.yaml

For GPU training, set ``training.device: cuda`` in the config YAML.
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drl_pathplanning.gymnasium.config import load_config
from drl_pathplanning.training import train_sb3_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDPG on Cartesian Path Planning",
    )
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
        help="Enable PyBullet GUI during training. Default: false.",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Print per-episode results during GUI training. Default: false.",
    )
    parser.add_argument(
        "--render-first-episodes",
        type=int,
        default=0,
        help="Render only the first N episodes in GUI mode. Default: 0.",
    )
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.0,
        help="Sleep time (seconds) between steps in GUI mode. Default: 0.0.",
    )
    parser.add_argument(
        "--safety-warmup-timesteps",
        type=int,
        default=None,
        help="Global timestep warmup before safety checks. Default: 0 for from-scratch.",
    )
    parser.add_argument(
        "--safety-warmup-episodes",
        type=int,
        default=None,
        help="Episode warmup before safety checks. Default: 0 for from-scratch.",
    )
    parser.add_argument(
        "--safety-min-success-rate",
        type=float,
        default=None,
        help="Override minimum success rate threshold. Default: 0.90.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # Script dictates algorithm, overriding whatever is in the YAML
    script_algo = "DDPG"
    cfg.training.algorithm = script_algo

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Algorithm requested by script: {script_algo}")
    print(f"[INFO] Training algorithm: {script_algo}")

    train_sb3_model(
        algorithm=script_algo,
        env_config=cfg,
        total_timesteps=args.timesteps,
        seed=cfg.training.seed,
        n_envs=cfg.training.n_envs,
        vec_env_type=cfg.training.vec_env_type,
        progress_bar=cfg.training.progress_bar,
        load_model=None,
        load_replay_buffer=None,
        gui=args.gui.lower() == "true",
        show=args.show.lower() == "true",
        render_sleep=float(args.render_sleep),
        render_first_episodes=int(args.render_first_episodes),
        safety_warmup_timesteps=args.safety_warmup_timesteps,
        safety_warmup_episodes=args.safety_warmup_episodes,
        safety_min_success_rate=args.safety_min_success_rate,
        config_path=args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
