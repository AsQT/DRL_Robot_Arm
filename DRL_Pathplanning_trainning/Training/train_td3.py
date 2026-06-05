"""
Twin Delayed DDPG (TD3) training entry point.

Loads the unified ``config/environment.yaml`` and trains a TD3 (or DDPG / SAC)
model on the Cartesian path planning task.

Run from-scratch::

    python Training/train_td3.py --config config/environment.yaml

Fine-tune an existing model::

    python Training/train_td3.py --config config/environment.yaml \\
        --load-model Data/Training/.../final_model.zip

For GPU training, set ``training.device: cuda`` in the config YAML.
For more envs, increase ``training.n_envs`` in the YAML.
"""

import argparse
import sys
import warnings
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drl_pathplanning.gymnasium.config import load_config, Config
from drl_pathplanning.training import train_sb3_model


# --------------------------------------------------------------------------- #
# Deprecated CLI arguments mapped to their config YAML equivalents
# --------------------------------------------------------------------------- #
_DEPRECATED_ARGS = {
    "--algo-config":      "training.algorithm + td3.*  (set in config/environment.yaml)",
    "--env-mode-override": "environment.mode          (set in config/environment.yaml)",
    "--total-timesteps":  "training.total_timesteps   (set in config/environment.yaml)",
    "--n-envs":           "training.n_envs             (set in config/environment.yaml)",
    "--vec-env":          "training.vec_env_type       (set in config/environment.yaml)",
    "--progress-bar":     "training.progress_bar        (set in config/environment.yaml)",
    "--seed":             "training.seed               (set in config/environment.yaml)",
    "--curriculum-config": "curriculum.enabled + stages (set in config/environment.yaml)",
}


def _warn_deprecated(arg: str, suggestion: str) -> None:
    warnings.warn(
        f"Deprecated: {arg} is no longer accepted.\n"
        f"  Please edit the config YAML instead.\n"
        f"  → {suggestion}",
        FutureWarning,
        stacklevel=3,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD3 on Cartesian Path Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR.parent / "config" / "environment.yaml",
        help="Path to the unified config YAML [default: config/environment.yaml]",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Path to a .zip model to fine-tune",
    )
    parser.add_argument(
        "--load-replay-buffer",
        type=Path,
        default=None,
        help="Path to replay_buffer.pkl to load before training",
    )
    parser.add_argument(
        "--gui",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable PyBullet GUI during training (short debug runs only). "
             "Requires training.n_envs=1 in YAML. Default: false.",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Print per-episode results during GUI training. Default: false.",
    )
    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.0,
        help="Sleep time (seconds) between steps in GUI mode. Default: 0.0.",
    )
    parser.add_argument(
        "--render-first-episodes",
        type=int,
        default=0,
        help="Render only the first N episodes in GUI mode. "
             "0 = render all episodes. Default: 0.",
    )
    parser.add_argument(
        "--safety-warmup-timesteps",
        type=int,
        default=None,
        help="Global timestep warmup before EarlyStopSafetyCallback checks success rate. "
             "Default: 50000 for fine-tune, 0 for from-scratch.",
    )
    parser.add_argument(
        "--safety-warmup-episodes",
        type=int,
        default=None,
        help="Episode warmup before EarlyStopSafetyCallback checks success rate. "
             "Default: 200 for fine-tune, 0 for from-scratch.",
    )
    parser.add_argument(
        "--safety-min-success-rate",
        type=float,
        default=None,
        help="Override minimum success rate threshold for EarlyStopSafetyCallback. "
             "Default: 0.90.",
    )

    args = parser.parse_args()

    # Check for deprecated arguments and warn
    for i, arg in enumerate(sys.argv):
        if arg in _DEPRECATED_ARGS:
            _warn_deprecated(arg, _DEPRECATED_ARGS[arg])

    return args


def main() -> None:
    args = _parse_args()

    # Load unified config
    cfg: Config = load_config(args.config)

    # Script dictates algorithm, overriding whatever is in the YAML
    script_algo = "TD3"
    cfg.training.algorithm = script_algo

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Algorithm requested by script: {script_algo}")
    print(f"[INFO] Training algorithm: {script_algo}")

    train_sb3_model(
        algorithm=script_algo,
        env_config=cfg,
        total_timesteps=cfg.training.total_timesteps,
        seed=cfg.training.seed,
        n_envs=cfg.training.n_envs,
        vec_env_type=cfg.training.vec_env_type,
        progress_bar=cfg.training.progress_bar,
        load_model=args.load_model,
        load_replay_buffer=args.load_replay_buffer,
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
