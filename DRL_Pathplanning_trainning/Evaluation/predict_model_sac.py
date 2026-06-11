"""
SAC model evaluation script — loads from config/environment.yaml.

Usage::

    python Evaluation/predict_model_sac.py --episodes 10 --gui false --show false

    python Evaluation/predict_model_sac.py \\
        --config config/environment.yaml \\
        --run Data/Training/SAC/FRAME_ONLY/run_xxx \\
        --episodes 10"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import SAC

from Evaluation.predict_common import (
    get_project_root,
    resolve_model_path,
    create_eval_env,
    build_viewer,
    update_viewer_obstacle,
    run_random_episodes,
    run_static_episodes,
    print_full_summary,
    add_common_args,
    parse_common_args,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALGORITHM = "SAC"

# --------------------------------------------------------------------------- #
# USER CONFIG — edit these values
# --------------------------------------------------------------------------- #
PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "environment.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "Data" / "Training" / "SAC" / "FRAME_ONLY"

CUSTOM_MODEL_ZIP = ""
# --------------------------------------------------------------------------- #


def _latest_run_with_model(base_dir: Path) -> Path:
    """Return the newest run directory that contains a known model zip."""
    if not base_dir.exists():
        return base_dir

    run_dirs = sorted(
        (p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        if resolve_model_path(run_dir, CUSTOM_MODEL_ZIP) is not None:
            return run_dir
    return base_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained {ALGORITHM} model",
    )
    add_common_args(parser)
    args = parser.parse_args()

    if args.config is None:
        args.config = DEFAULT_CONFIG_PATH
    if args.run is None:
        args.run = str(_latest_run_with_model(DEFAULT_RUN_DIR))

    return args


def main() -> int:
    args = _parse_args()
    p = parse_common_args(args)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    run_path = Path(args.run).expanduser().resolve()
    if not run_path.exists():
        print(f"[ERROR] Run directory not found: {run_path}")
        return 1

    custom = p["custom_model"] or CUSTOM_MODEL_ZIP
    model_path = resolve_model_path(run_path, custom)
    if model_path is None:
        print(f"[ERROR] Model not found in: {run_path / 'model'}")
        print(f"  Checked:")
        if custom:
            print(f"    - model/{custom}")
        print(f"    - model/best_model.zip")
        print(f"    - model/final_model.zip")
        return 1

    print("=" * 60)
    print(f"  {ALGORITHM} Model Evaluation")
    print("=" * 60)
    print(f"[INFO] Algorithm        : {ALGORITHM}")
    print(f"[INFO] Config          : {config_path}")
    print(f"[INFO] Env source      : YAML config (not inline)")
    print(f"[INFO] Run dir         : {run_path}")
    print(f"[INFO] Model           : {model_path.name}")
    print(f"[INFO] VecNormalize    : disabled")
    print(f"[INFO] Device          : {DEVICE}")
    print(f"[INFO] Episodes        : {p['num_episodes']}")
    print(f"[INFO] Mode            : {p['mode']}")
    print(f"[INFO] GUI             : {p['gui']}")
    print(f"[INFO] Show            : {p['show']}")
    print(f"[INFO] Deterministic   : {p['deterministic']}")
    print(f"[INFO] Step sleep      : {p['sleep']}s")
    print("=" * 60)

    start_pos = None
    if p["start_mode"] == "fixed" and args.start is not None:
        start_pos = tuple(args.start)

    env, cfg = create_eval_env(
        config_path=config_path,
        start_mode=p["start_mode"],
        start_pos=start_pos,
        target_pos=p["target_pos"],
    )
    print(f"[INFO] Env start_mode : {p['start_mode']}")
    print(f"[INFO] Env obstacle   : {cfg.obstacle.resolved_mode}")
    zmax = float(cfg.workspace.max_np[2])
    print(f"[INFO] Env workspace  : z_max={zmax:.3f}m")

    model = SAC.load(str(model_path), env=None, device=DEVICE)
    print(f"[INFO] Model loaded OK")

    viewer = build_viewer(cfg, p["gui"])

    output_dir = (
        _SCRIPT_DIR.parent / "Data" / "Prediction"
        / "Environment_Default" / ALGORITHM / "FRAME_ONLY"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if p["mode"] == "random":
        results = run_random_episodes(
            model=model, env=env, cfg=cfg,
            num_episodes=p["num_episodes"],
            deterministic=p["deterministic"],
            sleep=p["sleep"],
            viewer=viewer,
            viewer_update_fn=update_viewer_obstacle,
        )
    else:
        results = run_static_episodes(
            model=model, env=env, cfg=cfg,
            num_episodes=p["num_episodes"],
            deterministic=p["deterministic"],
            sleep=p["sleep"],
            viewer=viewer,
            viewer_update_fn=update_viewer_obstacle,
            output_dir=output_dir,
        )

    if p["show"]:
        print_full_summary(results, p["num_episodes"], output_dir)

    env.close()
    if viewer is not None:
        viewer.close()

    print(f"[INFO] Done. Run: {run_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
