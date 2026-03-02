"""Train XGBoost model and export to ONNX.

Usage:
    python -m tennis_predictor.training
    python -m tennis_predictor.training --train-sample-frac 1.0 --n-trials 20
    python -m tennis_predictor.training --n-trials 0   # skip tuning, use defaults
"""

import argparse
from pathlib import Path

from .config import TrainingConfig
from .pipeline import ModelTrainingPipeline

# training/ -> tennis_predictor/ -> src/ -> project_root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "data_download" / "training" / "processed"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def _discover_latest_run(processed_dir: Path) -> Path:
    """Find the most recent preprocessed_* folder and return training_dataset.csv path."""
    run_dirs = sorted(processed_dir.glob("preprocessed_*"), reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No preprocessed_* folders in {processed_dir}")
    return run_dirs[0] / "training_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost and export ONNX model.")
    parser.add_argument(
        "--data-path", type=Path, default=None,
        help="Path to training_dataset.csv (default: auto-discover latest run)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help=f"Output directory for model artifacts (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--train-sample-frac", type=float, default=0.001,
        help="Fraction of training matches to use (default: 0.001)",
    )
    parser.add_argument(
        "--tune-sample-frac", type=float, default=0.01,
        help="Fraction of training matches for Optuna tuning (default: 0.01)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of Optuna trials, 0 to skip tuning (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    data_path = args.data_path or _discover_latest_run(DEFAULT_PROCESSED_DIR)

    config = TrainingConfig(
        train_sample_frac=args.train_sample_frac,
        tune_sample_frac=args.tune_sample_frac,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    pipeline = ModelTrainingPipeline(
        data_path=data_path,
        model_dir=args.model_dir,
        config=config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
