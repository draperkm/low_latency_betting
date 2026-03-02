"""Build the training dataset from raw Sackmann CSV files.

Usage:
    python -m tennis_predictor.preprocessing
    python -m tennis_predictor.preprocessing --data-dir path/to/csvs --output-dir path/to/output
"""

import argparse
from pathlib import Path

from .pipeline import TrainingPipeline

# preprocessing/ → tennis_predictor/ → src/ → project_root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "data_download" / "training" / "raw" / "tennis_pointbypoint"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "data_download" / "training" / "processed"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training dataset from raw PBP data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing Sackmann CSV files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for processed CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline(data_dir=args.data_dir, output_dir=args.output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
