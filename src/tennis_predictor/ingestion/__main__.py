"""Run the ingestion pipeline from the command line.

Usage:
    python -m tennis_predictor.ingestion
    python -m tennis_predictor.ingestion --match-id <id>
    python -m tennis_predictor.ingestion --processed-csv path/to/full_game_states.csv
    python -m tennis_predictor.ingestion --match-id <id> --queue-max-size 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .pipeline import IngestionConfig, IngestionPipeline

# ingestion/ → tennis_predictor/ → src/ → project_root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "data_download" / "training" / "processed"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_server_wins.onnx"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "models" / "player_mapping.json"


def _discover_latest_csv(processed_dir: Path) -> Path:
    """Return full_game_states.csv from the most recent preprocessed_* folder."""
    run_dirs = sorted(processed_dir.glob("preprocessed_*"), reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No preprocessed_* folders in {processed_dir}")
    return run_dirs[0] / "full_game_states.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay tennis match events through the ingestion pipeline."
    )
    parser.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Replay a single match by ID (default: replay the longest match in the CSV).",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=None,
        help="Path to full_game_states.csv (default: auto-discover latest preprocessed run).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to ONNX model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--player-mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help=f"Path to player_mapping.json (default: {DEFAULT_MAPPING_PATH}).",
    )
    parser.add_argument(
        "--queue-max-size",
        type=int,
        default=1000,
        help="Maximum event queue depth before backpressure (default: 1000).",
    )
    args = parser.parse_args()

    processed_csv = args.processed_csv or _discover_latest_csv(DEFAULT_PROCESSED_DIR)

    config = IngestionConfig(
        processed_csv=processed_csv,
        model_path=args.model_path,
        player_mapping_path=args.player_mapping,
        queue_max_size=args.queue_max_size,
        speed_factor=0.0,
    )

    pipeline = IngestionPipeline(config)

    # Resolve match ID
    if args.match_id:
        match_id = args.match_id
    else:
        # Default: longest match (most data, best demo)
        lengths = pipeline._df_states.groupby("match_id").size().sort_values(ascending=False)
        match_id = str(lengths.index[0])
        p1 = pipeline._df_states[pipeline._df_states["match_id"] == match_id]["player_1"].iloc[0]
        p2 = pipeline._df_states[pipeline._df_states["match_id"] == match_id]["player_2"].iloc[0]
        print(f"No --match-id specified. Using longest match: {p1} vs {p2} ({lengths.iloc[0]} points)")

    t0 = time.perf_counter()
    odds_df = pipeline.run_match(match_id)
    t_total = (time.perf_counter() - t0) * 1000.0

    throughput = len(odds_df) / (t_total / 1000.0)
    print(f"\nReplayed {len(odds_df):,} points in {t_total:.1f} ms  "
          f"({throughput:,.0f} points/sec)")

    pipeline.print_summary(match_id, odds_df)


if __name__ == "__main__":
    main()
