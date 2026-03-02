"""Training pipeline: orchestrates loading, parsing, validation, and CSV export.

Wires SackmannLoader → MatchParser → DataFrame → CSV files.
Each run saves to a timestamped subfolder (e.g. preprocessed_2026_02_15_21_34).
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .loader import SackmannLoader
from .parser import MatchParser
from .validator import ScoreValidator


class TrainingPipeline:
    """End-to-end pipeline: raw Sackmann CSVs → training-ready CSV.

    Example:
        >>> pipeline = TrainingPipeline(
        ...     data_dir=Path("data/raw/tennis_pointbypoint"),
        ...     output_dir=Path("data/processed"),
        ... )
        >>> df = pipeline.run()
        # Prints stats, saves to data/processed/preprocessed_2026_02_15_18_15/
        #   full_game_states.csv    (2,820,681 x 21 — metadata + features + labels)
        #   training_dataset.csv    (2,820,681 x 14 — match_id + 12 features + server_wins)
        >>> df.shape
        (2820681, 21)
    """

    # Pre-point-state features only (no label leakage)
    TRAINING_FEATURES = [
        # Player identity (categorical — encoded at training time)
        "player_1", "player_2",
        # Game state
        "sets_p1", "sets_p2",
        "games_p1", "games_p2",
        "points_p1", "points_p2",
        "serving_player",
        "in_tiebreak",
        "is_deuce",
        "is_break_point",
    ]
    TARGET = "server_wins"
    MAX_ERROR_SAMPLES = 20

    def __init__(self, data_dir: Path, output_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._base_output_dir = Path(output_dir)
        self._loader = SackmannLoader(self._data_dir)
        self._parser = MatchParser()
        self._validator = ScoreValidator()

    @staticmethod
    def _make_run_dir(base: Path) -> Path:
        """Create a timestamped subfolder: preprocessed_YYYY_MM_DD_HH_MM."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        run_dir = base / f"preprocessed_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline: load → parse → validate → save.

        Steps:
            1. SackmannLoader.iter_matches() yields one dict per CSV row
            2. MatchParser.parse() expands each match's PBP string → ParseResult
            3. ScoreValidator.validate() checks parsed set_scores vs expected
            4. All points collected into a DataFrame
            5. Stats printed, CSVs saved to timestamped subfolder

        Returns:
            Full DataFrame with all columns (metadata + features + labels).
        """
        self._output_dir = self._make_run_dir(self._base_output_dir)

        all_points, stats = self._parse_all_matches()
        df = pd.DataFrame(all_points)

        self._print_stats(stats, df)
        self._save(df)

        return df

    def _parse_all_matches(self) -> tuple[List[Dict], Dict]:
        """Load and parse every match, tracking statistics."""
        all_points: List[Dict] = []
        stats = {
            "total_matches": 0,
            "parsed_ok": 0,
            "skipped_empty": 0,
            "validation_pass": 0,
            "validation_fail": 0,
            "validation_errors": [],
        }

        for match in self._loader.iter_matches():
            stats["total_matches"] += 1

            result = self._parser.parse(
                match_id=match["match_id"],
                player_1=match["player_1"],
                player_2=match["player_2"],
                pbp_sequence=match["pbp_sequence"],
                tournament=match["tournament"],
                date=match["date"],
            )

            if not result.points:
                stats["skipped_empty"] += 1
                continue

            stats["parsed_ok"] += 1
            all_points.extend(result.points)

            # Validate parsed scores against expected
            error = self._validator.validate(
                result.set_scores,
                match["expected_score"],
                match["match_winner"],
            )

            if error is None:
                stats["validation_pass"] += 1
            else:
                stats["validation_fail"] += 1
                if len(stats["validation_errors"]) < self.MAX_ERROR_SAMPLES:
                    stats["validation_errors"].append(
                        f"[{match['source_file']}] "
                        f"{match['player_1']} vs {match['player_2']}: {error}"
                    )

        return all_points, stats

    def _print_stats(self, stats: Dict, df: pd.DataFrame) -> None:
        """Print summary statistics to stdout."""
        total = len(df)
        srv_wins = df[self.TARGET].sum()

        print(f"{'=' * 60}")
        print(f"Training Pipeline Results")
        print(f"{'=' * 60}")
        print(f"  Matches processed:       {stats['parsed_ok']:>8}")
        print(f"  Skipped (empty):         {stats['skipped_empty']:>8}")
        print(f"  Total points:            {total:>8}")
        print()
        print(f"Score Validation:")
        print(f"  Pass: {stats['validation_pass']:>6}")
        print(f"  Fail: {stats['validation_fail']:>6}")
        if stats["parsed_ok"] > 0:
            pct = stats["validation_pass"] / stats["parsed_ok"] * 100
            print(f"  Rate: {pct:.1f}%")
        print()
        print(f"Target Distribution (server_wins):")
        print(f"  Server wins:   {srv_wins:>8} ({srv_wins / total * 100:.1f}%)")
        print(f"  Returner wins: {total - srv_wins:>8} ({(total - srv_wins) / total * 100:.1f}%)")

        if stats["validation_errors"]:
            print(f"\nSample validation errors (first {len(stats['validation_errors'])}):")
            for err in stats["validation_errors"]:
                print(f"  {err}")

    def _save(self, df: pd.DataFrame) -> None:
        """Save full dataset and training-only dataset to CSV."""

        # Full dataset (metadata + features + labels)
        full_path = self._output_dir / "full_game_states.csv"
        df.to_csv(full_path, index=False)
        print(f"\nSaved full dataset: {full_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Size:  {full_path.stat().st_size / 1024 ** 2:.1f} MB")

        # Training-only dataset (match_id + 12 features + target)
        training_cols = ["match_id"] + self.TRAINING_FEATURES + [self.TARGET]
        training_df = df[training_cols]
        training_path = self._output_dir / "training_dataset.csv"
        training_df.to_csv(training_path, index=False)
        print(f"\nSaved training dataset: {training_path}")
        print(f"  Shape: {training_df.shape}")
        print(f"  Size:  {training_path.stat().st_size / 1024 ** 2:.1f} MB")
        print(f"  Features: {len(self.TRAINING_FEATURES)}")
        print(f"  Target: '{self.TARGET}'")
