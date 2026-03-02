"""Data loading, train/val split by match, and categorical encoding."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainingConfig


@dataclass
class SplitData:
    """Container for train/val arrays after splitting and encoding."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    train_df: pd.DataFrame
    train_matches: set
    n_total_matches: int


class DataSplitter:
    """Load training CSV, split by match, encode categoricals.

    The split is by ``match_id`` (not random rows) to prevent within-match
    leakage. Player names are converted to ``pd.Categorical`` with a shared
    category list so that train and val share the same integer codes.

    Example:
        >>> splitter = DataSplitter(Path("training_dataset.csv"), TrainingConfig())
        >>> data = splitter.load_and_split()
        >>> data.X_train.shape
        (10449222, 12)
    """

    def __init__(self, data_path: Path, config: TrainingConfig) -> None:
        self._data_path = Path(data_path)
        self._config = config

    def load_and_split(self) -> SplitData:
        """Load CSV → 80/20 match-level split → categorical encoding."""
        df = pd.read_csv(self._data_path)
        self._print_load_stats(df)

        match_ids = df["match_id"].unique().tolist()
        n_matches = len(match_ids)

        rng = np.random.RandomState(self._config.seed)
        rng.shuffle(match_ids)

        split_idx = int(n_matches * 0.8)
        train_matches = set(match_ids[:split_idx])
        val_matches = set(match_ids[split_idx:])

        # Optionally subsample training matches
        if self._config.train_sample_frac < 1.0:
            all_train = list(train_matches)
            rng.shuffle(all_train)
            train_matches = set(
                all_train[: int(len(all_train) * self._config.train_sample_frac)]
            )

        train_df = df[df["match_id"].isin(train_matches)].copy()
        val_df = df[df["match_id"].isin(val_matches)].copy()

        # Shared categorical encoding across train + val
        all_players = pd.concat([df["player_1"], df["player_2"]]).unique()
        player_cat_type = pd.CategoricalDtype(categories=sorted(all_players))

        for col in self._config.categorical_features:
            train_df[col] = train_df[col].astype(player_cat_type)
            val_df[col] = val_df[col].astype(player_cat_type)

        X_train = train_df[self._config.features]
        y_train = train_df[self._config.target].values
        X_val = val_df[self._config.features]
        y_val = val_df[self._config.target].values

        self._print_split_stats(
            X_train, y_train, X_val, y_val,
            train_matches, val_matches, n_matches, train_df,
        )

        return SplitData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_df=train_df,
            train_matches=train_matches,
            n_total_matches=n_matches,
        )

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_load_stats(self, df: pd.DataFrame) -> None:
        target = self._config.target
        all_players = pd.concat([df["player_1"], df["player_2"]]).unique()
        print(f"\nLoaded {self._data_path.name}")
        print(f"  Shape:          {df.shape}")
        print(f"  Nulls:          {df[self._config.features + [target]].isnull().sum().sum()}")
        print(f"  server_wins=1:  {df[target].sum():,} ({df[target].mean()*100:.1f}%)")
        print(f"  Unique players: {len(all_players):,}")

    def _print_split_stats(
        self, X_train, y_train, X_val, y_val,
        train_matches, val_matches, n_matches, train_df,
    ) -> None:
        cfg = self._config
        print(f"\nSplit (80/20 by match, TRAIN_SAMPLE_FRAC={cfg.train_sample_frac:.1%}):")
        print(f"  Train: {len(X_train):>10,} points  ({len(train_matches):,} matches)")
        print(f"  Val:   {len(X_val):>10,} points  ({len(val_matches):,} matches)")
        print(f"  Train server win rate: {y_train.mean()*100:.1f}%")
        print(f"  Val   server win rate: {y_val.mean()*100:.1f}%")
        for col in cfg.categorical_features:
            print(f"  {col}: {train_df[col].cat.categories.size:,} categories")
