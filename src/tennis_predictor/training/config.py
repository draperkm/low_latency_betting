"""Training configuration: features, constants, and hyperparameter knobs."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    """All tuneable knobs for the training pipeline in one place.

    Example:
        >>> cfg = TrainingConfig(train_sample_frac=0.5, n_trials=10)
        >>> cfg.numeric_features
        ['sets_p1', 'sets_p2', 'games_p1', ...]
    """

    # --- Features (must match TrainingPipeline.TRAINING_FEATURES in preprocessing) ---
    features: list[str] = field(default_factory=lambda: [
        "player_1", "player_2",
        "sets_p1", "sets_p2",
        "games_p1", "games_p2",
        "points_p1", "points_p2",
        "serving_player",
        "in_tiebreak",
        "is_deuce",
        "is_break_point",
    ])
    categorical_features: list[str] = field(default_factory=lambda: [
        "player_1", "player_2",
    ])
    target: str = "server_wins"

    # --- Sampling ---
    train_sample_frac: float = 0.001
    tune_sample_frac: float = 0.01

    # --- Tuning ---
    n_trials: int = 20

    # --- Training ---
    n_estimators: int = 500
    early_stopping_rounds: int = 20
    seed: int = 42

    # --- Defaults (used when tuning is skipped: n_trials=0) ---
    default_params: dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
    })

    @property
    def numeric_features(self) -> list[str]:
        return [f for f in self.features if f not in self.categorical_features]
