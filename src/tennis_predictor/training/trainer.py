"""XGBoost model training."""

import xgboost as xgb

from .config import TrainingConfig
from .data import SplitData


class XGBoostTrainer:
    """Train an XGBoost binary classifier with given hyperparameters.

    Uses ``enable_categorical=True`` for native categorical feature support
    (player names) and ``tree_method="hist"`` for fast histogram-based training.

    Example:
        >>> trainer = XGBoostTrainer(TrainingConfig())
        >>> model = trainer.train(data, {"max_depth": 5, "learning_rate": 0.08})
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def train(self, data: SplitData, best_params: dict) -> xgb.XGBClassifier:
        """Train XGBoost with early stopping and return the fitted model."""
        cfg = self._config

        print(f"\nTraining XGBoost ({len(data.X_train):,} points)")
        print(f"  Params: {best_params}")

        model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=cfg.early_stopping_rounds,
            tree_method="hist",
            enable_categorical=True,
            random_state=cfg.seed,
            n_jobs=-1,
            **best_params,
        )

        model.fit(
            data.X_train, data.y_train,
            eval_set=[(data.X_train, data.y_train), (data.X_val, data.y_val)],
            verbose=50,
        )

        print(f"\n  Best iteration: {model.best_iteration}")
        print(f"  Best val logloss: {model.best_score:.6f}")

        return model
