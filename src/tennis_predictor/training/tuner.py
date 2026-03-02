"""Optuna hyperparameter tuning for XGBoost."""

import numpy as np
import xgboost as xgb

from .config import TrainingConfig
from .data import SplitData


class HyperparameterTuner:
    """Bayesian optimisation over XGBoost hyperparameters using Optuna.

    Trains each trial on a subsample of the training data (controlled by
    ``tune_sample_frac``) for speed, while always evaluating on the full
    validation set.

    Example:
        >>> tuner = HyperparameterTuner(TrainingConfig(n_trials=10))
        >>> best_params = tuner.tune(data)
        >>> best_params
        {'max_depth': 5, 'learning_rate': 0.08, ...}
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def tune(self, data: SplitData) -> dict:
        """Run Optuna study and return the best hyperparameters."""
        import optuna

        cfg = self._config

        # Subsample training data by match for faster tuning
        rng = np.random.RandomState(cfg.seed + 1)
        tune_match_ids = rng.choice(
            list(data.train_matches),
            size=max(1, int(len(data.train_matches) * cfg.tune_sample_frac)),
            replace=False,
        )
        tune_mask = data.train_df["match_id"].isin(set(tune_match_ids))
        X_tune = data.X_train[tune_mask]
        y_tune = data.y_train[tune_mask]

        print(f"\nOptuna tuning ({cfg.n_trials} trials)")
        print(
            f"  Tuning subsample: {len(X_tune):,} points "
            f"({len(tune_match_ids):,} matches, {cfg.tune_sample_frac:.0%} of train)"
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            }

            model = xgb.XGBClassifier(
                n_estimators=cfg.n_estimators,
                objective="binary:logistic",
                eval_metric="logloss",
                early_stopping_rounds=cfg.early_stopping_rounds,
                tree_method="hist",
                enable_categorical=True,
                random_state=cfg.seed,
                n_jobs=-1,
                **params,
            )

            model.fit(
                X_tune, y_tune,
                eval_set=[(data.X_val, data.y_val)],
                verbose=False,
            )

            return model.best_score

        study = optuna.create_study(direction="minimize", study_name="xgb_tune")
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\n  Best trial #{best.number} — val logloss: {best.value:.6f}")
        for k, v in best.params.items():
            print(f"    {k:<20} {v}")

        return best.params
