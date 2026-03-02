"""Model evaluation: metrics, calibration, and feature importance."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from .config import TrainingConfig
from .data import SplitData


class ModelEvaluator:
    """Evaluate a trained XGBoost model on validation data.

    Reports accuracy, log loss, ROC AUC, calibration bins, and feature
    importance — everything needed to assess model quality for a betting
    application where calibration matters more than raw accuracy.

    Example:
        >>> evaluator = ModelEvaluator(TrainingConfig())
        >>> metrics = evaluator.evaluate(model, data)
        >>> metrics["roc_auc"]
        0.545
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def evaluate(self, model: xgb.XGBClassifier, data: SplitData) -> dict:
        """Run full evaluation and print results. Returns metrics dict."""
        y_pred_proba = model.predict_proba(data.X_val)[:, 1]
        y_pred = model.predict(data.X_val)

        metrics = {
            "accuracy": accuracy_score(data.y_val, y_pred),
            "log_loss": log_loss(data.y_val, y_pred_proba),
            "roc_auc": roc_auc_score(data.y_val, y_pred_proba),
            "baseline": float(data.y_val.mean()),
        }

        self._print_metrics(metrics, data.y_val, y_pred)
        self._print_calibration(y_pred_proba, data.y_val)
        self._print_feature_importance(model)

        return metrics

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_metrics(self, metrics: dict, y_val, y_pred) -> None:
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Baseline:  {metrics['baseline']:.4f}")
        print(f"  Lift:      +{(metrics['accuracy'] - metrics['baseline'])*100:.1f}pp")
        print(
            f"\n{classification_report(y_val, y_pred, target_names=['Returner wins', 'Server wins'], zero_division=0)}"
        )

    def _print_calibration(self, y_pred_proba, y_val) -> None:
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_pred_proba, bins) - 1

        print(f"Calibration:")
        print(f"  {'Bin':>12}  {'Count':>8}  {'Predicted':>10}  {'Actual':>10}  {'Gap':>8}")
        print(f"  {'-' * 54}")
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_mean = y_pred_proba[mask].mean()
                actual_mean = y_val[mask].mean()
                gap = abs(pred_mean - actual_mean)
                print(
                    f"  {bins[i]:.1f} - {bins[i+1]:.1f}"
                    f"  {mask.sum():>8}"
                    f"  {pred_mean:>10.3f}"
                    f"  {actual_mean:>10.3f}"
                    f"  {gap:>8.3f}"
                )

    def _print_feature_importance(self, model: xgb.XGBClassifier) -> None:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        features = self._config.features

        print(f"\nFeature Importance (gain):")
        print(f"  {'Rank':>4}  {'Feature':<20}  {'Importance':>10}")
        print(f"  {'-' * 38}")
        for rank, idx in enumerate(sorted_idx, 1):
            bar = "\u2588" * int(importances[idx] / importances[sorted_idx[0]] * 20)
            print(f"  {rank:>4}  {features[idx]:<20}  {importances[idx]:>10.4f}  {bar}")
