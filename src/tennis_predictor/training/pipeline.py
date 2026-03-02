"""Training pipeline: orchestrates data loading, tuning, training, evaluation, and ONNX export.

Wires DataSplitter -> HyperparameterTuner -> XGBoostTrainer -> ModelEvaluator -> OnnxExporter.
"""

from pathlib import Path

from .config import TrainingConfig
from .data import DataSplitter
from .evaluator import ModelEvaluator
from .exporter import OnnxExporter
from .trainer import XGBoostTrainer
from .tuner import HyperparameterTuner


class ModelTrainingPipeline:
    """End-to-end pipeline: training CSV -> tuned XGBoost -> ONNX model.

    Example:
        >>> pipeline = ModelTrainingPipeline(
        ...     data_path=Path("data/processed/preprocessed_.../training_dataset.csv"),
        ...     model_dir=Path("models"),
        ... )
        >>> pipeline.run()
        # Prints stats, saves models/xgb_server_wins.onnx + player_mapping.json
    """

    def __init__(
        self,
        data_path: Path,
        model_dir: Path,
        config: TrainingConfig | None = None,
    ) -> None:
        self._config = config or TrainingConfig()
        self._splitter = DataSplitter(data_path, self._config)
        self._tuner = HyperparameterTuner(self._config)
        self._trainer = XGBoostTrainer(self._config)
        self._evaluator = ModelEvaluator(self._config)
        self._exporter = OnnxExporter(model_dir, self._config)

    def run(self) -> None:
        """Execute the full pipeline: load -> tune -> train -> evaluate -> export."""
        cfg = self._config

        print("=" * 60)
        print("Model Training Pipeline")
        print("=" * 60)
        print(f"  Features:      {len(cfg.features)} ({len(cfg.categorical_features)} cat + {len(cfg.numeric_features)} num)")
        print(f"  Train sample:  {cfg.train_sample_frac:.1%}")
        print(f"  Tune sample:   {cfg.tune_sample_frac:.1%}")
        print(f"  Optuna trials: {cfg.n_trials}")

        # 1. Load and split
        data = self._splitter.load_and_split()

        # 2. Hyperparameter tuning (optional)
        if cfg.n_trials > 0:
            best_params = self._tuner.tune(data)
        else:
            best_params = cfg.default_params
            print(f"\nSkipping tuning — using defaults: {best_params}")

        # 3. Train
        model = self._trainer.train(data, best_params)

        # 4. Evaluate
        self._evaluator.evaluate(model, data)

        # 5. Export ONNX
        self._exporter.export(model, data, best_params)

        print(f"\n{'=' * 60}")
        print("Pipeline complete.")
        print(f"{'=' * 60}")
