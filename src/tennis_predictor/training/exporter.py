"""ONNX export: label-encode categoricals, retrain, convert, verify, measure latency."""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .config import TrainingConfig
from .data import SplitData


class OnnxExporter:
    """Export a trained XGBoost model to ONNX format with player mapping.

    ONNX does not support categorical features, so this class:
    1. Label-encodes player names to integers
    2. Retrains an equivalent model on the numeric features
    3. Converts to ONNX via ``onnxmltools``
    4. Saves ``player_mapping.json`` for the serving layer
    5. Verifies ONNX predictions match XGBoost
    6. Measures single-row inference latency

    Example:
        >>> exporter = OnnxExporter(Path("models"), TrainingConfig())
        >>> model_path, mapping_path = exporter.export(model, data, best_params)
    """

    def __init__(self, model_dir: Path, config: TrainingConfig) -> None:
        self._model_dir = Path(model_dir)
        self._config = config

    def export(
        self,
        model: xgb.XGBClassifier,
        data: SplitData,
        best_params: dict,
    ) -> tuple[Path, Path]:
        """Full export pipeline: label-encode → retrain → ONNX → verify → latency."""
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnxruntime as ort  # noqa: F401 — validate import early

        cfg = self._config
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Label-encode categoricals ---
        label_encoders = {}
        for col in cfg.categorical_features:
            le = LabelEncoder()
            le.fit(data.X_train[col].cat.categories)
            label_encoders[col] = le

        X_train_enc = data.X_train.copy()
        X_val_enc = data.X_val.copy()
        for col in cfg.categorical_features:
            X_train_enc[col] = label_encoders[col].transform(data.X_train[col]).astype(np.float32)
            X_val_enc[col] = label_encoders[col].transform(data.X_val[col]).astype(np.float32)

        X_train_enc = X_train_enc.astype(np.float32)
        X_val_enc = X_val_enc.astype(np.float32)

        # --- Step 2: Retrain with label-encoded features ---
        # .values strips column names — onnxmltools requires f0, f1, ... pattern
        model_onnx = xgb.XGBClassifier(
            n_estimators=model.best_iteration + 1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=cfg.seed,
            n_jobs=-1,
            **best_params,
        )
        model_onnx.fit(X_train_enc.values, data.y_train, verbose=False)

        # --- Step 3: Convert to ONNX ---
        initial_types = [("X", FloatTensorType([None, len(cfg.features)]))]
        onnx_model = onnxmltools.convert_xgboost(
            model_onnx, initial_types=initial_types, target_opset=15,
        )

        model_path = self._model_dir / "xgb_server_wins.onnx"
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # --- Step 4: Save player mapping ---
        player_mapping = {
            col: {name: int(code) for name, code in zip(le.classes_, range(len(le.classes_)))}
            for col, le in label_encoders.items()
        }
        mapping_path = self._model_dir / "player_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(player_mapping, f, indent=2)

        print(f"\nONNX Export:")
        print(f"  Model:   {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        print(f"  Mapping: {mapping_path}")

        # --- Step 5: Verify ---
        self._verify(model_path, X_val_enc, model_onnx)

        # --- Step 6: Measure latency ---
        self._measure_latency(model_path, X_val_enc)

        return model_path, mapping_path

    # ------------------------------------------------------------------
    # Verification & latency
    # ------------------------------------------------------------------

    @staticmethod
    def _verify(model_path: Path, X_val_enc, model_onnx) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        onnx_out = session.run(None, {"X": X_val_enc.values.astype(np.float32)})

        onnx_probas_raw = onnx_out[1]
        if isinstance(onnx_probas_raw, list) and isinstance(onnx_probas_raw[0], dict):
            onnx_proba = np.array([p[1] for p in onnx_probas_raw])
        elif isinstance(onnx_probas_raw, np.ndarray) and onnx_probas_raw.ndim == 2:
            onnx_proba = onnx_probas_raw[:, 1]
        else:
            raise ValueError(f"Unexpected ONNX output format: {type(onnx_probas_raw)}")

        ref_proba = model_onnx.predict_proba(X_val_enc)[:, 1]
        max_diff = np.max(np.abs(onnx_proba - ref_proba))
        status = "PASS" if max_diff < 1e-4 else "FAIL"
        print(f"  Verification: {status} (max diff: {max_diff:.8f})")

    @staticmethod
    def _measure_latency(model_path: Path, X_val_enc) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        sample = X_val_enc.values[:1].astype(np.float32)
        n_runs = 1000

        # Warm up
        for _ in range(100):
            session.run(None, {"X": sample})

        start = time.perf_counter()
        for _ in range(n_runs):
            session.run(None, {"X": sample})
        elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

        print(f"  Latency: {elapsed_ms:.3f} ms (single row, {n_runs} runs)")
        print(f"  Headroom: {100 / elapsed_ms:.0f}x under 100ms budget")
