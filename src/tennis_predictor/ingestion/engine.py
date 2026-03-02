"""InferenceEngine: stateless ONNX wrapper with per-call latency measurement.

Thread-safe: ONNX Runtime InferenceSession supports concurrent calls.
Load once at service startup and share the session across all matches.

Interview talking points:
- ONNX Runtime is the production standard for serving tree models at low
  latency. XGBoost's native predict() requires Python + numpy overhead;
  ONNX executes the graph directly in optimised C++ (~0.02 ms vs ~0.1 ms).
- Load the session once (expensive, ~10 ms) and reuse it indefinitely.
  In production this happens at service boot, not per request.
- time.perf_counter() has nanosecond resolution — essential for sub-ms
  measurements. time.time() is too coarse for this use case.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


class InferenceEngine:
    """Stateless ONNX inference wrapper with per-call latency measurement.

    Accepts a pre-built (1, n_features) float32 feature vector and returns
    the server-win probability plus the wall-clock inference time.

    Responsibility boundary: this class only accepts numeric feature vectors.
    Player name → integer encoding is GameStateManager's responsibility
    (done once at match start, not per point). Separation of concerns keeps
    the hot path (predict) minimal.

    Example:
        engine = InferenceEngine(Path("models/xgb_server_wins.onnx"))
        p_server_wins, latency_ms = engine.predict(feature_vector_np)
    """

    def __init__(self, model_path: Path) -> None:
        """Load the ONNX session once at instantiation.

        Args:
            model_path: Path to the exported .onnx model file.

        Note:
            InferenceSession initialisation is ~10 ms. Create one instance at
            startup and share it — do not instantiate per request.
        """
        self._session = ort.InferenceSession(str(model_path))
        # Query the input name rather than hardcoding it, so this engine works
        # with any single-input ONNX model without modification.
        self._input_name: str = self._session.get_inputs()[0].name

    def predict(self, feature_vector: np.ndarray) -> tuple[float, float]:
        """Run inference on a (1, n_features) float32 array.

        Args:
            feature_vector: Numpy array of shape (1, 12) and dtype float32,
                            produced by GameStateManager.to_feature_vector().

        Returns:
            p_server_wins: P(server wins next point) — raw ONNX output.
            latency_ms:    Wall-clock time for this single inference call.

        Note on outputs[1][0][1]:
            outputs[0] = predicted class label (int)
            outputs[1] = per-class probability map
            [0]  → first (and only) sample in the batch
            [1]  → class-1 probability = P(server wins)
            Works for both XGBoost and sklearn ONNX export formats.
        """
        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: feature_vector})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        p_server_wins = float(outputs[1][0][1])
        return p_server_wins, latency_ms
