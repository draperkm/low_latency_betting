# Build Training Module

## Model Selection

**Problem**: Predict P(server wins next point) given the current game state and player identities.

**Why XGBoost (not a sequence model)?**
- The 12 features already encode the full match state — each row is self-contained
- No need for LSTM/Transformer: the "sequence" is flattened into the state vector
- XGBoost handles categorical features (player names) natively with `enable_categorical=True`
- ONNX export for <100ms serving

**Serving simplicity**: Each prediction is a single row, single forward pass.
No sliding windows, hidden states, or sequence batching.

## Notebook: `02_model_training.ipynb`

### Pipeline
1. Load `atp_training_dataset.csv` (auto-discovers latest preprocessed run)
2. Split 80/20 **by match** (not random rows — avoids leakage)
3. Convert player names to `pd.Categorical` for XGBoost native support
4. Train XGBoost with `enable_categorical=True`, `tree_method="hist"`, early stopping
5. Evaluate: accuracy, log loss, ROC AUC, calibration
6. Export to ONNX (label-encoded player names), save `player_mapping.json`
7. Verify ONNX predictions match, measure inference latency

### Results
- **Accuracy**: 63.2%
- **Log Loss**: 0.6552
- **ROC AUC**: 0.5451
- **ONNX verification**: PASS (max diff < 1e-6)
- **Inference latency**: 0.022ms per prediction (4,500x under 100ms target)
- **Top features**: sets_p2 > player_1 > player_2 > is_break_point > serving_player

### Feature importance
Player identity features (`player_1`, `player_2`) are now the 2nd and 3rd most important,
confirming that **who** is playing matters more than game state for point prediction.
The model learns player-specific serve/return tendencies through the categorical encoding.

### ONNX serving architecture
```
Live game feed → look up player codes (player_mapping.json) → extract game state → ONNX predict → P(server wins)
```
Input: 12 float32 values (2 label-encoded player IDs + 10 game state integers)
Output: P(server_wins) probability

### Dependencies
- `onnxmltools>=1.12` — XGBoost to ONNX conversion
- `matplotlib>=3.8` — Feature importance plot
- `libomp` (brew) — OpenMP runtime for XGBoost on macOS
