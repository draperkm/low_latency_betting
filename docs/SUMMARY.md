## Project Summary

**Goal**: Build a tennis point prediction system (<100ms latency) to demonstrate ML systems architecture for interviews.

**Why Tennis**: Minimal state (12 features), binary classification (server wins?), commercially relevant (high-volume betting market), demo-able end-to-end.

---

## Architecture

```
src/tennis_predictor/
├── preprocessing/           # Data pipeline (SRP: one class per responsibility)
│   ├── models.py            # GameState, ParseResult, tennis scoring
│   ├── loader.py            # SackmannLoader: CSV discovery + iteration
│   ├── parser.py            # MatchParser: PBP string → point records
│   ├── validator.py         # ScoreValidator: parsed vs expected scores
│   ├── pipeline.py          # TrainingPipeline: orchestrate + save CSVs
│   └── __main__.py          # CLI entry: python -m tennis_predictor.preprocessing
├── training/                # Training module (mirrors preprocessing/ structure)
│   ├── config.py            # TrainingConfig dataclass
│   ├── data.py              # DataSplitter + SplitData
│   ├── tuner.py             # HyperparameterTuner (Optuna)
│   ├── trainer.py           # XGBoostTrainer
│   ├── evaluator.py         # ModelEvaluator (metrics, calibration, importance)
│   ├── exporter.py          # OnnxExporter (label-encode, convert, verify)
│   ├── pipeline.py          # ModelTrainingPipeline orchestrator
│   └── __main__.py          # CLI: python -m tennis_predictor.training
notebooks/
├── 01_raw_data_to_training_set.ipynb  # Ground truth parser
├── 02_model_training.ipynb            # XGBoost training + ONNX export
├── 03_live_match_simulation.ipynb     # Match replay + win probability
models/
├── xgb_server_wins.onnx    # Production model (~395 KB)
└── player_mapping.json     # Player name → integer mapping for ONNX
```

## Current Status

### Completed

**Preprocessing module** (`preprocessing/`)
- Parses Jeff Sackmann's compressed PBP strings (S/R/A/D/;/./) into game state records
- 18,249 matches → 2,820,681 point records
- Score validation: 86.6% pass rate
- Outputs timestamped CSVs: `atp_full_game_states.csv` (21 cols) + `atp_training_dataset.csv` (14 cols: match_id + 12 features + target)

**Model training** (notebook 02)
- XGBoost binary classifier: predict P(server wins next point)
- 12 features: 2 categorical (player_1, player_2) + 10 numeric (game state)
- Player names handled as native XGBoost categoricals (`enable_categorical=True`)
- Match-level train/val split (avoids within-match leakage)
- Early stopping at ~104 iterations
- ONNX export with label-encoded player names + `player_mapping.json`
- ONNX verification: max prediction difference < 1e-6
- Inference latency: ~0.022ms (4,500x under 100ms target)

**Training module** (`training/`)
- Production-ready OOP extraction of notebook 02 logic
- Optuna hyperparameter tuning (7 XGBoost params, 20 trials, Bayesian optimisation)
- Match-level subsampling knobs: `TRAIN_SAMPLE_FRAC`, `TUNE_SAMPLE_FRAC`
- CLI: `python -m tennis_predictor.training --n-trials 20 --train-sample-frac 1.0`

**Live match simulation** (notebook 03)
- Point-by-point replay of historical matches through ONNX model
- Rolling point-win probability per player (smoothed with rolling window)
- Analytic match-win probability model using recursive tennis scoring formulas:
  - Point → Game: closed-form P(hold serve) from P(server wins point)
  - Game → Set: recursive over game scores, with tiebreak at 6-6
  - Set → Match: recursive over set scores (best-of-3 or best-of-5)
- EMA-based serve probability estimation (adapts to in-match performance)
- Combined visualisation with break-of-serve and set boundary annotations

**Key metrics**:
- Accuracy: 63.2%
- Log loss: 0.6552
- ROC AUC: 0.5451
- Top features: sets_p2, player_1, player_2, is_break_point, serving_player

### Next Steps
1. Build serving layer: FastAPI + ONNXRuntime + player mapping
2. Retrain model with `TRAIN_SAMPLE_FRAC=1.0` for player-specific predictions
3. Add surface feature when available from live data feed
