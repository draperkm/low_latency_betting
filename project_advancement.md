# Project Advancement Log

---

## Phase 1 — Complete

### 2024 (retroactive)

**Preprocessing module**
- `SackmannLoader`: CSV discovery + match row iteration
- `MatchParser`: PBP string → point records (GameState)
- `ScoreValidator`: parsed vs expected score comparison (86.6% pass rate)
- `TrainingPipeline`: orchestrate + save `atp_full_game_states.csv` + `atp_training_dataset.csv`
- CLI: `python -m tennis_predictor.preprocessing`
- 18,249 matches → 2,820,681 point records

**Model training**
- Notebook `02_model_training.ipynb`: XGBoost binary classifier, 12 features, ONNX export
- ONNX verification: max diff < 1e-6
- Inference latency: ~0.022ms (4,500× under 100ms target)
- Accuracy: 63.2%, ROC AUC: 0.545, log loss: 0.655

**Training module**
- `TrainingConfig` dataclass, `DataSplitter`, `HyperparameterTuner` (Optuna, 20 trials)
- `XGBoostTrainer`, `ModelEvaluator`, `OnnxExporter`, `ModelTrainingPipeline`
- CLI: `python -m tennis_predictor.training`

**Live match simulation**
- Notebook `03_live_match_simulation.ipynb`: point-by-point ONNX replay
- Analytic match-win probability model (recursive: point → game → set → match)
- EMA-based serve probability estimation
- Break-of-serve + set boundary annotations

---

## Phase 2 — In Progress

### 2026-02-28

**Planning**
- Defined Phase 2 development plan in `PHASE_2_DEVELOPMENT.md`
- Notebook renumbering:
  - New `03_data_ingestion_and_queues.ipynb` (to build)
  - Current `03_live_match_simulation.ipynb` → becomes foundation for `04_streamlit_dashboard.ipynb`
- Identified two core engineering patterns from interview experience:
  1. Structured CSV ingestion → typed event model (producer pattern)
  2. OOP queue management (FIFO + priority queue, thread-safe)
- Planned `ingestion/` src module with 7 classes:
  `MatchEventProducer`, `EventQueue`, `PriorityEventQueue`, `GameStateManager`,
  `InferenceEngine`, `EventConsumer`, `OddsPublisher`
- Planned `dashboard/app.py` Streamlit app as final deliverable

**Notebook 03 written**: `notebooks/03_data_ingestion_and_queues.ipynb`
- 10 sections: event models → producer → queue → state manager → inference engine → consumer/publisher → e2e demo → latency analysis → visualisation → summary
- 7 classes: `MatchEvent`, `OddsUpdate`, `MatchEventProducer`, `EventQueue`, `GameStateManager`, `InferenceEngine`, `EventConsumer`, `OddsPublisher`
- `PriorityEventQueue` excluded for simplicity (noted in PHASE_2_DEVELOPMENT.md)

**Next action**: User validates notebook 03 before src integration (Gate 1)

### 2026-03-01

**Step 1b — ingestion/ src module complete**: `src/tennis_predictor/ingestion/`
- `models.py`   — `MatchEvent`, `OddsUpdate` dataclasses (Kafka message envelopes)
- `producer.py` — `MatchEventProducer` (CSV → event stream; Kafka consumer analogy)
- `queue.py`    — `EventQueue` (thread-safe FIFO, backpressure, Prometheus-style metrics)
- `state.py`    — `GameStateManager` (incremental delta updates; Redis hash analogy), `FEATURE_COLS`
- `engine.py`   — `InferenceEngine` (stateless ONNX wrapper with per-call latency)
- `consumer.py` — `EventConsumer`, `OddsPublisher` (processing loop + in-memory store)
- `pipeline.py` — `IngestionConfig` dataclass, `IngestionPipeline` (orchestrator)
- `__init__.py` — full public API export
- `__main__.py` — CLI: `python -m tennis_predictor.ingestion [--match-id <id>]`

**CLI smoke test** (Dimitrov vs Monfils, 904 points):
- Mean per-point latency: ~0.009 ms | p99: ~0.013 ms | max: ~0.26 ms
- Well within 200 ms production budget

**Gate 1b passed** — src integration validated.

**Removed**: `src/tennis_predictor/serving/` directory (obsolete)

### Step 2 — 2026-03-01

**Notebook `04_streamlit_dashboard.ipynb` written**
- 10 sections: setup → match selector → pipeline run → live chart prototype → score ticker → latency panel → session state design → streaming pattern → app.py sketch → summary
- Imports `IngestionPipeline` from src — no classes re-defined in the notebook
- Smoothing: WINDOW=20 rolling mean at render time (matches notebook 03 style); publisher stores raw values
- Designs full `st.session_state` schema ([INIT] / [MATCH] / [MUTABLE] lifecycle)
- Validates one-event-per-tick: ~0.01 ms/tick → 1,600× under Streamlit 16 ms budget
- app.py structure sketch: 6 sections, ~70 lines

**Gate 2 reached** — notebook 04 validated (score ticker + clean-match detection bugs fixed).

**Score ticker bug fix (notebook 04)**:
- Root cause: Dimitrov vs Monfils PBP split across 4 CSV rows → parser resets → all-zero scores at points 226, 452, 678.
- Fix 1 (match selector): clean-match detection using 6-field all-zero condition (including `points_p1/p2`) to identify reset rows.
- Fix 2 (score ticker): removed `POINT_LABELS`; `fmt_score`/`fmt_server` use `getattr()` for dual pandas Series / MatchEvent compatibility.
- Result: 92,274 clean matches; default demo match → Marin Cilic vs Sam Querrey (499 pts).

### Step 3 — 2026-03-01

**Gate 3 complete**: `src/tennis_predictor/dashboard/` built.
- `__init__.py` — package marker + launch instructions
- `app.py`       — runnable Streamlit live-odds dashboard (~160 lines, 7 sections)

**app.py architecture**:
1. Load heavy singletons once (`IngestionPipeline`, `InferenceEngine`, catalogue) — persisted in `st.session_state`
2. Sidebar controls: match selectbox, play/pause buttons, live score ticker placeholders
3. Match initialisation: pre-fill `EventQueue` with all events, reset `GameStateManager` + `OddsPublisher` on new match selection
4. Play/pause: set `running` flag in `st.session_state`
5. Streaming tick: one `queue.pop()` → `state.apply_event()` → `engine.predict()` → `publisher.publish()` per `st.rerun()`
6. Render: `st.line_chart` (rolling mean WINDOW=20), sidebar `st.metric` score ticker, 3-column latency panel
7. Schedule next tick: `st.rerun()` when `running=True`

**Key implementation notes**:
- `fmt_server()` uses `ev.server` (MatchEvent field), NOT `serving_player` (CSV column) — important distinction
- `_build_catalogue()` encapsulates the clean-match detection logic (same algorithm as notebook 04 section 2)
- Match complete detection: auto-sets `running=False` when queue drains

**Launch**:
```bash
uv run streamlit run src/tennis_predictor/dashboard/app.py
```

### Step 3b — 2026-03-01

**Recursive match-win probability integrated into ingestion pipeline.**

Motivation: raw ONNX `p_server_wins` zigzags ~0.60/~0.40 each game as the server
alternates, producing a flat mirrored chart with no visible trend. The analytic
model propagates point probability upward through game → set → match, showing a
real trend line where a dominant player pulls ahead.

**Changes — no new module, all within existing `ingestion/`:**

`state.py`:
- `_p_game(p, si, sj)` — recursive P(server wins game | score). Deuce closed-form.
- `_p_set_memo(...)` — lru_cache'd P(P1 wins set). Integer-rounded EMA float keys.
  Tiebreak at 6-6 approximated with deuce formula on average point probability.
- `_p_match(si, sj, pw, target=2)` — P(P1 wins match). Uses `>=` terminal to
  handle parser-corrupted set counts gracefully.
- `GameStateManager` gains `_ema_p1 / _ema_p2` (α=0.15, prior 0.60), updated in
  `apply_event()`, and new public method `match_win_probability(p_srv)`.

`models.py`: `OddsUpdate` gains `p1_match_win`, `p2_match_win` (default 0.5).

`consumer.py`: `run_match()` calls `state.match_win_probability()`; `to_dataframe()`
includes the two new columns.

`dashboard/app.py`: streaming tick computes match-win prob; chart plots it.

**Latency**: 0.009 ms → 0.073 ms per point (+0.064 ms for recursion).
Still 2,700× under the 200 ms budget.

**Also fixed in Step 3:**
- `streamlit` upgraded to 1.54.0; `pandas` relaxed to `>=2.0.0` for compatibility.
- Chart batching + score flashing: `time.sleep(tick_delay)` before `st.rerun()`;
  speed slider (0.05–2.0 s/point) added to sidebar.
- Score ticker: replaced opaque metric strings with a markdown table (Sets | Games | Pts).

### Step 3c — 2026-03-01

**Switched analytic probability from match-win to set-win.**

Motivation: match-win probability converges to 1.0 too quickly in best-of-3 matches
(especially with parser-corrupted set counts) giving a boring chart. Set-win probability
resets to ~0.5 at the start of each new set, showing within-set momentum clearly and
producing a more informative oscillating trend line.

**Model change**: removed `_p_match` layer (point→game→set→match) from `state.py`.
`set_win_probability(p_srv)` now returns `(p1_wins_cur_set, 1 - p1_wins_cur_set)` —
two recursive layers only: point → game → set.

**Renames across 4 files**:
- `state.py`: `match_win_probability` → `set_win_probability`
- `models.py`: `OddsUpdate.p1_match_win / p2_match_win` → `p1_set_win / p2_set_win`
- `consumer.py`: method call + `to_dataframe()` column names updated
- `dashboard/app.py`: method call + chart column selection updated
