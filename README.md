# 🎾 Low-Latency Tennis Betting Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-5C4EE5.svg)](https://docs.astral.sh/uv/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17+-blue.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5+-6236FF.svg)](https://optuna.org/)

A **production-style ML pipeline** that ingests ATP point-by-point match data, trains a binary
classifier exported to ONNX, and streams live set-win probabilities through a fully decoupled
event-queue architecture — displayed in a real-time Streamlit dashboard.

Built as a portfolio piece to demonstrate the engineering fundamentals behind low-latency
in-play betting systems: producer/consumer patterns, O(1) incremental state management,
stateless ONNX model serving, and recursive analytic probability modelling.

---

## 🎯 What This Project Demonstrates

This project started as a response to two engineering interview problems:

1. **CSV → prediction pipeline**: load a play-by-play sports dataset, structure it as a typed
   dictionary, feed it to an ML model, and serve predictions efficiently.
2. **OOP queue management**: design a class that manages a FIFO event buffer with backpressure,
   thread-safety, and Prometheus-style metrics.

These two problems are not coincidental — they are exactly the two critical joints in any
high-throughput in-play betting system. This project builds both from scratch, wires them
together into a complete pipeline, and adds a recursive analytic probability model and a
live Streamlit dashboard on top.

The goal was to demonstrate:
- **Technical depth** — a working end-to-end ML system, not a proof of concept
- **Critical thinking** — selecting the engineering decisions that create the most value in a
  betting infrastructure (latency, decoupling, incremental state, model serving)
- **Speed** — building the complete system from a problem statement without external scaffolding

---

## 🏗️ Architecture

### The One Core Insight

> **Separate the rate at which events happen from the rate at which you process them.**

In a live match, events arrive unpredictably and in bursts. ML inference and odds calculation
cannot keep up if called synchronously on every event. You need a buffer in the middle — one
that absorbs bursts and lets the downstream run at its own pace. That buffer is the queue.

This is the fundamental principle behind every high-throughput system, from Kafka to OS
interrupt handlers. The architecture below is a concrete application of it.

### System Diagram

```
 ATP CSV Data (Sackmann)
         │
         ▼
 ┌────────────────────┐
 │   Preprocessing    │   SackmannLoader → MatchParser → ScoreValidator
 │   18,249 matches   │   → 2,820,681 typed point records
 └─────────┬──────────┘
           │  atp_training_dataset.csv
           ▼
 ┌────────────────────┐
 │      Training      │   XGBoost → Optuna (20 trials) → ONNX export
 └─────────┬──────────┘
           │  model.onnx
           ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                      Ingestion Pipeline                      │
 │                                                              │
 │   MatchEventProducer                                         │
 │   (CSV → typed MatchEvent stream / Kafka consumer analogy)   │
 │           │                                                  │
 │           ▼                                                  │
 │   EventQueue  ←─────────────────── Kafka analogy            │
 │   (thread-safe FIFO, backpressure, Prometheus metrics)       │
 │           │                                                  │
 │           ▼                                                  │
 │   GameStateManager  ←────────────── Redis hash analogy      │
 │   (O(1) incremental delta updates)                           │
 │           │                                                  │
 │           ▼                                                  │
 │   InferenceEngine                                            │
 │   (stateless ONNX, ~0.009 ms/point)                         │
 │           │                                                  │
 │           ▼                                                  │
 │   set_win_probability()                                      │
 │   (Carter–Pollard recursion, point → game → set)            │
 │           │                                                  │
 │           ▼                                                  │
 │   OddsPublisher  ←───────────────── Redis pub/sub analogy   │
 │   (in-memory store, decoupled read path)                     │
 └──────────────────────────────────────────────────────────────┘
           │  OddsUpdate(p1_set_win, p2_set_win, latency_ms)
           ▼
 ┌────────────────────┐
 │     Dashboard      │   Streamlit — live odds chart + score ticker
 │     app.py         │   + latency panel + speed slider (0.05–2.0 s/pt)
 └────────────────────┘
```

### Three Decoupled Time Scales

The architecture is clean because each layer operates at a different time scale:

| Layer | Time scale | Mechanism |
|---|---|---|
| Events arrive (tennis points) | ~seconds | Live match / CSV replay |
| ML inference + recursion | ~milliseconds | ONNX Runtime + CPU arithmetic |
| User reads odds | ~sub-milliseconds | In-memory / Redis cache read |

Each layer is fully independent. Bottlenecks are isolated and scalable horizontally
without touching adjacent components — the same property that makes Kafka → Flink → Redis
the canonical production pattern.

---

## 🧱 Deep Dive: Every Class Explained

### 1. `MatchEventProducer` — The Parsing Boundary

> **Role**: receive raw, messy input and convert it into a clean, typed structure.
> Everything downstream trusts the data shape.

The producer is a Kafka consumer analogy. Its `produce()` method yields typed `MatchEvent`
dataclasses one at a time, optionally sleeping between events to simulate real-time pace.
The interface is source-agnostic: replacing CSV replay with a live Kafka consumer requires
changing only this class — the rest of the pipeline is unchanged.

```python
class MatchEventProducer:
    """
    speed_factor=0.0 → instant replay (backtesting, default)
    speed_factor=1.0 → real-time simulation (1 point every ~20 s)
    """

    def produce(self) -> Iterator[MatchEvent]:
        for i, row in self._df.iterrows():
            if self._speed > 0:
                target_s = t0 + (simulated_ts / 1000.0) / self._speed
                sleep_s = target_s - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

            yield MatchEvent(
                match_id=str(row["match_id"]),
                server=int(row["serving_player"]),
                sets_p1=int(row["sets_p1"]),
                # ... all other fields typed and validated
            )
```

**Why it matters**: without this boundary, every downstream component would need to handle
raw CSV dtypes, missing values, and semantic ambiguity (`serving_player` = 0 vs 1 vs 2).
Typing once at the boundary keeps the hot path clean.

---

### 2. `EventQueue` — The Decoupling Buffer (Kafka Analogy)

> **Role**: absorb event bursts and let the consumer process at its own pace.
> Explicit backpressure instead of silent blocking.

`EventQueue` wraps Python's `queue.Queue` (thread-safe) — not `collections.deque`
(which requires an explicit lock). `push()` returns `False` when the queue is full,
giving the producer an explicit signal to slow down or alert — equivalent to Kafka's
`max.block.ms` producer configuration.

```python
class EventQueue:
    def push(self, event: MatchEvent) -> bool:
        """Returns False (backpressure signal) if the queue is full."""
        try:
            self._q.put_nowait(event)       # non-blocking
            with self._lock:
                self._total_pushed += 1
                depth = self._q.qsize()
                if depth > self._peak_depth:
                    self._peak_depth = depth
            return True
        except queue.Full:
            return False                    # caller knows consumer is lagging

    def pop(self, timeout_ms: float = 10.0) -> Optional[MatchEvent]:
        """Returns None if the queue is empty after timeout_ms."""
        try:
            event = self._q.get(timeout=timeout_ms / 1000.0)
            with self._lock:
                self._total_popped += 1
            return event
        except queue.Empty:
            return None

    def metrics(self) -> dict:
        """Prometheus-style snapshot: depth, peak, drop count."""
        with self._lock:
            return {
                "total_pushed":  self._total_pushed,
                "total_popped":  self._total_popped,
                "current_depth": self._q.qsize(),
                "peak_depth":    self._peak_depth,
                "drop_count":    self._total_pushed - self._total_popped - self._q.qsize(),
            }
```

**Why it matters**: without the queue, the producer and consumer are tightly coupled —
if inference is slow, the producer blocks. With the queue, they operate independently.
This is the enabling abstraction for horizontal scaling: run one consumer thread per
live match, each reading from its own queue partition.

---

### 3. `GameStateManager` — O(1) Incremental State (Redis Hash Analogy)

> **Role**: maintain a live game state dict and apply point-level deltas.
> Never recompute from raw history.

This is the key performance insight of the whole system. There are two naive alternatives,
both slow:

- **Recompute from scratch**: re-read the entire match history on every point → O(N) per point.
- **Feed raw MatchEvent to the model**: the model expects a fixed-length float32 array, not
  strings, booleans, and player names.

`GameStateManager` solves both: it maintains an in-memory dict and writes only the fields
that changed per event (O(1)), then serialises it to the exact float32 array the ONNX model
expects.

```python
FEATURE_COLS = [
    "player_1", "player_2",          # integer-encoded at match start
    "sets_p1",  "sets_p2",
    "games_p1", "games_p2",
    "points_p1", "points_p2",
    "serving_player",
    "in_tiebreak", "is_deuce", "is_break_point",
]

class GameStateManager:
    def apply_event(self, event: MatchEvent) -> dict:
        """O(1) per point — only writes fields present in this event."""
        self._state = {
            "player_1":       self._p1_enc,    # encoded once at match start
            "player_2":       self._p2_enc,
            "sets_p1":        event.sets_p1,
            "points_p1":      event.points_p1,
            # ...
            "is_deuce":       int(event.is_deuce),
            "is_break_point": int(event.is_break_point),
        }
        # EMA serve-win rate: updated per point, used by the recursion.
        # α=0.15 (slow update), prior=0.60 (ATP average).
        if event.server == 1:
            self._ema_p1 = 0.15 * event.server_wins + 0.85 * self._ema_p1
        else:
            self._ema_p2 = 0.15 * event.server_wins + 0.85 * self._ema_p2
        return self._state

    def to_feature_vector(self) -> np.ndarray:
        """Serialise to (1, 12) float32 in FEATURE_COLS order — one HGETALL."""
        return np.array(
            [self._state[col] for col in FEATURE_COLS], dtype=np.float32
        ).reshape(1, -1)
```

**Production analogy**: in a deployed system this object lives in Redis. `apply_event()`
issues `HSET` per changed field. `to_feature_vector()` is a single `HGETALL`. A DB query
per inference call would cost 1–10 ms — a 10–100× latency hit compared to the 0.009 ms
we achieve here.

---

### 4. `InferenceEngine` — Stateless ONNX Serving

> **Role**: accept a feature vector, return a probability. Pure function, no state,
> no Python ML framework overhead at inference time.

XGBoost's native `predict()` carries Python + numpy overhead. ONNX Runtime executes the
same computation graph in optimised C++ at ~0.009 ms per call. The session is loaded once
at startup (≈10 ms) and reused indefinitely across all matches.

```python
class InferenceEngine:
    def __init__(self, model_path: Path) -> None:
        self._session = ort.InferenceSession(str(model_path))
        # Query input name dynamically — works with any single-input ONNX model.
        self._input_name: str = self._session.get_inputs()[0].name

    def predict(self, feature_vector: np.ndarray) -> tuple[float, float]:
        """Run inference on a (1, 12) float32 array.

        Returns:
            p_server_wins: P(server wins next point) — raw ONNX output.
            latency_ms:    Wall-clock time for this single inference call.
        """
        t0 = time.perf_counter()            # nanosecond resolution
        outputs = self._session.run(None, {self._input_name: feature_vector})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # outputs[1] = per-class probability map; [0][1] = class-1 prob (server wins)
        p_server_wins = float(outputs[1][0][1])
        return p_server_wins, latency_ms
```

**Why ONNX over pickle/joblib?** ONNX is framework-agnostic — the serving runtime (C++,
Java, Go) has no dependency on the training environment (Python, XGBoost). The model file
contains the full computation graph and weights; swapping the inference server language
requires no retraining or re-export.

| Scenario | Best choice |
|---|---|
| sklearn, Python serving | Joblib |
| XGBoost, any language | Native UBJ/JSON |
| PyTorch, Python serving | `state_dict` |
| PyTorch, C++ / mobile | TorchScript |
| **Any model, optimised serving** | **ONNX** ← this project |

---

### 5. `set_win_probability()` — Carter–Pollard Recursive Model

> **Role**: translate a point-level ONNX probability into the set-win probability
> that actually appears on a betting exchange.

The raw ONNX output (`p_server_wins ≈ 0.60`) zigzags every game as the server alternates.
Without transformation it is useless for a bettor — bookmakers price *set* and *match*
markets, not individual points. The Carter–Pollard recursion propagates the point-level
signal upward through two layers.

**Layer 1 — Point → Game**

```python
def _p_game(p: float, si: int, sj: int) -> float:
    """P(server wins game | server at si pts, returner at sj pts).

    si, sj ∈ {0,1,2,3} (0→0pts, 1→15, 2→30, 3→40).
    At deuce (3-3): closed-form geometric series — no infinite recursion.
    """
    if si >= 4:  return 1.0    # server won
    if sj >= 4:  return 0.0    # server lost
    if si == 3 and sj == 3:    # deuce
        q = 1.0 - p
        return (p * p) / (p * p + q * q)   # closed-form
    q = 1.0 - p
    return p * _p_game(p, si + 1, sj) + q * _p_game(p, si, sj + 1)
```

**Layer 2 — Game → Set**

```python
@lru_cache(maxsize=4096)
def _p_set_memo(gi: int, gj: int, p1_next: bool, pg1_r: int, pg2_r: int) -> float:
    """Memoised P(P1 wins set | gi games P1, gj games P2).

    Servers alternate each game — pg1_r used when P1 serves, pg2_r when P2 serves.
    Tiebreak at 6-6 approximated with the deuce formula on average point probability.
    Float keys rounded to integers for cache-key stability.
    """
    pg1 = pg1_r / 1000.0
    pg2 = pg2_r / 1000.0
    if gi >= 6 and gi - gj >= 2:  return 1.0
    if gj >= 6 and gj - gi >= 2:  return 0.0
    if gi == 7:                    return 1.0
    if gj == 7:                    return 0.0
    p1_wins_game = pg1 if p1_next else (1.0 - pg2)
    return (
        p1_wins_game * _p_set_memo(gi+1, gj, not p1_next, pg1_r, pg2_r)
        + (1.0 - p1_wins_game) * _p_set_memo(gi, gj+1, not p1_next, pg1_r, pg2_r)
    )
```

**Composing the two layers**

```python
def set_win_probability(self, p_srv: float) -> tuple[float, float]:
    # Layer 1: who wins the current game?
    p_cur_game = _p_game(p_srv, si_pts, sj_pts)
    p1_wins_cur_game = p_cur_game if server == 1 else (1.0 - p_cur_game)

    # EMA serve rates drive all future-game probabilities.
    pg1 = _p_game(self._ema_p1, 0, 0)   # P(P1 wins game when P1 serves)
    pg2 = _p_game(self._ema_p2, 0, 0)   # P(P2 wins game when P2 serves)

    # Layer 2: who wins the set?
    p1_wins_cur_set = (
        p1_wins_cur_game * _p_set(gi+1, gj, p1_serves_next, pg1, pg2)
        + (1.0 - p1_wins_cur_game) * _p_set(gi, gj+1, p1_serves_next, pg1, pg2)
    )
    return float(p1_wins_cur_set), float(1.0 - p1_wins_cur_set)
```

**Why this produces a better chart**: the set-win probability resets to ~0.5 at the start
of each new set and oscillates within it, clearly showing momentum shifts. The raw ONNX
probability is flat and uninformative.

---

### 6. `EventConsumer` — The Engine That Closes the Loop

> **Role**: sit in a loop, pull from the queue, drive every component in order.
> This is the only class that knows how all the pieces connect.

Without `EventConsumer`, you have a beautifully designed assembly line with nobody
switching it on. The consumer is the orchestrator — it handles only control flow,
delegating all domain logic to the components it holds.

```python
class EventConsumer:
    def __init__(self, eq, state_mgr, engine, publisher):
        # Dependency injection — each component is decoupled.
        self._queue   = eq
        self._state   = state_mgr
        self._engine  = engine
        self._pub     = publisher

    def run_match(self, events: list[MatchEvent]) -> pd.DataFrame:
        for event in events:
            t_start = time.perf_counter()

            if not self._queue.push(event):
                continue                   # backpressure: drop this event

            consumed = self._queue.pop(timeout_ms=50.0)
            if consumed is None:
                continue                   # timeout (shouldn't happen in replay)

            # Step 1: update state (O(1))
            self._state.apply_event(consumed)
            fv = self._state.to_feature_vector()

            # Step 2: run ONNX inference
            p_server_wins, _ = self._engine.predict(fv)
            p1_point = p_server_wins if consumed.server == 1 else 1.0 - p_server_wins

            # Step 3: recursive set-win probability
            p1_sw, p2_sw = self._state.set_win_probability(p_server_wins)

            # Step 4: publish (write to cache)
            self._pub.publish(OddsUpdate(
                match_id=consumed.match_id,
                point_index=consumed.point_index,
                p1_win_prob=p1_point,
                p2_win_prob=1.0 - p1_point,
                p_server_wins=p_server_wins,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                p1_set_win=p1_sw,
                p2_set_win=p2_sw,
            ))

        return self._pub.to_dataframe()
```

**Production note**: `run_match()` is synchronous here for straightforward replay. In
production it becomes an `asyncio` task or a dedicated thread per match. The interface
does not change — this is what makes the design correct.

---

### 7. `OddsPublisher` — Decoupling the Output Side

> **Role**: write inference results to a store that external consumers can read
> independently, at their own pace, without knowing anything about the pipeline.

This is the same decoupling principle as the queue — but on the output side. Just as the
`EventQueue` decouples the producer from the consumer, `OddsPublisher` decouples the
inference result from whoever reads it (dashboard, API, WebSocket feed).

```python
class OddsPublisher:
    """In-memory store → Redis PUBLISH + LPUSH in production."""

    def publish(self, update: OddsUpdate) -> None:
        self._history.append(update)    # Redis PUBLISH in production

    def to_dataframe(self) -> pd.DataFrame:
        """Columns: point_index, p1_win_prob, p2_win_prob,
                    p_server_wins, p1_set_win, p2_set_win, latency_ms"""
        return pd.DataFrame([
            {"point_index": u.point_index,
             "p1_set_win":  u.p1_set_win,
             "p2_set_win":  u.p2_set_win,
             "latency_ms":  u.latency_ms, ...}
            for u in self._history
        ])
```

**The symmetry**:

```
Raw input → [parsing boundary] → Queue → [EventConsumer] → Publisher → External readers
             (MatchEventProducer)                          (OddsPublisher)
```

The queue decouples the input side. The publisher decouples the output side. The consumer
is the engine in the middle. Every production streaming system — Kafka, Spark Streaming,
Flink — has exactly this shape, just with more infrastructure around each piece.

---

## 🚀 Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/low-latency-betting.git
cd low-latency-betting
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Add the Data

Download Jeff Sackmann's ATP point-by-point data and place the `*_pbp.csv` files
under `data/sackmann/`.
→ [tennis_atp on GitHub](https://github.com/JeffSackmann/tennis_atp)

---

## 💻 Running

### Preprocessing
```bash
uv run python -m tennis_predictor.preprocessing
# → data/atp_full_game_states.csv   (2.8M rows)
# → data/atp_training_dataset.csv
```

### Training
```bash
uv run python -m tennis_predictor.training
# → models/model.onnx
```

### Ingestion CLI (smoke test)
```bash
uv run python -m tennis_predictor.ingestion
# Expected output:
# [IngestionPipeline] match_id=20040105-M-...  points=904
#   mean latency: 0.009 ms | p99: 0.013 ms | max: 0.261 ms
```

### Streamlit Dashboard
```bash
uv run streamlit run src/tennis_predictor/dashboard/app.py
```

Open `http://localhost:8501`, select a match, press **Play**, and watch the set-win
probability chart update point by point with the score ticker and latency panel.

---

## 🔧 Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Package manager | uv | Fast, lock-file-based dependency management |
| Classifier | XGBoost 2.0+ | Point-win binary classification |
| Hyperparameter search | Optuna | 20-trial Bayesian optimisation |
| Model serving | ONNX Runtime | Stateless sub-millisecond inference |
| Data wrangling | pandas / numpy | Feature engineering and serialisation |
| Dashboard | Streamlit | Live streaming chart + score ticker |
| Probability model | Carter–Pollard recursion | Point → game → set win probability |

---

## 📄 Data

Point-by-point match data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp). The preprocessing pipeline discovers all `*_pbp.csv` files automatically.

> The Sackmann dataset is not included in this repository. Download it separately and
> place the files under `data/sackmann/` before running the preprocessing pipeline.

---

## 👤 Author

**Jean Charles**

- LinkedIn: [jean-charles-k](https://www.linkedin.com/in/jean-charles-k/)
- GitHub: [@draperkm](https://github.com/draperkm)

---

⭐ **Star this repo** if you found it useful!
