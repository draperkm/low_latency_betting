# Phase 2 Development Plan

## Motivation

After the technical interview, two patterns stood out as the real engineering challenges in live sports betting systems:

1. **Structured event ingestion** — Loading a CSV of play-by-play data, parsing it into a typed dictionary/object model, and feeding it downstream. This is exactly what HackerRank problem 1 was testing: can you design a clean data-loading pipeline?

2. **Queue management** — A class that manages a queue of game events, with controlled push/pop and priority handling. This is exactly what HackerRank problem 2 was testing: can you write a production-grade OOP queue abstraction?

The architecture that emerges from combining these two ideas is the canonical low-latency betting system:

```
CSV / live feed
      │
      ▼
MatchEventProducer          ← reads & structures events
      │
      ▼
EventQueue  (Kafka analogy) ← decouples producer from consumer
      │
      ▼
GameStateManager            ← applies incremental deltas (no full recompute)
      │
      ▼
InferenceEngine (ONNX)      ← <1ms point-win probability
      │
      ▼
OddsPublisher               ← stores odds history for dashboard
```

**Latency budget**: 50–200ms end-to-end. ONNX inference is ~0.022ms. The bottleneck is queue throughput and state update logic.

---

## Notebook & src Renumbering

| Old name | New name | Status |
|---|---|---|
| `01_raw_data_to_training_set.ipynb` | unchanged | done |
| `02_model_training.ipynb` | unchanged | done |
| `03_live_match_simulation.ipynb` | → `04_streamlit_dashboard.ipynb` | to be transformed |
| _(new)_ | `03_data_ingestion_and_queues.ipynb` | **Step 1 — build now** |

The current notebook 03 already contains the core live simulation logic (ONNX replay, win probability, analytic match-win model). It will be the foundation for the Streamlit dashboard after the queue layer is in place.

---

## Step 1 — Notebook `03_data_ingestion_and_queues.ipynb`

> **Gate**: You validate this notebook before we proceed to any src integration.

### Learning objectives (interview narrative)

- Demonstrate the **producer/consumer pattern** (Kafka in prod → `queue.Queue` in Python for simulation)
- Show **incremental state updates** instead of full model recompute on every event
- Measure and discuss the **latency budget** at each hop
- Show OOP design: one class per responsibility, dataclasses for messages

### Section 1 — Event modelling

Define typed dataclasses for the message envelope that flows through the system:

```python
@dataclass
class MatchEvent:
    match_id: str
    point_index: int
    server: int           # 1 or 2
    point_result: str     # 'S', 'R', 'A', 'D', etc.
    timestamp_ms: float   # simulated wall-clock time
    set_scores: tuple[int, int]
    game_scores: tuple[int, int]
    point_scores: tuple[int, int]

@dataclass
class OddsUpdate:
    match_id: str
    point_index: int
    p1_win_prob: float
    p2_win_prob: float
    latency_ms: float     # time from event arrival to odds publish
```

**Interview talking point**: Why separate `MatchEvent` from `OddsUpdate`? Because producers and consumers are decoupled — the producer does not know what the consumer does with the event (Single Responsibility, Open/Closed).

### Section 2 — MatchEventProducer

Class that reads a single match row (already parsed by `SackmannLoader` + `MatchParser` from Phase 1) and emits a stream of `MatchEvent` objects:

```python
class MatchEventProducer:
    def __init__(self, match_row: dict, speed_factor: float = 1.0)
    def produce(self) -> Iterator[MatchEvent]
```

- Replays the pre-parsed `GameState` records from Phase 1 output CSVs
- Attaches a simulated `timestamp_ms` based on configurable `speed_factor`
- **Interview talking point**: In production this class is replaced by a Kafka consumer reading from a topic. The rest of the system is identical.

### Section 3 — EventQueue (the OOP queue exercise)

Two queue classes that mirror what was asked in HackerRank problem 2:

```python
class EventQueue:
    """Thread-safe FIFO queue for MatchEvent objects.

    Wraps queue.Queue to add:
    - max_size enforcement with backpressure signalling
    - depth / throughput metrics
    - context manager support
    """
    def __init__(self, max_size: int = 1000)
    def push(self, event: MatchEvent) -> bool      # False if full (backpressure)
    def pop(self, timeout_ms: float = 10.0) -> MatchEvent | None
    def depth(self) -> int
    def is_empty(self) -> bool
    def metrics(self) -> dict                       # total_pushed, total_popped, peak_depth

class PriorityEventQueue(EventQueue):
    """Priority queue: lower point_index = higher priority (for out-of-order delivery).

    Uses heapq under the hood. Demonstrates why Kafka partitions preserve order
    but cross-partition delivery can arrive out of order.
    """
    def push(self, event: MatchEvent, priority: int | None = None) -> bool
```

**Interview talking point**: `queue.Queue` is thread-safe; `collections.deque` is not (without a lock). Kafka partitions are the production equivalent of `PriorityEventQueue` — events within a partition are ordered, but you need a reorder buffer across partitions.

### Section 4 — GameStateManager (incremental delta updates)

The core engineering insight: **do not recompute the full feature vector from scratch on every point**.

```python
class GameStateManager:
    """Maintains a live game state and applies point-level deltas.

    Analogy: Redis hash that gets HINCRBY on each event, not a full SET.
    """
    def __init__(self, player_1: str, player_2: str)
    def apply_event(self, event: MatchEvent) -> dict   # returns updated feature vector
    def current_state(self) -> dict
    def reset(self) -> None
```

Internal state dict maps directly to the 12 ONNX model features:
`serving_player, games_p1, games_p2, points_p1, points_p2, sets_p1, sets_p2, is_tiebreak, is_break_point, serve_streak_p1, serve_streak_p2, player_1_enc, player_2_enc`

**Interview talking point**: In production this state object lives in Redis (sub-millisecond reads). The model does not query a database — it reads a pre-built feature vector from a Redis key. This is the feature store pattern.

### Section 5 — InferenceEngine

Thin wrapper around ONNX Runtime that accepts a feature dict and returns `(p_server_wins, latency_ms)`:

```python
class InferenceEngine:
    def __init__(self, model_path: Path, player_mapping: dict)
    def predict(self, feature_vector: dict) -> tuple[float, float]   # prob, latency_ms
```

Already exists implicitly in notebook 03. Here we formalise it as a class and measure latency explicitly.

### Section 6 — EventConsumer + OddsPublisher

The consumer glues queue → state → inference → publish:

```python
class EventConsumer:
    def __init__(self, queue: EventQueue, state_manager: GameStateManager,
                 engine: InferenceEngine, publisher: OddsPublisher)
    def run_match(self) -> list[OddsUpdate]   # synchronous replay for notebook

class OddsPublisher:
    """In-memory store of OddsUpdate objects (Redis pub/sub in production)."""
    def publish(self, update: OddsUpdate) -> None
    def history(self) -> list[OddsUpdate]
    def to_dataframe(self) -> pd.DataFrame
```

### Section 7 — End-to-end demo + latency analysis

Wire everything together:
1. Pick a real match from the Phase 1 output CSV
2. Produce events → push to `EventQueue` → consume → publish odds
3. Plot odds progression (reuse Phase 1 notebook 03 chart style)
4. Print a latency breakdown table:

| Stage | Mean (ms) | P99 (ms) |
|---|---|---|
| Event production | ~0.01 | ~0.05 |
| Queue push + pop | ~0.02 | ~0.10 |
| State delta apply | ~0.05 | ~0.20 |
| ONNX inference | ~0.022 | ~0.05 |
| **Total** | **~0.10** | **~0.40** |

**Interview talking point**: The end-to-end budget is well under 1ms in this Python simulation. In production you'd add network hops: ~1ms to Kafka broker, ~0.5ms Redis read, ~0.5ms Kafka write for the output topic. Still well within 200ms.

---

## Step 1b — src integration: `ingestion/` module

> **Gate**: You validate the notebook AND the src integration before we proceed to Step 2.

After notebook validation, extract the classes into:

```
src/tennis_predictor/
└── ingestion/
    ├── __init__.py
    ├── models.py        # MatchEvent, OddsUpdate dataclasses
    ├── producer.py      # MatchEventProducer
    ├── queue.py         # EventQueue, PriorityEventQueue
    ├── state.py         # GameStateManager
    ├── engine.py        # InferenceEngine
    ├── consumer.py      # EventConsumer, OddsPublisher
    ├── pipeline.py      # IngestionPipeline: orchestrates a full match replay
    └── __main__.py      # CLI: python -m tennis_predictor.ingestion --match-id <id>
```

`IngestionPipeline` is the single entry point:

```python
class IngestionPipeline:
    def __init__(self, config: IngestionConfig)
    def run_match(self, match_id: str) -> pd.DataFrame   # odds history
    def run_all(self) -> dict[str, pd.DataFrame]
```

`IngestionConfig` dataclass:
```python
@dataclass
class IngestionConfig:
    processed_csv: Path       # Phase 1 output: atp_full_game_states.csv
    model_path: Path          # models/xgb_server_wins.onnx
    player_mapping_path: Path # models/player_mapping.json
    queue_max_size: int = 1000
    speed_factor: float = 1.0
```

---

## Step 2 — Notebook `04_streamlit_dashboard.ipynb`

> **Gate**: Steps 1 and 1b are fully validated before starting this.

This is the transformation of the current `03_live_match_simulation.ipynb` into a prototype for the Streamlit app. The notebook becomes a planning + design document; the actual runnable app lives in `src/tennis_predictor/dashboard/app.py`.

### Dashboard features

- **Match selector** — dropdown of available match IDs from the processed CSV
- **Simulation speed** — slider: 0.1× (slow motion) to 10× (fast forward)
- **Live chart** — `st.line_chart` or Plotly: two lines, P1 win probability and P2 win probability, updating point-by-point
- **Score ticker** — current set/game/point score in the sidebar
- **Annotations** — vertical dashed lines at set boundaries; markers at break-of-serve points
- **Latency panel** — rolling mean inference latency displayed as a metric

### Architecture

The dashboard calls `IngestionPipeline` (from Step 1b) in streaming mode. Each Streamlit re-render loop consumes one event from the queue and updates the chart.

```
st.session_state["queue"]          # EventQueue persisted across rerenders
st.session_state["state_manager"]  # GameStateManager
st.session_state["odds_history"]   # list[OddsUpdate] → plotted as DataFrame
```

### src location

```
src/tennis_predictor/
└── dashboard/
    ├── __init__.py
    └── app.py           # streamlit run src/tennis_predictor/dashboard/app.py
```

---

## Summary of gates

```
[NOW]     Write notebook 03_data_ingestion_and_queues.ipynb
    │
    ▼
[GATE 1]  You run and validate notebook 03
    │
    ▼
[GATE 1b] Extract ingestion/ module to src, validate CLI
    │
    ▼
[GATE 2]  Write notebook 04_streamlit_dashboard.ipynb (design + prototype)
    │
    ▼
[GATE 3]  Build src/tennis_predictor/dashboard/app.py (runnable Streamlit app)
```

---

## Key interview concepts this phase demonstrates

| Concept | Where it appears |
|---|---|
| Producer/consumer pattern (Kafka analogy) | `MatchEventProducer` → `EventQueue` → `EventConsumer` |
| Thread-safe queue OOP design | `EventQueue` wrapping `queue.Queue` |
| Priority queue (heapq) | `PriorityEventQueue` |
| Incremental state updates (Redis analogy) | `GameStateManager.apply_event()` |
| Feature store pattern | `GameStateManager` → fixed feature vector fed to ONNX |
| Sub-millisecond inference | `InferenceEngine` with latency measurement |
| Latency budget analysis | Section 7 latency breakdown table |
| Streamlit real-time UI | Dashboard notebook + `app.py` |

