# NFL play prediction system

-> What do we want to know about the game(FEATURES)  and what do we want to predict(DEPENDANT VARIABLE)?

# Doing tennis instead
-> Tennis is the ideal choice. Here's why it maps so well:
Minimal game state — the entire context fits in a small object: set score, game score, point score (0/15/30/40/Ad), who's serving, surface type. Compare that to NFL where you need personnel packages, formations, field position, play history, etc.
Binary outcome per event — each point is won by either the server or the returner. That's a simple binary classification, whereas NFL next play has dozens of possible outcomes (run left, pass short right, screen, etc.).
Clear sequential structure — points → games → sets → match. Each level has well-defined transitions, so your state machine is trivial to implement. A MatchState Pydantic model would have maybe 8-10 fields total.
Still commercially relevant — tennis is one of the highest-volume in-play betting markets globally, and DraftKings actively offers live tennis betting. So it's not a toy example.
The prediction pipeline stays structurally identical:
Offline features: Historical serve percentages, surface-specific win rates, player fatigue curves (performance vs. rally length, set number), head-to-head stats. All computable with Spark over ATP/WTA match data.
Online features: Current match momentum (points won in last N points), break point conversion in this match, serving stats in this match vs. career average.
Model: XGBoost on tabular features for point-win probability given the current state. Sub-millisecond inference, and you can still tell the latency optimisation story.
Resilience: Same patterns — fallback to pre-match priors if live feed drops, schema validation on incoming point data, model drift detection as a tournament progresses.
The simplicity means you can actually build the whole thing end-to-end as a portfolio piece and demo it, rather than hand-waving over complexity. And you still get to showcase every architectural decision — the offline/online split, the modular pipeline, the CI/CD — just with a much cleaner domain.

# What is is an ML system in the end? (The external data must be parsed by an interface before getting to the system)

Live NFL feed (nflverse)
        │
        ▼
  ┌───────────┐     validates incoming data
  │ GameState │ ◄── Pydantic rejects bad data immediately
  └─────┬─────┘     (e.g. down=7 → error, yard_line=200 → error)
        │
        │ .to_feature_dict()
        ▼
  ┌────────────────┐
  │ Feature Engine  │ ◄── converts dict → numeric vector for ML
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │ ML Model       │ ◄── XGBoost / ONNX: "given this situation, what's next?"
  └───────┬────────┘
          │
          ▼
  ┌────────────────────┐
  │ PredictionResponse  │ ◄── "68% pass, 27% run, took 4.2ms"
  └────────────────────┘
          │
          ▼
      Betting app

## The aim of this app is to simulate an interface where we manually feed real data feeds and plot predictions on a self-updating chart that plots latency time on the screen

I hope I demonstrated a discrete technical competency, and know I'd like to demonstrate a sense of critical thinking, which is as important, if not even more, as the technical competency. I want here to demonstrate how quickly I can board on information and start building without the need of an external input... 

I also wanted to demonstrate a certain familiarity with managing an end-to-end ml-learning pipeline, and structuring it to be production ready (ci/cd, package management) and scalable (intelligent choices)

I also hoped of having demonstrated human competency.

This project is a continuation of the technical interview. We talked about 2 main problems, getting prediction from an play-by-play data set (with a particular data manipulation) and in general an OOP project highlighting the use of Kafka.

From those information, and from what we talked about in the two interviews with Aaron and Daria, I've developed a simulation of the problem I've understood is relevant to Draftikings with a particular attention to select the most relevant engineering challenges that create the most value in Draftikings infrastructure.

This is not an end-to-end project, in the sense that I have developed particualrly some aspects of the system that I think would be functional to start my introduction to the Draftkings engineering team.

(3 - OOP: Encapsulation groups related data and methods while hiding internal state.

2- Pre-computation and incremental updates.)

With this project I want to demonstrate:
- few important cs concepts (even if not production ready)
- how I can quickly build the foundation to deal with a problem and be able to not end in a dead end (very important for a software engineer)

I have modeled your system, and 2 core elements are fundamental:

1. From the first question, that fundamentally was loading a csv with data representing a play-by-play game and structuring it in a dictionary and running the predictions, probably the structuring the input would be model specific. The most interesting aspect in such a situation would be Pre-computation and incremental updates


2. Feature Computation (pre-match vs in-play)
This is where your PySpark experience becomes relevant — but with a twist. In betting, you have two very different regimes:

Pre-match (hours/days before): batch-friendly, latency is relaxed. You can afford heavier models — gradient boosting on historical match statistics, team ELO ratings, player injury data. This is close to what you already know.

In-play (during the game): latency budget is often 50–200ms end-to-end. You cannot re-run a full XGBoost model on every shot on target. So the architecture shifts to:

Pre-computed model outputs stored in a feature store (Redis is standard — sub-millisecond reads)
Incremental state updates — instead of recomputing everything, you maintain a game state object (possession %, shots, momentum score) and apply deltas as events arrive
Lightweight scoring models at serving time — often logistic regression or shallow neural nets with fixed input vectors
4. Serving Layer
The price the user sees is served from a cache. The write path (ML update → new price) and read path (user sees price) are decoupled. This is the key architectural insight:

[Kafka event] → [Feature updater] → [Model scorer] → [Redis cache] → [API / WebSocket to user]


The user-facing API reads from Redis (< 1ms). The ML pipeline writes to Redis asynchronously. The user never waits for the model to run.


---------------------
let's look again at this project and let's try make emerge some engineering aspects that will be relevant in my personal project deep dive interview. Especially after I had a technical interview yesterday, where they asked me to solve some hackerrank problems. The first one was about loading a csv with data representing a play-by-play game and structuring it in a dictionary and running the predictions and the second one was an OOP simple exercise for building a class that manages queues. This made me think that the real engineeing challenges in this scenario are managing a low-latency message queues (kafka is a canonical choice usually) and In-play (during the game): latency budget is often 50–200ms end-to-end. You cannot re-run a full XGBoost model on every shot on target. So the architecture shifts to:
Pre-computed model outputs stored in a feature store (Redis is standard — sub-millisecond reads)
Incremental state updates — instead of recomputing everything, you maintain a game state object (possession %, shots, momentum score) and apply deltas as events arrive
Lightweight scoring models at serving time — often logistic regression or shallow neural nets with fixed input vectors

Let's start a new development story, in which we will write a new  notebook following 02, about data ingestion and and queue managing. As ususal after that we will integrate the notebook in the src folder.

The ultimate goal is to get a final system which would be a dashboard in streamlit that simulates a play-by-play game and shows in a self-updating chart the odds progression of the two players as the game goes on, so the current 03 noteboook will become the new 04 and it will implement this stramlit app, that also will be later on integrated in the src project.

Write this instructions in a PHASE_2_DEVELOPMENT md document, as we will proceeed step by step, as I will have to validate the first notebook and the src integration, before moving on with the second one.

-----------------------

read phase_02_development.md file in low_latency_betting folder

----------------------

The One Core Insight

Separate the rate at which events happen from the rate at which you process them.

In a live match, events arrive unpredictably and in bursts — a goal, then a red card, then a penalty, all in 30 seconds. Your ML model and odds calculation can't keep up if they're called synchronously on every event. So you need a buffer in the middle — something that absorbs the burst and lets the downstream process at its own pace. That buffer is the queue.

-------------------------


Good instinct to zoom out first. Let me give you the one principle that makes everything else fall into place.

The One Core Insight
Separate the rate at which events happen from the rate at which you process them.

In a live match, events arrive unpredictably and in bursts — a goal, then a red card, then a penalty, all in 30 seconds. Your ML model and odds calculation can't keep up if they're called synchronously on every event. So you need a buffer in the middle — something that absorbs the burst and lets the downstream process at its own pace. That buffer is the queue.

This is the fundamental principle behind every high-throughput system, from Kafka to operating system interrupt handlers. The architecture you described is just a concrete application of it.

Why Each Layer Exists
Think of it as an assembly line in a factory. Each station has one job, and they're connected by conveyor belts (queues) so no station is blocked waiting for another.

MatchEventProducer — the loading dock. It receives raw, messy input (CSV, live feed, whatever) and converts it into a clean, typed structure. Everything downstream can trust the shape of the data. This is a classic parsing boundary.

EventQueue — the conveyor belt. It decouples the producer from the consumer. The producer doesn't know or care what happens next. The consumer processes at its own pace. If the consumer is slow or crashes, nothing is lost — events wait in the queue.

GameStateManager — the key efficiency insight. Instead of recomputing the full game state from scratch on every event, you maintain a running state and apply small deltas. A goal scored adds 1 to a counter — you don't re-read the entire match history. This is why the system is fast.

InferenceEngine — a frozen, pre-compiled model that just does arithmetic on the current state vector. No training, no heavy computation. Sub-millisecond.

OddsPublisher — writes to a cache that the user-facing API reads from. The user never waits for any of the above to run. They just read from the cache.

Why This Is the Right Shape
The architecture works because it cleanly maps to three different time scales:

Events arrive on the order of seconds
ML inference runs on the order of milliseconds
Users read odds on the order of sub-milliseconds (cache reads)
Each layer runs at its own time scale, independently. That independence is what makes the system robust and scalable. If any single layer becomes a bottleneck, you can scale it horizontally without touching anything else.


The two HackerRank problems were testing exactly the two joints in this chain — the parsing boundary (producer) and the queue abstraction — because those are where the real engineering decisions live in practice.

--------------------------

The One Core Insight Here
Someone has to close the loop.

You have a queue full of events, and you have components that know how to process them — but nothing actually drives the process. The EventConsumer is the thing that says "go". It's the engine that pulls from the queue, passes each event through the pipeline, and keeps things moving.

Without it, you have a beautifully designed assembly line with nobody switching it on.

Why Each Piece Exists
EventConsumer is the orchestrator. Its only job is to sit in a loop, pull the next event from the queue, and hand it to each downstream component in the right order — state update, then inference, then publish. It doesn't do any of those things itself; it just coordinates. This separation matters because the consumer is the thing you'd parallelise in production — one consumer thread per match running concurrently.

The fact that run_match() is synchronous here is an honest simplification. In production it becomes an asyncio task or a thread, but the interface doesn't change. That's good design — the logic is the same regardless of the concurrency model.

OddsPublisher exists for a subtler reason. The inference engine produces a number — a probability. But a number sitting in memory is useless unless something reads it. The publisher is the write side of the cache — it takes that number and puts it somewhere that an external system (a dashboard, a betting API, a WebSocket feed) can read from independently, at its own pace, without needing to know anything about the pipeline that produced it.

This is the same decoupling principle as the queue, but on the output side. Just as the queue decouples the producer from the consumer, the publisher decouples the inference result from whoever consumes it.

The Symmetry
Once you see it, the architecture is symmetric:

Raw input  →  [parsing boundary]  →  Queue  →  [EventConsumer]  →  Publisher  →  External readers

The queue decouples the input side. The publisher decouples the output side. The EventConsumer is the engine in the middle that keeps everything moving. Every production streaming system — Kafka, Spark Streaming, Flink — has exactly this shape, just with more infrastructure around each piece.

--------------------------

1-> data ingestion and training set build

2-> queue mechanism to account of live data (that's what you were missing in the first phase dev)

3-> MatchEventProducer is needed ONLY to simulate a live feed process from a csv file

4 -> EventQueue is needed to simulate an queue which is an IMPORTANT production abstraction to design systems that are decoupled and can work independently (THIS WAS A TECHNICAL INTERVIEW TOPIC)

5 -> GameStateManager: Why incremental state matters: Latency: updating a few keys is O(1) vs recomputing a full feature transform from raw history for every point. In the interview explain O(1) incremental updates vs O(N) recompute and why that matters for low-latency systems. creating a ready-to-use numeric array is cheap (O(1) incremental update) and avoids expensive recomputation from raw history on every point.

Why feeding the raw MatchEvent directly usually fails

Different formats: MatchEvent contains strings, ints, booleans and semantic fields (player names, point winner, etc.). Models accept a numeric array of fixed length/dtype, with a specific ordering and encoding.
Missing derived features: models are trained on engineered features (rolling averages, ratios, encoded IDs, one‑hots, interaction terms). Those are not present in a single raw event and must be computed from past events.
Recompute cost: building derived features from raw history (e.g., last 10 points win-rate, serve-success rolling stats) requires scanning k past events → O(k) per point. For long matches or many workers this becomes expensive.
In this case the update on the features is minimal and direct, but in a more production environment the class would implement some logic to compute more complex engineered features:

FEATURE_COLS = [
    "player_1", "player_2",
    "sets_p1", "sets_p2",
    "games_p1", "games_p2",
    "points_p1", "points_p2",
    "serving_player",
    "in_tiebreak", "is_deuce", "is_break_point",
    # engineered features:
    "last_10_p1_winrate", "p1_serve_winrate", "p2_serve_winrate",
    "ema_p1_recent", "game_diff_normalized",
]

# compute engineered features (cheap, O(1) or O(window) with small window)
        last_k_len = len(self._last_k) or 1
        last_10_p1_winrate = sum(1 for w in self._last_k if w == 1) / last_k_len
        p1_serve_winrate = (self._p1_server_wins / self._p1_serves) if self._p1_serves else 0.5
        p2_serve_winrate = (self._p2_server_wins / self._p2_serves) if self._p2_serves else 0.5
        game_diff_normalized = (event.games_p1 - event.games_p2) / max(1, event.games_p1 + event.games_p2)

        # attach engineered features to state
        self._state.update({
            "last_10_p1_winrate": last_10_p1_winrate,
            "p1_serve_winrate": p1_serve_winrate,
            "p2_serve_winrate": p2_serve_winrate,
            "ema_p1_recent": self._ema_p1,
            "game_diff_normalized": game_diff_normalized,
        })

6 -> InferenceEngine: basically is about model serialisation for model serving:

Great question — model serialization is one of those topics where the "right" choice depends heavily on your deployment context. Let me walk through each library and then explain ONNX as a unifying concept.

Saving ML Models: The Core Problem
When you train a model, you need to persist two things: the learned parameters (weights, thresholds, coefficients) and enough structural information to reconstruct the model at inference time. Different libraries solve this differently.

Scikit-learn
Scikit-learn models are plain Python objects, so the natural serialization tools are Python-native.

1. Pickle — the default, ships with Python.

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
2. Joblib — preferred for scikit-learn because it's more efficient for objects containing large NumPy arrays (uses memory-mapped files internally).

import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
3. ONNX (via sklearn-onnx) — for cross-platform deployment (more on this below).

⚠️ Both pickle and joblib are version-sensitive: a model pickled with scikit-learn 1.2 may fail to load with 1.4. Always pin your library versions — which aligns well with your poetry.lock discipline.

XGBoost
XGBoost has its own native serialization formats, separate from Python's pickle machinery.

1. Native binary format — XGBoost's own format, most reliable across versions.

model.save_model("model.ubj")   # UBJ (Universal Binary JSON) — recommended
model.load_model("model.ubj")
2. JSON — human-readable, good for inspection and debugging.

model.save_model("model.json")
model.load_model("model.json")
3. Pickle/Joblib — works but not recommended; the native formats are more stable and language-agnostic (a model saved from Python can be loaded in R or C++).

4. ONNX (via onnxmltools) — for cross-runtime deployment.

The native UBJ/JSON formats are the right default for XGBoost because they're language-agnostic by design — relevant if you ever need to serve from a Go or Java microservice.

PyTorch
PyTorch is richer here because you're dealing with neural networks that can have arbitrary computation graphs.

1. state_dict only — saves only the learned parameters (weights and biases), not the model architecture. This is the recommended approach in PyTorch.

# Save
torch.save(model.state_dict(), "model.pth")

# Load — you must reconstruct the architecture first
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()
Clean separation of code (architecture) and data (weights). Robust to refactoring.

2. Full model pickle — saves both architecture and weights using pickle under the hood. Brittle: breaks if you rename or move the class definition.

torch.save(model, "model.pth")
model = torch.load("model.pth")
3. TorchScript — compiles the model into an intermediate representation that is independent of the Python runtime. Two sub-modes:

torch.jit.trace — records operations on example inputs. Fast but can't handle dynamic control flow.
torch.jit.script — fully compiles Python control flow. More robust.
scripted = torch.jit.script(model)
scripted.save("model.pt")
model = torch.jit.load("model.pt")
This is the production-grade path when you need to serve from C++ (LibTorch) or mobile.

4. ONNX export — PyTorch has native torch.onnx.export().

5. Safetensors — a newer, safer alternative to pickle for weights, gaining traction in the HuggingFace ecosystem.

What is ONNX?
ONNX (Open Neural Network Exchange) is an open standard for representing ML models as a computational graph — think of it as the PDF of machine learning models: a format any compliant reader can consume, regardless of what tool created it.

The analogy: just as a PDF separates a document's content from the application used to create it (Word, InDesign), ONNX separates your model's computation from the framework used to train it.

Architecturally, an ONNX file contains:

A computation graph of operators (MatMul, ReLU, Conv, etc.) defined in a versioned opset
The weights embedded as tensors
Input/output type and shape metadata
Training Framework          ONNX Runtime / Hardware
(PyTorch, sklearn, XGB)  →  [.onnx file]  →  (TensorRT, OpenVINO, ONNX Runtime, CoreML)
Why it matters in production:

Runtime independence: you train in PyTorch but serve via ONNX Runtime, which is leaner and often faster
Hardware optimization: runtimes like TensorRT (NVIDIA) or OpenVINO (Intel) can take an ONNX graph and apply hardware-specific optimizations (operator fusion, quantization) automatically
Cross-language serving: the inference server can be written in C++, Java, or Go
How to export:

# PyTorch → ONNX (native)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)

# Scikit-learn → ONNX
from skl2onnx import convert_sklearn
onnx_model = convert_sklearn(sklearn_model, initial_types=[...])

# XGBoost → ONNX
from onnxmltools import convert_xgboost
onnx_model = convert_xgboost(xgb_model, initial_types=[...])
The trade-off: not every operator or model type maps cleanly to ONNX. Custom layers or unusual sklearn pipelines can require extra work to export correctly. ONNX is most valuable when you have a clear separation between a training environment (Python, GPU, ML frameworks) and a serving environment (optimized runtime, edge device, or multi-language system).

Quick Decision Guide
Scenario	Best choice
Sklearn, staying in Python	Joblib
XGBoost, any language	Native UBJ/JSON
PyTorch, Python serving	state_dict
PyTorch, C++ / mobile serving	TorchScript
Any model, optimized serving	ONNX
Debugging / human inspection	JSON (XGBoost), ONNX (visualize with Netron)

++++++++++++++++

The call in question is to the `InferenceEngine` defined a few cells above.  
Here’s the relevant bit from `InferenceEngine.predict()`:

```python
def predict(self, feature_vector: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    outputs = self._session.run(None, {self._input_name: feature_vector})
    latency_ms = (time.perf_counter() - t0) * 1000
    p_server_wins = float(outputs[1][0][1])
    return p_server_wins, latency_ms
```

### 📦 What you get back

- **First element** – `p_server_wins`: the model’s probability that the *server* wins the next point.  
- **Second element** – `latency_ms`: the wall‑clock time (in milliseconds) spent executing that single ONNX inference call.

### ⏱ How latency is measured

- A high‑resolution timer (`time.perf_counter()`) is read immediately **before** and **after** `session.run()`.  
- The difference, multiplied by 1 000, gives the elapsed time in milliseconds.  
- This measures *just* the time taken by the ONNX runtime to execute the graph on the provided 1×12 float32 vector; it excludes any surrounding Python overhead.

So when you execute `engine.predict(fv_np)` you’re getting both the model’s prediction and a per‑call latency measurement that’s used later for the benchmarking and budget analysis cells.

7 -> EventConsumer (this is a queue of event to decouple events from inference)

8 -> OddsPublisher (this is an in-memory odds store to improve latency)

9 -> ## 8. End-to-End Pipeline Demo

Here's the step-by-step breakdown of what happens in the e2e cell and inside run_match.

Step 1 — Wire up the components


eq        = EventQueue(max_size=5000)
state_mgr = GameStateManager(player_1_enc=p1_enc, player_2_enc=p2_enc)
publisher = OddsPublisher()
consumer  = EventConsumer(eq, state_mgr, engine, publisher)
Four objects are created and injected into EventConsumer. The consumer holds references to all of them but does nothing yet. This is dependency injection — each component is decoupled and only the consumer knows how they connect. engine was already created in the previous cell.

Step 2 — Start the clock, run the match


t0      = time.perf_counter()
odds_df = consumer.run_match(events_all)
t_total = (time.perf_counter() - t0) * 1000
events_all is the list of 904 MatchEvent objects produced earlier. run_match processes every one synchronously and returns a DataFrame. The outer timer measures total wall time for all 904 points.

Step 3 — Inside run_match: iterate over every point


for event in events:
    t_start = time.perf_counter()
One iteration = one tennis point. t_start marks the moment this specific event entered the pipeline — used at the end to compute per-event latency.

Step 4 — Push to the queue (producer side)


if not self._queue.push(event):
    continue  # backpressure: queue full, drop this event
The event is handed to the EventQueue. If the queue is at max_size, push() returns False (backpressure) and the event is silently dropped with continue — we skip to the next point. In this demo the queue is 5000 deep and there are only 904 points so no drops occur, but the logic is there for production correctness.

Step 5 — Pop from the queue (consumer side)


consumed = self._queue.pop(timeout_ms=50.0)
if consumed is None:
    continue  # timed out — queue was empty
In this synchronous single-threaded demo, push and pop happen back-to-back so pop always succeeds immediately. In a real threaded system the producer and consumer run in separate threads — pop blocks for up to 50ms waiting for the next event.

Step 6 — Update game state (incremental delta)


self._state.apply_event(consumed)
fv = self._state.to_feature_vector()
apply_event writes the new scores, server, and flags into GameStateManager's internal dict. No recompute from scratch — only the fields in this event are written (O(1)). to_feature_vector() then serialises that dict into a (1, 12) float32 numpy array in the exact column order the ONNX model expects.

Step 7 — Run inference


p_server_wins, _ = self._engine.predict(fv)
p1_point = p_server_wins if consumed.server == 1 else 1 - p_server_wins
The ONNX model outputs p_server_wins = P(server wins this point). This is server-relative. To get a player-relative probability we flip it: if player 1 is serving, their win probability is p_server_wins; if player 2 is serving, player 1's probability is 1 - p_server_wins.

Step 8 — Publish the odds update


self._pub.publish(OddsUpdate(
    match_id=consumed.match_id,
    point_index=consumed.point_index,
    p1_win_prob=p1_point,
    p2_win_prob=1 - p1_point,
    p_server_wins=p_server_wins,
    latency_ms=(time.perf_counter() - t_start) * 1000,
))
An OddsUpdate is created with the inference result and the end-to-end latency for this point (from t_start at step 3). publisher.publish() appends it to an in-memory list. In production this would be a Redis PUBLISH call to a channel the Streamlit dashboard is subscribed to.

Step 9 — Return the full odds history


return self._pub.to_dataframe()
After all 904 events, to_dataframe() converts the list of OddsUpdate objects into a pandas DataFrame with columns point_index, p1_win_prob, p2_win_prob, p_server_wins, latency_ms. This is what gets plotted in section 10.

The full data flow for one point:


events_all[i]
    → push() into EventQueue
    → pop() out of EventQueue
    → GameStateManager.apply_event()   [update state dict]
    → GameStateManager.to_feature_vector()  [dict → float32 array]
    → InferenceEngine.predict()        [ONNX → p_server_wins]
    → flip if player_2 is serving
    → OddsPublisher.publish(OddsUpdate)

11 -> Smoothing the winning probabilities: 

Good question. In notebook 03 section 10, the same WINDOW=20 rolling mean is applied to both win probability and latency — yes, the chart I was designing for notebook 04 is identical in style.

Here's why the smoothing is there:

Win probability: The raw ONNX model outputs P(server wins this point), which jumps significantly every time the server changes (e.g. it flips from ~0.60 to ~0.40 every game). Without smoothing you get a jagged zigzag that hides the real momentum trend. The rolling mean shows "form" — who has been dominating the last 20 points.

12 -> creating a more meaningful odds chart: THIS PHASE IS MATHEMATICALLY DELICATE AND COMPLEX, to add match-win probability instead of current-game prob, which is much easier, because it's the raw prediction from the ONNX model, THERE IS RECURSION LOGIC TO CALCULATE MATCH-WIN PROB. Let's actually use the SET-winner, to make the chart more intersting

regarding the odds, how difficult would it be to replace the recursive-match-win probability across the entire project? How does that takes into account of the onnx model predicting the server winning probability and the fact that the server keeps switching?

The ONNX model gives you the serve probability for whoever is currently serving — it doesn't give you the other player's serve probability in the same call. Notebook 03 solved this with an EMA (exponential moving average): for each player, maintain a running estimate of their serve win rate from points already played. After a few games it converges to a stable estimate.

How server switching is handled
The recursive formula naturally absorbs it:


P(server wins game | current pts score)
    → uses p_srv for the current game only

P(server wins set | current games score)
    → alternating games: uses p_srv_p1 for P1-serve games,
                                p_srv_p2 for P2-serve games

P(P1 wins match | current sets score)
    → uses the set-win probabilities for both players
       as server and returner

Because the formula explicitly accounts for which player serves in each future game of the set, the zigzag artefact disappears — the match-win probability is smooth and reflects real momentum.

Concrete changes to each file
ingestion/state.py — the bulk of the work:

Add _p1_serve_ema, _p2_serve_ema fields (updated in apply_event)
Add private module-level functions _p_win_game(), _p_win_set(), _p_win_match() (ported from notebook 03)
Add public method match_win_probability(p_srv: float) -> tuple[float, float]
ingestion/models.py — minimal:

Add p1_match_win: float and p2_match_win: float to OddsUpdate
ingestion/consumer.py — one line:


# after engine.predict():
p1_match, p2_match = self._state.match_win_probability(p_server_wins)
# add to OddsUpdate
dashboard/app.py — chart columns only:


# change:   odds_df[["p1_win_prob", "p2_win_prob"]]
# to:       odds_df[["p1_match_win", "p2_match_win"]]

Good observation — this reveals an important subtlety in the model. There are two layered reasons:

1. Who is serving right now?

The dashboard's sidebar shows the serving player. At 40-30, there are two very different situations:

Isner serving at 40-30 → game point on serve (great for Isner, ~87% to win the game)
Haas serving at 30-40 → Isner has a break point, but Haas is still ~41% to win the game
If Haas is the current server, the 40-30 score is actually a break-point situation — Isner has the advantage, but the server still has a real chance to hold.

2. The set model looks at ALL future games, not just this one

Even if Isner wins this game (say at 59%), the set probability also accounts for every future game via EMA serve rates. At 4-4, the remaining path could be:


Isner wins → 5-4, Haas serves → Haas holds → 5-5 → tiebreak territory
Haas wins  → 4-5, Isner serves → Isner holds → 4-6 or 5-5
If Haas's EMA (_ema_p2) has been higher in this match than Isner's (_ema_p1), the model knows Haas is more likely to hold future service games — which dominates the set calculation. A 59% game-win advantage for Isner on one game can be outweighed by Haas's advantage on multiple future games.

How to verify this in the code:

In state.py:237-238, you can temporarily add a print to inspect the live EMA values:


pg1 = _p_game(self._ema_p1, 0, 0)   # P(Isner wins game when serving)
pg2 = _p_game(self._ema_p2, 0, 0)   # P(Haas wins game when serving)
If pg2 > pg1 at that point in the match, Haas is the set favourite regardless of the 40-30 score — the model correctly reflects that he's been the stronger server in this match up to that point.

13 -> how are this information useful for a betting system??

Great question — it connects directly to why the project is called low_latency_betting. There are three layers.

Layer 1 — Convert probability to fair odds

A bookmaker quotes odds like 2.10 for Isner to win the set. That implies they believe:


P(Isner wins) = 1 / 2.10 = 0.476
Your model says p1_set_win = 0.55. You have an edge:


edge = model_probability - implied_probability
     = 0.55 - 0.476 = +0.074   (+7.4%)
A positive edge means the bookmaker is mispricing the bet. You bet Isner.

Layer 2 — Kelly Criterion for bet sizing

Once you have an edge, Kelly tells you what fraction of your bankroll to bet:


f* = edge / (odds - 1)
   = 0.074 / (2.10 - 1) = 6.7%
Bet 6.7% of your bankroll. Over many bets with a real edge, this maximises long-run growth. Without a probability model you can't compute this.

Layer 3 — Why latency is the moat

In-play (live) betting markets update every few seconds. The bookmaker's odds lag the real state of the match. Your system:


point played → state.apply_event() → engine.predict() → set_win_probability()
             ← 0.073 ms total →
A 0.073 ms pipeline means you can price the bet before the bookmaker updates their line. The window is typically 1–3 seconds after a point ends — you have 40,000× more time than you need. That time advantage is the entire business model.

Why the recursive model (not raw ONNX) matters for betting

The raw ONNX output (p_server_wins ≈ 0.60) zigzags every game and tells you nothing about who is winning the set. Bookmakers price set and match markets, not individual points.

The Carter-Pollard recursion translates a point-level signal into the probability that actually appears on the betting exchange — which is what you compare against the bookmaker's line to find the edge.

In one sentence for an interview:

The model gives you a fair price; the bookmaker gives you a market price; the difference is your edge; Kelly tells you how much to bet; and the sub-millisecond pipeline ensures you act before the market corrects.

