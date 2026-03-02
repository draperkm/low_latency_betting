# System Architecture — Tennis Point Prediction (DDD)

## Overview

This system predicts tennis point winners in <100ms using **Domain-Driven Design** principles. Each **bounded context** is self-contained with its own domain models and interfaces.

**Key Principle**: The domain layer has zero dependencies. Everything else is an adapter around it.

---

## 1. High-Level System Flow

```
  Live Match Feed
        │
        ▼
  ┌─────────────────┐
  │  INGESTION      │  Raw event → domain MatchState
  │  (bounded ctx)  │  Validates at boundary
  └────────┬────────┘
           │ MatchState
           ▼
  ┌─────────────────┐
  │  FEATURES       │  Offline: historical stats (Spark → Redis)
  │  (bounded ctx)  │  Online: delta compute + Redis lookup
  └────────┬────────┘
           │ FeatureVector
           ▼
  ┌─────────────────┐
  │  SERVING        │  ONNX inference + caching
  │  (bounded ctx)  │  FastAPI /predict endpoint
  └────────┬────────┘
           │ PredictionResponse
           ▼
  Betting Odds Engine
```

---

## 2. Bounded Contexts (DDD Structure)

```
  src/tennis_predictor/
  │
  ├── domain/                    ← THE HEART: pure domain logic, zero deps
  │   ├── match_state.py            MatchState, SetScore, GameScore (Pydantic)
  │   ├── transitions.py            State machine: point → game → set → match
  │   ├── value_objects.py          Surface, ServingPlayer, PointWinner (enums)
  │   └── events.py                 PointPlayed, GameCompleted events
  │
  ├── features/                  ← Bounded context: feature computation
  │   ├── domain/
  │   │   ├── feature_set.py        FeatureVector schema (Pydantic)
  │   │   └── registry.py           Feature definitions and metadata
  │   ├── offline/
  │   │   └── historical.py         Spark jobs for batch feature computation
  │   ├── online/
  │   │   ├── calculator.py         Real-time delta computation
  │   │   └── store.py              Redis feature store interface
  │   └── interface.py              Public API for this context
  │
  ├── training/                  ← Bounded context: model training
  │   ├── domain/
  │   │   ├── experiment.py         Experiment, ModelArtifact schemas
  │   │   └── metrics.py            Domain-specific eval (log-loss, calibration)
  │   ├── pipelines/
  │   │   └── train.py              XGBoost training orchestration
  │   ├── exporters/
  │   │   └── onnx_export.py        Model → ONNX conversion
  │   └── interface.py
  │
  ├── serving/                   ← Bounded context: live inference
  │   ├── domain/
  │   │   ├── prediction.py         PredictionRequest, PredictionResponse
  │   │   └── health.py             Liveness/readiness definitions
  │   ├── api/
  │   │   ├── app.py                FastAPI application
  │   │   └── routes.py             /predict, /health endpoints
  │   ├── inference/
  │   │   ├── engine.py             ONNX Runtime model loading + predict
  │   │   └── cache.py              Prediction caching (~10k state buckets)
  │   └── interface.py
  │
  └── shared/                    ← Cross-cutting concerns
      ├── ports.py                  Abstract interfaces (FeatureStore, ModelStore)
      ├── errors.py                 Domain exceptions
      └── config.py                 Hydra config loader
```

---

## 3. Dependency Direction (Clean Architecture)

```
  ┌─────────────────────────────────────────────────────────────┐
  │                         domain/                             │
  │  Pure logic — no framework deps (only Pydantic + stdlib)   │
  │  • match_state.py, transitions.py, value_objects.py        │
  └────────────────────────┬────────────────────────────────────┘
                           │ ALL OTHER LAYERS DEPEND ON DOMAIN
                           ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  features/  │  training/  │  serving/  │  (bounded contexts)  │
  │  Each context has its own domain/ subfolder                    │
  │  Communication via interface.py files                          │
  └────────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  Infrastructure adapters (Spark, Redis, FastAPI, ONNX)        │
  │  Framework-specific code stays at the edges                    │
  └────────────────────────────────────────────────────────────────┘

  KEY: Domain is the stable center. Infrastructure is pluggable.
```

---

## 4. Offline vs Online Split (Within Features Context)

```
  FEATURES BOUNDED CONTEXT
  ═══════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────┐  ┌─────────────────────────┐
  │  OFFLINE (Before Match)             │  │  ONLINE (During Match)  │
  │  features/offline/historical.py     │  │  features/online/       │
  └─────────────────────────────────────┘  └─────────────────────────┘

  ┌──────────────────────────┐            ┌─────────────────────────┐
  │ ATP/WTA Historical Data  │            │ New Point Completed     │
  │ ~50k matches             │            │ (MatchState)            │
  └───────────┬──────────────┘            └──────────┬──────────────┘
              │                                      │
              ▼                                      ▼
  ┌──────────────────────────┐            ┌─────────────────────────┐
  │ Spark Processing         │            │ calculator.py           │
  │ • Serve % per player     │            │ Compute Delta:          │
  │ • Surface win rates      │            │ • Match momentum        │
  │ • Fatigue curves         │            │ • Serve % this match    │
  │ • Head-to-head stats     │            └──────────┬──────────────┘
  └──────────┬───────────────┘                       │
             │                                       ▼
             ▼                            ┌─────────────────────────┐
  ┌──────────────────────────┐           │ store.py                │
  │ Redis Feature Store      │◄──────────│ Redis Lookup:           │
  │ Pre-computed player stats│           │ • Historical serve %    │
  │ Key: player_id_surface   │           │ • Surface win rate      │
  └──────────────────────────┘           └──────────┬──────────────┘
                                                    │
                                                    ▼
                                         ┌─────────────────────────┐
                                         │ interface.py            │
                                         │ get_features(           │
                                         │   match_state           │
                                         │ ) → FeatureVector       │
                                         └─────────────────────────┘

  KEY: Training and Serving don't care about offline/online split.
  They call features.interface.get_features() and receive a FeatureVector.
```

---

## 5. Latency Budget (<100ms)

```
  0ms         20ms                                         100ms
  ├───────────┼────────────────────────────────────────────────┤
  │   USED    │         HEADROOM (~80ms)                       │
  │   ~20ms   │                                                │
  ├───────────┼────────────────────────────────────────────────┤

  Breakdown:
  ┌──────────────────┬─────────────────────────────┬──────────┐
  │ Layer            │ Operation                   │ Time     │
  ├──────────────────┼─────────────────────────────┼──────────┤
  │ Ingestion        │ Pydantic validation         │  ~1ms    │
  │ Features (online)│ Redis lookup + delta        │  ~9ms    │
  │ Serving          │ ONNX inference              │  ~5ms    │
  │ Serving          │ Cache lookup + response     │  ~5ms    │
  ├──────────────────┼─────────────────────────────┼──────────┤
  │ TOTAL            │                             │ ~20ms    │
  └──────────────────┴─────────────────────────────┴──────────┘
```

---

## 6. Data Flow Example (Single Point Prediction)

```
  1. Live feed pushes raw point data
     ↓
  2. ingestion/parsers/normalise.py validates → MatchState
     {
       "match_id": "wimbledon_2025_final",
       "surface": "grass",
       "sets_player_1": 2, "sets_player_2": 1,
       "games_player_1": 5, "games_player_2": 4,
       "points_player_1": 2, "points_player_2": 2,  # 30-30
       "serving_player": 1,
       "points_won_last_10_player_1": 6
     }
     ↓
  3. features.interface.get_features(match_state) → FeatureVector
     {
       "serve_pct_player_1_grass": 0.72,  # from Redis (offline)
       "momentum_player_1": 0.6,           # computed (online delta)
       "is_break_point": 0,
       "fatigue_factor": 0.85,             # from Redis
       ...
     }
     ↓
  4. serving.interface.predict(feature_vector) → PredictionResponse
     {
       "prob_server_wins": 0.68,
       "prob_returner_wins": 0.32,
       "predicted_winner": "server",
       "confidence": 0.68,
       "latency_ms": 18.3
     }
     ↓
  5. Odds engine updates live betting markets
```

---

## 7. Why DDD for ML Systems?

```
  Traditional ML Structure          DDD Structure
  ═══════════════════════          ═══════════════
  data/                            domain/ (pure logic, zero deps)
  features/                          ↓
  models/                          features/ (owns offline/online split)
  serving/                           ↓
                                   training/ (owns evaluation)
  ❌ Change to "momentum"            ↓
     touches 3 folders             serving/ (owns latency path)

  ❌ Swapping Redis → DynamoDB     ✅ Change to "momentum" stays in features/
     requires refactoring
                                   ✅ Swapping Redis → DynamoDB touches
  ❌ Domain logic scattered           only features/online/store.py

  ❌ Hard to test (needs infra)    ✅ Domain tests run in milliseconds
                                      (no Redis, no Spark, no FastAPI)
```

---

## 8. Interview Talking Points

1. **Domain-first design**: "I structure ML systems with DDD because domain logic changes at a different pace than infrastructure. The core domain (match state, transitions) has zero framework dependencies."

2. **Bounded contexts**: "Each context (features, training, serving) is self-contained with its own domain models. They communicate through interface.py files — the same Pydantic validation pattern I use at Markerstudy."

3. **Offline/online split is a feature concern**: "Training and serving don't know whether features come from Spark or Redis. They call features.interface.get_features() and receive a FeatureVector. This keeps the latency-critical path isolated."

4. **Testability**: "The domain layer tests run in milliseconds because there's no infra. Integration tests with Redis/ONNX run separately."

5. **Swappable adapters**: "If I need to replace Redis with DynamoDB, I only touch features/online/store.py. The rest of the system doesn't care."
