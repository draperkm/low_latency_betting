# Production System Design: Tennis Live Betting Predictor

> **Interview follow-up**: "This demo is great, but how would you take it to production?"

---

## 1. Demo vs Production — Gap Analysis

| Aspect | Demo (what we built) | Production (what's needed) |
|--------|---------------------|---------------------------|
| Data source | CSV replay of historical matches | Live WebSocket feed from data provider |
| Feature computation | Batch (pandas DataFrame) | Real-time per-event (<10ms budget) |
| Model serving | Notebook / single-process | Horizontally scaled API behind load balancer |
| Player mapping | Static JSON file | Versioned registry, handles unknown players |
| Retraining | Manual CLI run | Scheduled pipeline with experiment tracking |
| Monitoring | Print statements | Metrics, drift detection, alerting |
| Latency target | ~0.022ms (ONNX only) | <100ms end-to-end (network + feature + inference) |

The demo proves the **ML core** works. Production wraps it with reliability, observability, and real-time data flow.

---

## 2. High-Level Architecture

```
+------------------------------------------------------------------+
|                        PRODUCTION SYSTEM                         |
+------------------------------------------------------------------+

  REAL-TIME PATH (per point, <100ms)
  ==================================

  +----------------+     +----------------+     +-----------------+
  |  Sports Data   |     |   Message      |     |  Feature        |
  |  Provider API  |---->|   Queue        |---->|  Service        |
  |  (Sportradar,  |     |   (Kafka)      |     |  (compute +     |
  |   Betfair)     |     |                |     |   cache in      |
  +----------------+     +--------+-------+     |   Redis)        |
        |                         |             +--------+--------+
        | WebSocket/SSE           |                      |
        | point-by-point          |  also consumed by    |  feature vector
        | events                  |  batch pipeline      |  (12 features)
        |                         |  for retraining      |
        v                         v                      v
  +----------------+     +----------------+     +-----------------+
  |  Event         |     |  Event Store   |     |  Prediction     |
  |  Normaliser    |     |  (Postgres /   |     |  Service        |
  |  (validate,    |     |   S3 parquet)  |     |  (FastAPI +     |
  |   dedup,       |     |                |     |   ONNX Runtime) |
  |   schema map)  |     +----------------+     +--------+--------+
  +----------------+                                     |
                                                         |  P(server wins)
                                                         |  P(match win)
                                                         v
                                                +-----------------+
                                                |  Client API     |
                                                |  (WebSocket     |
                                                |   streaming to  |
                                                |   consumers)    |
                                                +-----------------+


  OFFLINE PATH (daily/weekly)
  ===========================

  +----------------+     +----------------+     +-----------------+
  |  Event Store   |     |  Training      |     |  Model          |
  |  (historical   |---->|  Pipeline      |---->|  Registry       |
  |   points)      |     |  (Optuna +     |     |  (MLflow /      |
  |                |     |   XGBoost +    |     |   Weights &     |
  +----------------+     |   ONNX export) |     |   Biases)       |
                         +----------------+     +--------+--------+
                                                         |
                                                         |  champion model
                                                         |  promoted after
                                                         |  validation
                                                         v
                                                +-----------------+
                                                |  Prediction     |
                                                |  Service        |
                                                |  (hot-reload    |
                                                |   ONNX model)   |
                                                +-----------------+


  MONITORING PATH (continuous)
  ============================

  +----------------+     +----------------+     +-----------------+
  |  Prediction    |     |  Metrics       |     |  Alerting       |
  |  Service logs  |---->|  Collector     |---->|  (PagerDuty /   |
  |  (predictions, |     |  (Prometheus / |     |   Slack)        |
  |   latencies,   |     |   Datadog)     |     |                 |
  |   outcomes)    |     +--------+-------+     +-----------------+
  +----------------+              |
                                  v
                         +----------------+
                         |  Dashboards    |
                         |  (Grafana)     |
                         |  - latency P99 |
                         |  - accuracy    |
                         |  - drift       |
                         +----------------+
```

---

## 3. Component Deep Dives

### 3.1 Live Data Ingestion

```
  Provider WebSocket             Event Normaliser              Kafka
  =====================          ==================          =========

  { "match": "AO-2025-F",       { "match_id": "...",        topic: tennis.points
    "point_server": "Sinner",      "player_1": "Sinner",    key: match_id
    "score": "6-4 3-2 40-15",     "player_2": "Djokovic",   (ordered per match)
    "event": "ace" }               "sets_p1": 1,
         |                         "sets_p2": 0,
         |  raw event              "games_p1": 3,
         +---------------------->  "games_p2": 2,
                                   "points_p1": 40,
                                   "points_p2": 15,
                                   "serving_player": 1,
                                   "server_wins": null }
                                        |
                                        +----> Kafka
```

**Why Kafka?**
- Decouples ingestion from processing (if prediction service is slow, events queue up)
- Replay capability: re-process events when model changes
- Partitioned by `match_id`: all points for a match go to the same partition (ordering guarantee)
- Consumer groups: multiple prediction service instances can share load

**Provider choice**: Sportradar, Betfair Exchange API, or IMG Arena. Budget ~$500-5k/month depending on sport coverage. The normaliser maps provider-specific schemas to our 12-feature format.

---

### 3.2 Feature Service + Feature Store

```
                    FEATURE STORE
  =============================================
  |                                           |
  |  ONLINE (Redis)          OFFLINE (S3)     |
  |  ===============        ==============    |
  |                                           |
  |  player:sinner:          parquet files    |
  |    elo: 2100             partitioned by   |
  |    surface_wr: 0.82      date + tour     |
  |    serve_pct_30d: 0.67                    |
  |    h2h:djokovic: 3-5    used for model   |
  |                          retraining      |
  |  match:AO-2025-F:                        |
  |    current_state: {...}                   |
  |    point_history: [...]                   |
  |                                           |
  =============================================
```

**Online path** (per point, <5ms):
1. Kafka consumer receives normalised point event
2. Look up pre-computed player features from Redis (ELO, surface win rate, recent form)
3. Combine with live match state (score, serving player, tiebreak flags)
4. Assemble 12-feature vector, forward to prediction service

**Offline path** (daily batch):
1. Compute aggregate player statistics from Event Store
2. Update ELO ratings after completed matches
3. Write to both Redis (online serving) and S3 (training)

**Why a feature store?**
- **Train-serve skew prevention**: training and serving use the same feature computation code
- **Point-in-time correctness**: offline features are timestamped to prevent future leakage during retraining
- **Pre-computation**: player-level stats (ELO, surface win rate) are too expensive to compute per request

---

### 3.3 Prediction Service

```
                     PREDICTION SERVICE
  ========================================================
  |                                                      |
  |  Load Balancer (nginx / AWS ALB)                     |
  |       |         |         |                          |
  |       v         v         v                          |
  |  +--------+ +--------+ +--------+                   |
  |  | FastAPI| | FastAPI| | FastAPI|   N replicas       |
  |  | + ONNX | | + ONNX | | + ONNX |   (auto-scaled)  |
  |  +---+----+ +---+----+ +---+----+                   |
  |       |         |         |                          |
  |       +----+----+----+----+                          |
  |            |              |                          |
  |            v              v                          |
  |     Player Mapping    Model File                     |
  |     (in-memory dict)  (ONNX, ~400KB,                |
  |                        loaded once)                  |
  |                                                      |
  ========================================================

  Request flow (single prediction):
  -----------------------------------------------
  Feature vector (12 floats) ──> ONNX Runtime
                                     |
                                     v
                              P(server_wins) = 0.64
                                     |
                                     v
                              Analytic model:
                              P(p1 wins match) = 0.72
                                     |
                                     v
                              Response: {
                                "p_server_wins_point": 0.64,
                                "p1_match_win": 0.72,
                                "latency_ms": 0.03
                              }
```

**Design decisions**:

| Decision | Choice | Why |
|----------|--------|-----|
| Runtime | ONNX Runtime | 0.022ms inference, no Python GIL bottleneck for the model itself |
| Framework | FastAPI | async, auto-docs, Pydantic validation |
| Deployment | Docker + K8s | horizontal scaling, rolling updates |
| Model loading | Once at startup | 400KB ONNX file, ~10ms load time |
| Player mapping | In-memory dict | ~1,800 players, <1MB, O(1) lookup |
| Unknown players | Fallback to median encoding | Graceful degradation for new players |
| Batch endpoint | Yes (`/predict/batch`) | Score entire match state in one call |

**Latency budget** (end-to-end <100ms):

```
  Component              Target    Demo Measured
  =====================  ========  ============
  Network (client→LB)    < 20ms    n/a
  Load balancer           < 2ms    n/a
  Feature lookup (Redis)  < 5ms    n/a
  ONNX inference          < 1ms    0.022ms
  Match-win computation   < 2ms    ~0.1ms (cached)
  Serialisation + return  < 5ms    n/a
  Network (LB→client)    < 20ms    n/a
  ─────────────────────  ────────
  Total                  < 55ms    well under 100ms
```

---

### 3.4 Model Training Pipeline

```
  TRAINING PIPELINE (runs daily or on-demand)
  ============================================

  +----------+    +-----------+    +----------+    +----------+
  |  Event   |    |  Feature  |    |  Optuna  |    |  ONNX    |
  |  Store   |--->|  Join +   |--->|  Tuning  |--->|  Export  |
  |  (S3)    |    |  Split    |    |  (20     |    |  + Label |
  +----------+    |  (match-  |    |  trials) |    |  Encode  |
                  |   level)  |    +-----+----+    +-----+----+
                  +-----------+          |               |
                                         v               v
                                   +-----------+   +-----------+
                                   | Evaluate  |   |  Model    |
                                   | (AUC,     |   |  Registry |
                                   |  calib,   |   |  (MLflow) |
                                   |  log loss)|   +-----------+
                                   +-----+-----+
                                         |
                                         v
                                   +-----------+
                                   |  Champion |
                                   |  vs       |
                                   |  Candidate|
                                   |  (auto    |
                                   |   promote |
                                   |   if AUC  |
                                   |   improves|
                                   +-----------+
```

**What we already have** (demo → production mapping):
- `DataSplitter` → same match-level split logic, reads from S3 instead of local CSV
- `HyperparameterTuner` → same Optuna logic, results logged to MLflow
- `XGBoostTrainer` → same training loop, artifacts saved to model registry
- `OnnxExporter` → same export, plus versioned `player_mapping.json`
- `ModelEvaluator` → same metrics, plus comparison against current champion

**New for production**:
- **Scheduled orchestration**: Airflow DAG or similar triggers retraining on new data
- **Model registry**: MLflow tracks every experiment, promotes champion model
- **Automatic promotion**: if candidate AUC > champion AUC by threshold, auto-deploy
- **Player mapping versioning**: new players added weekly, mapping must be backward-compatible

---

### 3.5 Model Update (Zero-Downtime)

```
  Model Registry                    Prediction Service
  ==============                    ==================

  v1.0 (champion) ─────────────── currently loaded
  v1.1 (candidate) ── evaluate ──> passes gate?
                                        |
                                   yes  |  no
                                   ┌────+────┐
                                   v         v
                              promote     discard
                              to v1.1     keep v1.0
                                   |
                                   v
                              rolling restart:
                              pod 1: load v1.1 ✓
                              pod 2: load v1.1 ✓
                              pod 3: load v1.1 ✓
                              (zero downtime)
```

Since ONNX models are tiny (~400KB), the simplest approach is:
1. Store model artifacts in S3 with version prefix
2. Prediction service polls for new version every 60s (or receives webhook)
3. Load new model into memory, atomic swap of the inference session pointer
4. No restart needed — hot-reload in the same process

---

### 3.6 Monitoring & Observability

```
  MONITORING STACK
  ================

  +--------------+     +-------------+     +-----------+
  |  Prediction  |     |  Prometheus |     |  Grafana  |
  |  Service     |---->|  (scrape    |---->|  Dashboards|
  |              |     |   /metrics) |     |           |
  |  exports:    |     +------+------+     +-----------+
  |  - latency   |            |
  |  - pred dist |            v
  |  - error rate|     +-------------+
  |  - throughput|     |  Alerting   |
  +--------------+     |  Rules      |
                       +-------------+
                            |
                            v
  +---------------------------------------------------+
  |  ALERTS                                           |
  |                                                   |
  |  P99 latency > 50ms         --> page on-call      |
  |  Prediction mean drift > 5% --> trigger retrain   |
  |  Error rate > 1%             --> page on-call      |
  |  No events for 5min         --> check data feed   |
  |  Model accuracy < baseline  --> block promotion   |
  +---------------------------------------------------+
```

**Three pillars of ML monitoring**:

1. **Operational metrics** (standard SRE):
   - Request latency (P50, P95, P99)
   - Error rate, throughput, availability
   - CPU/memory per pod

2. **Data quality metrics** (input monitoring):
   - Feature distribution drift (KL divergence or PSI)
   - Missing/null feature rates
   - Unknown player frequency
   - Schema violations from data provider

3. **Model performance metrics** (ML-specific):
   - Prediction distribution shift (are outputs clustering around 0.5?)
   - Calibration drift (do 60% predictions win 60% of the time?)
   - Accuracy/AUC on resolved outcomes (delayed feedback — outcome known after point)
   - Feature importance stability across retrains

**Calibration monitoring** is critical for betting: a model that outputs 0.65 should correspond to the server actually winning ~65% of those points. Drift here means the model is mispricing.

---

## 4. Scaling Considerations

### 4.1 Load Estimation

```
  Peak load calculation:
  =====================
  Concurrent matches at peak:     ~50 (Grand Slam day session)
  Points per match per hour:       ~150
  Predictions per point:           1 (point-win) + 1 (match-win)

  Steady-state:  50 * 150 / 3600 = ~2 req/sec
  Burst (many points at once):     ~20 req/sec

  This is LOW volume. A single FastAPI instance handles ~1,000 req/sec.
  Scaling is for reliability (redundancy), not throughput.
```

### 4.2 Why This System Is Latency-Sensitive, Not Throughput-Sensitive

Tennis betting is the opposite of web search or social feeds:
- **Low QPS**: tens, not millions
- **Hard latency SLA**: odds must update before the next point starts (~15-25 seconds between points)
- **High value per request**: each prediction drives pricing decisions

This means:
- **Don't over-architect**: 2-3 FastAPI pods behind an ALB is sufficient
- **Focus on P99 latency**, not horizontal scaling
- **Co-locate** prediction service and Redis in the same region as the data provider
- **Pre-warm** ONNX session and player mappings at startup

---

## 5. Handling Edge Cases

| Edge Case | Strategy |
|-----------|----------|
| Unknown player (new/unranked) | Encode as a special `UNKNOWN` token (code 0). Model degrades to game-state-only prediction |
| Data provider outage | Cache last known match state. Alert on-call. Serve stale predictions with confidence flag |
| Model returns NaN/Inf | Input validation (Pydantic), output clamping to [0.01, 0.99], alert on anomaly |
| Retirement / walkover | Detect match-end event, stop predicting, set P(winner) = 1.0 |
| Rain delay / suspension | Freeze state, resume when play restarts. Long delays may warrant model re-inference with updated features |
| Tiebreak scoring | Already handled: `in_tiebreak` feature + analytic model uses tiebreak recursion |

---

## 6. Infrastructure Choices

```
  DEPLOYMENT TOPOLOGY
  ====================

  Region: eu-west-1 (close to London betting exchanges)

  +--------------------------------------------------+
  |  Kubernetes Cluster                               |
  |                                                   |
  |  namespace: tennis-predictor                      |
  |                                                   |
  |  +-----------+  +-----------+  +-----------+      |
  |  | pred-svc  |  | pred-svc  |  | pred-svc  |     |
  |  | (FastAPI) |  | (FastAPI) |  | (FastAPI) |     |
  |  | pod 1     |  | pod 2     |  | pod 3     |     |
  |  +-----------+  +-----------+  +-----------+      |
  |        |              |              |             |
  |  +-----+--------------+--------------+------+     |
  |  |              Redis (ElastiCache)         |     |
  |  |         feature store (online)           |     |
  |  +-----------------------------------------+     |
  |                                                   |
  |  +-----------------------------------------+     |
  |  |          Kafka (MSK)                     |     |
  |  |     topic: tennis.points                 |     |
  |  |     partitions: 16 (by match_id hash)    |     |
  |  +-----------------------------------------+     |
  |                                                   |
  +--------------------------------------------------+

  External:
  - S3: model artifacts, training data, event archive
  - RDS/Postgres: player registry, match metadata
  - MLflow (EC2 or managed): experiment tracking
  - Airflow (MWAA): training pipeline orchestration
  - CloudWatch / Datadog: monitoring + alerting
```

**Cost estimate** (AWS, minimal production):

| Component | Spec | Monthly Cost |
|-----------|------|-------------|
| EKS cluster | 3x t3.medium | ~$120 |
| ElastiCache Redis | cache.t3.small | ~$25 |
| MSK Kafka | kafka.t3.small, 3 broker | ~$200 |
| S3 | <10GB | ~$1 |
| RDS Postgres | db.t3.micro | ~$15 |
| Data provider | Sportradar basic | ~$500-2000 |
| **Total** | | **~$900-2400/mo** |

The data provider is the dominant cost. Infrastructure is cheap because volume is low.

---

## 7. What the Demo Already Proves

The demo isn't just a toy — it validates the hardest parts:

```
  PRODUCTION CONCERN          DEMO VALIDATION
  =====================       ===========================

  "Can your model serve       ONNX inference: 0.022ms
   under 100ms?"              (4,500x margin)

  "How do you handle          XGBoost native categoricals
   1,800 players without      (no one-hot explosion)
   feature explosion?"

  "How do you avoid           Match-level train/val split
   data leakage?"             (not random row split)

  "How do you get from        Analytic recursive model:
   P(point) to P(match)?"     point → game → set → match

  "How do you retrain?"       Optuna + CLI pipeline:
                              python -m tennis_predictor.training

  "How do you export          ONNX export + verification
   for production?"           (max diff < 1e-6)

  "How do you handle          Label encoding + player_mapping.json
   categoricals in ONNX?"     (ONNX doesn't support categoricals)
```

**Interview tip**: lead with "here's what I built and measured" before discussing what you'd add. Concrete numbers (0.022ms, 400KB model, 2.8M training points) are more persuasive than theoretical architecture.

---

## 8. Phased Rollout Plan

If asked "how would you actually ship this?":

**Phase 1 — Serving layer (1-2 weeks)**
- FastAPI + ONNX Runtime endpoint (`/predict`)
- Docker container, health checks
- Player mapping loaded at startup
- Deploy to single EC2 instance or Cloud Run

**Phase 2 — Live data integration (2-3 weeks)**
- Connect to data provider WebSocket
- Event normaliser → Kafka → prediction service
- Redis for match state caching
- WebSocket output to consumers

**Phase 3 — Training automation (1-2 weeks)**
- Airflow DAG: ingest new data → retrain → evaluate → promote
- MLflow experiment tracking
- Model versioning in S3

**Phase 4 — Monitoring + hardening (1-2 weeks)**
- Prometheus metrics export
- Grafana dashboards (latency, drift, accuracy)
- Alerting rules
- Load testing with historical replay

**Phase 5 — Advanced features (ongoing)**
- Surface-specific models (clay vs hard vs grass)
- In-match momentum features (consecutive points won)
- Head-to-head history features
- Live ELO updates
