"""Real-time tennis live-odds dashboard.

Architecture (mirrors the streaming pattern from notebook 04, section 8):

    st.session_state["queue"]         EventQueue   — pre-filled at match start
    st.session_state["state_manager"] GameStateManager — live score state
    st.session_state["publisher"]     OddsPublisher — grows by 1 per tick
    st.session_state["engine"]        InferenceEngine — loaded once at app start
          │
          ▼  (each st.rerun() tick)
    queue.pop() → state.apply_event() → engine.predict() → publisher.publish()
          │
          ▼
    odds_df.rolling(WINDOW).mean() → st.line_chart   # chart grows one point
    st.rerun()                                        # schedule next tick

Interview talking point: each re-render is a "tick" — read current state,
consume one event, write updated state. This mirrors the actor model (Akka)
or Flink's stateful stream processing at the single-process level.

Launch:
    uv run streamlit run src/tennis_predictor/dashboard/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from tennis_predictor.ingestion import (
    EventQueue,
    GameStateManager,
    IngestionConfig,
    IngestionPipeline,
    MatchEvent,
    MatchEventProducer,
    OddsPublisher,
    OddsUpdate,
)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW = 20  # rolling mean window (win probability + latency)

# ── Paths ─────────────────────────────────────────────────────────────────────
# app.py: src/tennis_predictor/dashboard/app.py → project root is 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "xgb_server_wins.onnx"
MAPPING_PATH = PROJECT_ROOT / "models" / "player_mapping.json"
PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "data_download" / "training" / "processed"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_processed_csv() -> Path:
    """Return full_game_states.csv from the most recent preprocessed_* folder."""
    run_dirs = sorted(PROCESSED_DIR.glob("preprocessed_*"), reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No preprocessed_* dirs in {PROCESSED_DIR}")
    return run_dirs[0] / "full_game_states.csv"


def _build_catalogue(pipeline: IngestionPipeline) -> pd.DataFrame:
    """Build a clean-match catalogue sorted by length (most points first).

    Filters out matches where the Sackmann CSV PBP string was split across
    multiple source rows — these produce spurious all-zero score resets
    mid-match (see notebook 04, section 2 for full explanation).

    Interview talking point: in production this metadata lives in a separate
    index table (not computed from the raw event log every startup).
    """
    _df      = pipeline._df_states
    _row_num = _df.groupby("match_id").cumcount()

    # A reset row: after row 0, all six score fields are simultaneously 0.
    # Adding points_p1/p2 = 0 is the key fix: once the first point is played
    # they become non-zero, so this condition only fires at genuine resets.
    _is_reset_row = (
        (_row_num > 0)
        & (_df["sets_p1"]   == 0) & (_df["sets_p2"]   == 0)
        & (_df["games_p1"]  == 0) & (_df["games_p2"]  == 0)
        & (_df["points_p1"] == 0) & (_df["points_p2"] == 0)
    )
    _clean_flags = (
        ~_df.assign(_reset=_is_reset_row)
        .groupby("match_id")["_reset"].any()
    ).rename("is_clean")

    catalogue = (
        _df.groupby("match_id")
        .agg(
            n_points=("server_wins", "count"),
            player_1=("player_1",    "first"),
            player_2=("player_2",    "first"),
            sets_p1 =("sets_p1",     "max"),
            sets_p2 =("sets_p2",     "max"),
        )
        .sort_values("n_points", ascending=False)
        .reset_index()
        .join(_clean_flags, on="match_id")
    )
    catalogue["label"] = (
        catalogue["player_1"] + " vs " + catalogue["player_2"]
        + "  (" + catalogue["n_points"].astype(str) + " pts, "
        + catalogue["sets_p1"].astype(str) + "-"
        + catalogue["sets_p2"].astype(str) + " sets)"
    )
    return catalogue[catalogue["is_clean"]].reset_index(drop=True)


def fmt_score(ev: MatchEvent, player: int) -> str:
    """Format a MatchEvent into a sidebar metric string.

    Example output: '1 sets  4 games  30'
    The CSV stores tennis scores directly (0, 15, 30, 40) — no conversion.
    """
    sets  = getattr(ev, f"sets_p{player}")
    games = getattr(ev, f"games_p{player}")
    pts   = getattr(ev, f"points_p{player}")
    return f"{sets} sets  {games} games  {pts}"


def fmt_server(ev: MatchEvent, p1: str, p2: str) -> str:
    """Format the serving / tiebreak / break-point status line.

    Uses ev.server (MatchEvent field), not 'serving_player' (CSV column name).
    """
    name = p1 if ev.server == 1 else p2
    tb   = "  [TIEBREAK]"   if ev.in_tiebreak    else ""
    bp   = "  BREAK POINT"  if ev.is_break_point  else ""
    return f"Serving: {name}{tb}{bp}"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Live Odds",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 1. Load heavy singletons once ────────────────────────────────────────────
# InferenceEngine (ONNX session), IngestionPipeline (CSV + player mapping),
# and match catalogue are expensive to build — load once at app start and
# persist in st.session_state. Never re-instantiate per re-render.
if "pipeline" not in st.session_state:
    with st.spinner("Loading model and match data (~500 ms)…"):
        config = IngestionConfig(
            processed_csv=_discover_processed_csv(),
            model_path=MODEL_PATH,
            player_mapping_path=MAPPING_PATH,
        )
        st.session_state["pipeline"]  = IngestionPipeline(config)
        st.session_state["engine"]    = st.session_state["pipeline"]._engine
        st.session_state["catalogue"] = _build_catalogue(
            st.session_state["pipeline"]
        )

pipeline  = st.session_state["pipeline"]
catalogue = st.session_state["catalogue"]

# ── 2. Sidebar controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("Tennis Live Odds")
    st.caption(f"{len(catalogue):,} clean matches available")

    selected_label = st.selectbox(
        "Match",
        options=catalogue["label"].tolist(),
        index=0,
    )
    match_id = catalogue.loc[
        catalogue["label"] == selected_label, "match_id"
    ].iloc[0]

    st.divider()
    col1, col2 = st.columns(2)
    start = col1.button("▶ Play",  use_container_width=True)
    stop  = col2.button("⏸ Pause", use_container_width=True)

    # Tick delay controls how long the app sleeps between re-renders.
    # Lower = faster replay; higher = easier to read each update.
    tick_delay = st.slider(
        "Speed (s / point)", min_value=0.05, max_value=2.0,
        value=0.15, step=0.05,
        help="Time between successive point updates",
    )

# ── 3. Match initialisation ───────────────────────────────────────────────────
# Reset all per-match state when the user picks a different match.
# Heavy objects (engine, pipeline) are NOT re-created — only the per-match
# queue, state_manager, and publisher are reset.
if st.session_state.get("match_id") != match_id:
    match_df = (
        pipeline._df_states[pipeline._df_states["match_id"] == match_id]
        .reset_index(drop=True)
    )
    p1 = str(match_df["player_1"].iloc[0])
    p2 = str(match_df["player_2"].iloc[0])
    p1_enc = int(pipeline._player_mapping["player_1"].get(p1, 0))
    p2_enc = int(pipeline._player_mapping["player_2"].get(p2, 0))

    # Pre-fill the queue with all events for this match.
    # In production the queue is populated by a live Kafka topic consumer.
    events = list(MatchEventProducer(match_df, speed_factor=0.0).produce())
    eq = EventQueue(max_size=len(events) + 10)
    for ev in events:
        eq.push(ev)

    st.session_state.update(
        dict(
            match_id=match_id,
            p1_name=p1,
            p2_name=p2,
            queue=eq,
            state_manager=GameStateManager(p1_enc, p2_enc),
            publisher=OddsPublisher(),
            current_event=None,
            running=False,
        )
    )

# ── 4. Play / pause ───────────────────────────────────────────────────────────
if start:
    st.session_state["running"] = True
if stop:
    st.session_state["running"] = False

# ── 5. Streaming tick: one event per re-render ────────────────────────────────
# Cost per tick: ~0.01 ms → 1,600× under Streamlit's 16 ms frame budget.
# Interview talking point: this is the actor pattern — each re-render is a
# deterministic state transition. No threads, no shared mutable state.
if st.session_state.get("running"):
    if not st.session_state["queue"].is_empty():
        ev = st.session_state["queue"].pop(timeout_ms=0)
        if ev is not None:
            st.session_state["state_manager"].apply_event(ev)
            fv         = st.session_state["state_manager"].to_feature_vector()
            p_srv, lat = st.session_state["engine"].predict(fv)
            p1_pt      = p_srv if ev.server == 1 else 1.0 - p_srv
            p1_sw, p2_sw = st.session_state["state_manager"].set_win_probability(p_srv)
            st.session_state["publisher"].publish(
                OddsUpdate(
                    match_id=ev.match_id,
                    point_index=ev.point_index,
                    p1_win_prob=p1_pt,
                    p2_win_prob=1.0 - p1_pt,
                    p_server_wins=p_srv,
                    latency_ms=lat,
                    p1_set_win=p1_sw,
                    p2_set_win=p2_sw,
                )
            )
            st.session_state["current_event"] = ev
    else:
        # Match complete — stop the loop automatically
        st.session_state["running"] = False

# ── 6. Render ─────────────────────────────────────────────────────────────────
odds_df = st.session_state["publisher"].to_dataframe()
p1_name = st.session_state.get("p1_name", "Player 1")
p2_name = st.session_state.get("p2_name", "Player 2")
ev      = st.session_state.get("current_event")

n_played = len(odds_df)
n_total  = int(catalogue.loc[catalogue["match_id"] == match_id, "n_points"].iloc[0])

# Header
st.title(f"{p1_name} vs {p2_name}")
st.caption(f"Point {n_played} / {n_total}")

# Live set-win probability chart.
# Uses the recursive analytic model (point→game→set), not the raw ONNX
# serve probability. Resets to ~0.5 at the start of each new set, showing
# within-set momentum clearly. Rolling mean (WINDOW=20) smooths noise.
if not odds_df.empty:
    smoothed = (
        odds_df[["p1_set_win", "p2_set_win"]]
        .rolling(WINDOW, min_periods=1)
        .mean()
    )
    smoothed.columns = [p1_name, p2_name]
    st.line_chart(smoothed, use_container_width=True)
else:
    st.info("Press **▶ Play** in the sidebar to start the simulation.")

# Score ticker — rendered in a second sidebar block so it always appears
# below the controls without the two-phase placeholder flash.
# Markdown table format: column headers make each number self-explanatory.
#   Sets  = sets won in the match so far
#   Games = games won in the current set
#   Pts   = point score in the current game (0 / 15 / 30 / 40)
with st.sidebar:
    st.divider()
    st.caption("Live score")
    if ev is not None:
        st.markdown(
            f"| | Sets | Games | Pts |\n"
            f"|:--|:--:|:--:|:--:|\n"
            f"| **{p1_name}** | {ev.sets_p1} | {ev.games_p1} | {ev.points_p1} |\n"
            f"| **{p2_name}** | {ev.sets_p2} | {ev.games_p2} | {ev.points_p2} |"
        )
        st.caption(fmt_server(ev, p1_name, p2_name))
    else:
        st.caption("Waiting for first point…")

# Latency panel
if not odds_df.empty:
    rolling_lat = float(odds_df["latency_ms"].tail(WINDOW).mean())
    overall_lat = float(odds_df["latency_ms"].mean())
    delta_lat   = rolling_lat - overall_lat

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label=f"Inference latency (rolling mean, last {WINDOW} pts)",
        value=f"{rolling_lat:.3f} ms",
        delta=f"{delta_lat:+.4f} ms",
    )
    col2.metric(
        label="p99 latency",
        value=f"{odds_df['latency_ms'].quantile(0.99):.3f} ms",
    )
    col3.metric(
        label="Budget headroom",
        value=f"{200.0 / overall_lat:,.0f}×",
        help="vs. 200 ms production target",
    )

# ── 7. Schedule next tick ─────────────────────────────────────────────────────
# sleep(tick_delay) lets the browser render the current frame before the next
# rerun replaces it. Without this the loop runs faster than the browser can
# paint, causing the chart to batch updates and the score to flash empty.
# Interview talking point: tick_delay is the equivalent of a Kafka consumer
# poll interval — it controls throughput vs. freshness trade-off.
if st.session_state.get("running"):
    time.sleep(tick_delay)
    st.rerun()
