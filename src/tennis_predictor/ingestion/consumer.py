"""EventConsumer and OddsPublisher: the downstream half of the pipeline.

OddsPublisher  — in-memory store (Redis pub/sub in production).
EventConsumer  — processing loop: queue → state → inference → publish.

In production EventConsumer runs as a thread (or asyncio task) per live match,
continuously polling the EventQueue. Here run_match() is synchronous for
straightforward replay and benchmarking.

Interview talking point: OddsPublisher.publish() is a single Redis PUBLISH
call in production. The Streamlit dashboard subscribes to the same channel
and re-renders on each new OddsUpdate — no polling required.
"""

from __future__ import annotations

import time

import pandas as pd

from .engine import InferenceEngine
from .models import MatchEvent, OddsUpdate
from .queue import EventQueue
from .state import GameStateManager


class OddsPublisher:
    """In-memory odds store — Redis pub/sub + time-series DB in production.

    Stores every OddsUpdate in an ordered list, providing a history view
    and DataFrame export for downstream consumers (e.g. Streamlit dashboard).

    Example:
        pub = OddsPublisher()
        pub.publish(OddsUpdate(...))
        df = pub.to_dataframe()
    """

    def __init__(self) -> None:
        self._history: list[OddsUpdate] = []

    def publish(self, update: OddsUpdate) -> None:
        """Append an OddsUpdate. In production: Redis PUBLISH + LPUSH to history key."""
        self._history.append(update)

    def history(self) -> list[OddsUpdate]:
        """Return a snapshot of all published updates (read-only copy)."""
        return list(self._history)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a DataFrame for plotting and analysis.

        Columns: point_index, p1_win_prob, p2_win_prob, p_server_wins, latency_ms
        """
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "point_index":   u.point_index,
                "p1_win_prob":   u.p1_win_prob,
                "p2_win_prob":   u.p2_win_prob,
                "p_server_wins": u.p_server_wins,
                "p1_set_win":    u.p1_set_win,
                "p2_set_win":    u.p2_set_win,
                "latency_ms":    u.latency_ms,
            }
            for u in self._history
        ])


class EventConsumer:
    """Processing loop: queue → state → inference → publish.

    Wires EventQueue, GameStateManager, InferenceEngine, and OddsPublisher
    together. Each consumed event goes through the full pipeline:
        pop() → apply_event() → to_feature_vector() → predict() → publish()

    Interview talking point: in production this runs as a dedicated thread per
    live match. Synchronous run_match() here is equivalent to a single-threaded
    Kafka consumer processing a replay topic.

    Example:
        consumer = EventConsumer(eq, state_mgr, engine, publisher)
        odds_df  = consumer.run_match(events)
    """

    def __init__(
        self,
        eq: EventQueue,
        state_mgr: GameStateManager,
        engine: InferenceEngine,
        publisher: OddsPublisher,
    ) -> None:
        self._queue = eq
        self._state = state_mgr
        self._engine = engine
        self._pub = publisher

    def run_match(self, events: list[MatchEvent]) -> pd.DataFrame:
        """Push all events through the full pipeline and return the odds DataFrame.

        Per-event path:
            push() → pop() → state.apply_event() → engine.predict() → publisher.publish()

        Returns:
            DataFrame with columns: point_index, p1_win_prob, p2_win_prob,
            p_server_wins, latency_ms.
        """
        for event in events:
            t_start = time.perf_counter()

            if not self._queue.push(event):
                continue  # queue full — backpressure, drop this event

            consumed = self._queue.pop(timeout_ms=50.0)
            if consumed is None:
                continue  # timeout — should not happen in synchronous replay

            self._state.apply_event(consumed)
            fv = self._state.to_feature_vector()

            p_server_wins, _ = self._engine.predict(fv)
            # Convert server-relative probability to player-1-relative:
            # if player 1 is serving, p1_point = p_server_wins
            # if player 2 is serving, p1_point = 1 - p_server_wins
            p1_point = p_server_wins if consumed.server == 1 else 1.0 - p_server_wins

            # Recursive analytic set-win probability (point→game→set).
            p1_set_win, p2_set_win = self._state.set_win_probability(p_server_wins)

            self._pub.publish(OddsUpdate(
                match_id=consumed.match_id,
                point_index=consumed.point_index,
                p1_win_prob=p1_point,
                p2_win_prob=1.0 - p1_point,
                p_server_wins=p_server_wins,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                p1_set_win=p1_set_win,
                p2_set_win=p2_set_win,
            ))

        return self._pub.to_dataframe()
