"""MatchEventProducer: converts a match DataFrame into a stream of MatchEvent objects.

In production this class is replaced by a Kafka consumer reading from a live
'tennis.raw' topic. The produce() interface is identical — the downstream
pipeline is completely source-agnostic (producer/consumer decoupling).

speed_factor=0.0  → instant replay (backtesting, default)
speed_factor=1.0  → real-time simulation (1 point every ~20 s)
"""

from __future__ import annotations

import time
from typing import Iterator

import pandas as pd

from .models import MatchEvent


class MatchEventProducer:
    """Reads play-by-play game states and emits MatchEvent objects.

    Replays the pre-parsed GameState records from Phase 1 output CSVs
    and attaches a simulated timestamp based on configurable speed_factor.

    Interview talking point: In production, replace the CSV read with a
    KafkaConsumer loop on the 'tennis.raw' topic. The produce() method
    signature and the MatchEvent envelope stay identical — the rest of
    the pipeline never changes.

    Example:
        producer = MatchEventProducer(match_df, speed_factor=0.0)
        for event in producer.produce():
            queue.push(event)
    """

    _POINT_INTERVAL_S: float = 20.0  # average seconds between points in a real match

    def __init__(self, match_df: pd.DataFrame, speed_factor: float = 0.0) -> None:
        """
        Args:
            match_df:     Rows from full_game_states.csv for a single match.
                          Must contain columns produced by TrainingPipeline.
            speed_factor: 0.0 = instant (backtesting), 1.0 = real-time.
                          Values > 1.0 fast-forward (e.g. 10.0 = 10× speed).
        """
        self._df = match_df.reset_index(drop=True)
        self._speed = speed_factor

    def produce(self) -> Iterator[MatchEvent]:
        """Yield one MatchEvent per point, optionally sleeping to simulate real time.

        Yields:
            MatchEvent with a simulated timestamp_ms proportional to point_index.
        """
        t0 = time.monotonic()

        for i, row in self._df.iterrows():
            simulated_ts = i * self._POINT_INTERVAL_S * 1000  # convert to ms

            if self._speed > 0:
                target_s = t0 + (simulated_ts / 1000.0) / self._speed
                sleep_s = target_s - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

            yield MatchEvent(
                match_id=str(row["match_id"]),
                point_index=int(i),
                player_1=str(row["player_1"]),
                player_2=str(row["player_2"]),
                server=int(row["serving_player"]),
                point_winner=int(row["point_winner"]),
                server_wins=int(row["server_wins"]),
                sets_p1=int(row["sets_p1"]),
                sets_p2=int(row["sets_p2"]),
                games_p1=int(row["games_p1"]),
                games_p2=int(row["games_p2"]),
                points_p1=int(row["points_p1"]),
                points_p2=int(row["points_p2"]),
                in_tiebreak=bool(row["in_tiebreak"]),
                is_deuce=bool(row["is_deuce"]),
                is_break_point=bool(row["is_break_point"]),
                timestamp_ms=float(simulated_ts),
            )
