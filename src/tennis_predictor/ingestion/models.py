"""Typed message envelopes for the tennis event stream.

Two dataclasses, two topics (Kafka analogy):
- MatchEvent  → 'tennis.points'  — emitted by the producer upstream
- OddsUpdate  → 'tennis.odds'    — published by the consumer downstream

Keeping them separate enforces Single Responsibility: the producer does not
know what the consumer does with the event (Open/Closed Principle).
In production these would be Avro / Protobuf schemas in a schema registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchEvent:
    """A single point in a tennis match — the atomic unit of the event stream.

    Contains the full game state BEFORE the point is played.
    Analogous to a Kafka message on a 'tennis.points' topic.

    Example:
        event = MatchEvent(
            match_id="Federer_Nadal_2024",
            point_index=0,
            player_1="Federer",
            player_2="Nadal",
            server=1,
            point_winner=1,
            server_wins=1,
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            in_tiebreak=False,
            is_deuce=False,
            is_break_point=False,
        )
    """

    match_id:       str
    point_index:    int          # 0-based position within the match
    player_1:       str
    player_2:       str
    server:         int          # 1 or 2
    point_winner:   int          # 1 or 2
    server_wins:    int          # 1 if server won, 0 if returner won
    sets_p1:        int
    sets_p2:        int
    games_p1:       int
    games_p2:       int
    points_p1:      int
    points_p2:      int
    in_tiebreak:    bool
    is_deuce:       bool
    is_break_point: bool
    timestamp_ms:   float = field(default=0.0)  # simulated wall-clock time


@dataclass
class OddsUpdate:
    """Enriched output after inference — published to downstream consumers.

    Contains raw model output plus the end-to-end per-point latency.
    Analogous to a Kafka message on a 'tennis.odds' topic.
    In production this would be published to Redis pub/sub and a time-series DB.

    Example:
        update = OddsUpdate(
            match_id="Federer_Nadal_2024",
            point_index=0,
            p1_win_prob=0.60,
            p2_win_prob=0.40,
            p_server_wins=0.60,
            latency_ms=0.04,
        )
    """

    match_id:      str
    point_index:   int
    p1_win_prob:   float        # P(player_1 wins next point) — raw ONNX, server-relative
    p2_win_prob:   float        # 1 - p1_win_prob
    p_server_wins: float        # raw ONNX output
    latency_ms:    float        # time from event arrival → odds published
    p1_set_win:  float = 0.5  # P(player_1 wins current set) — recursive analytic model
    p2_set_win:  float = 0.5  # 1 - p1_set_win
