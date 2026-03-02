"""GameStateManager: incremental delta state for live match tracking.

Core insight: do NOT recompute the full feature vector from scratch on every
point. Maintain a stateful object and apply deltas — only the fields that
change after each event. This is O(1) per point vs O(n_features).

Redis analogy: this object is a Redis hash. apply_event() issues HSET per
changed field; to_feature_vector() is a single HGETALL. In production the
hash lives in Redis (sub-millisecond reads), decoupled from the inference
service. This is the feature store pattern.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .models import MatchEvent


# Feature order must match the ONNX model's input schema.
# Defined once here and imported by InferenceEngine.
FEATURE_COLS: list[str] = [
    "player_1",
    "player_2",
    "sets_p1",
    "sets_p2",
    "games_p1",
    "games_p2",
    "points_p1",
    "points_p2",
    "serving_player",
    "in_tiebreak",
    "is_deuce",
    "is_break_point",
]

# ── Analytic match-win probability (Carter–Pollard tennis model) ──────────────
# Three recursive layers: point → game → set → match.
# Called on every event, so _p_set is memoised with integer-rounded floats.

_PTS_IDX: dict[int, int] = {0: 0, 15: 1, 30: 2, 40: 3}
_EMA_ALPHA: float = 0.15   # smoothing factor for per-player serve-rate EMA
_EMA_INIT:  float = 0.60   # ATP average serve-win-rate prior


def _p_game(p: float, si: int, sj: int) -> float:
    """P(server wins game | server_pts_idx=si, returner_pts_idx=sj).

    si, sj ∈ {0,1,2,3} (0→0 pts, 1→15, 2→30, 3→40).
    At deuce (3-3) uses the closed-form geometric-series solution.
    Depth ≤ 8 recursive calls — fast enough without caching.
    """
    if si >= 4:
        return 1.0
    if sj >= 4:
        return 0.0
    if si == 3 and sj == 3:          # deuce
        q = 1.0 - p
        return (p * p) / (p * p + q * q)
    q = 1.0 - p
    return p * _p_game(p, si + 1, sj) + q * _p_game(p, si, sj + 1)


@lru_cache(maxsize=4096)
def _p_set_memo(gi: int, gj: int, p1_next: bool,
                pg1_r: int, pg2_r: int) -> float:
    """Memoised P(P1 wins set | gi games P1, gj games P2, P1 serves next if p1_next).

    pg1_r = round(P(P1 wins game when P1 serves) × 1000)  — integer key.
    pg2_r = round(P(P2 wins game when P2 serves) × 1000)  — integer key.

    Servers alternate each game. Tiebreak at 6-6 is approximated with the
    deuce formula applied to each player's average point-win probability.
    """
    pg1 = pg1_r / 1000.0
    pg2 = pg2_r / 1000.0

    if gi >= 6 and gi - gj >= 2:   return 1.0   # 6-0 … 6-4, 7-5
    if gj >= 6 and gj - gi >= 2:   return 0.0
    if gi == 7:                     return 1.0   # won tiebreak
    if gj == 7:                     return 0.0

    if gi == 6 and gj == 6:        # tiebreak — deuce approximation on avg prob
        q_avg = 0.5 * (pg1 + 1.0 - pg2)
        return (q_avg * q_avg) / (q_avg * q_avg + (1.0 - q_avg) ** 2)

    p1_wins_game = pg1 if p1_next else (1.0 - pg2)
    return (
        p1_wins_game       * _p_set_memo(gi + 1, gj, not p1_next, pg1_r, pg2_r)
        + (1.0 - p1_wins_game) * _p_set_memo(gi, gj + 1, not p1_next, pg1_r, pg2_r)
    )


def _p_set(gi: int, gj: int, p1_next: bool, pg1: float, pg2: float) -> float:
    """Public wrapper: rounds pg1/pg2 to integers for cache keying."""
    return _p_set_memo(gi, gj, p1_next, round(pg1 * 1000), round(pg2 * 1000))


def _p_match(si: int, sj: int, pw: float, target: int = 2) -> float:
    """P(P1 wins match | si sets P1, sj sets P2). target=2 for best-of-3.

    pw = P(P1 wins a set), assumed constant across future sets (average of
    serve-first and return-first scenarios — standard simplification).
    Depth ≤ 2 recursive calls for best-of-3.
    """
    if si >= target:  return 1.0
    if sj >= target:  return 0.0
    return (
        pw       * _p_match(si + 1, sj, pw, target)
        + (1.0 - pw) * _p_match(si, sj + 1, pw, target)
    )


class GameStateManager:
    """Maintains a live game state and applies point-level deltas.

    Player names are encoded to integers once at match start (not per point).
    All per-point updates are O(1) dictionary writes.

    Interview talking point: in production this object lives in Redis. Each
    field is a hash key updated with HINCRBY / HSET on each event. The model
    reads a pre-built feature vector with a single HGETALL (~0.1 ms). A DB
    query per inference call would cost 1–10 ms — a 10–100× latency hit.

    Example:
        mgr = GameStateManager(player_1_enc=42, player_2_enc=17)
        feature_dict = mgr.apply_event(event)
        fv_np = mgr.to_feature_vector()   # float32 array for ONNX
        mgr.reset()                        # between matches
    """

    def __init__(self, player_1_enc: int, player_2_enc: int) -> None:
        """
        Args:
            player_1_enc: Integer encoding for player 1 (from player_mapping.json).
            player_2_enc: Integer encoding for player 2.
        """
        self._p1_enc = player_1_enc
        self._p2_enc = player_2_enc
        self._state: dict = {}
        # EMA serve-win rates — updated on each event, used by set_win_probability.
        # Initialised to the ATP average (~60 %).
        self._ema_p1: float = _EMA_INIT   # P(P1 wins point when P1 serves)
        self._ema_p2: float = _EMA_INIT   # P(P2 wins point when P2 serves)

    def apply_event(self, event: MatchEvent) -> dict:
        """Update state from a point event and return the current feature dict.

        Only writes the fields present in the event — O(1) regardless of
        total match length. In production each write is a Redis HSET call.
        """
        self._state = {
            "player_1":       self._p1_enc,
            "player_2":       self._p2_enc,
            "sets_p1":        event.sets_p1,
            "sets_p2":        event.sets_p2,
            "games_p1":       event.games_p1,
            "games_p2":       event.games_p2,
            "points_p1":      event.points_p1,
            "points_p2":      event.points_p2,
            "serving_player": event.server,
            "in_tiebreak":    int(event.in_tiebreak),
            "is_deuce":       int(event.is_deuce),
            "is_break_point": int(event.is_break_point),
        }
        # Update EMA serve-win rate for whoever served this point.
        if event.server == 1:
            self._ema_p1 = (
                _EMA_ALPHA * event.server_wins
                + (1.0 - _EMA_ALPHA) * self._ema_p1
            )
        else:
            self._ema_p2 = (
                _EMA_ALPHA * event.server_wins
                + (1.0 - _EMA_ALPHA) * self._ema_p2
            )
        return self._state

    def to_feature_vector(self) -> np.ndarray:
        """Serialise current state to a (1, 12) float32 array in FEATURE_COLS order.

        The shape (1, n_features) matches ONNX Runtime's expected batch input.
        In production: equivalent to Redis HGETALL + fixed-order serialisation.
        """
        return np.array(
            [self._state[col] for col in FEATURE_COLS], dtype=np.float32
        ).reshape(1, -1)

    def current_state(self) -> dict:
        """Return a copy of the current state dict (read-only snapshot)."""
        return dict(self._state)

    def set_win_probability(self, p_srv: float) -> tuple[float, float]:
        """Compute P(P1 wins the current set) using a two-layer recursive model.

        p_srv: P(current server wins this point) — raw ONNX output.

        Two-layer recursion:
          1. P(server wins current game | current game score + p_srv)
          2. P(P1 wins current set   | current games score + layer 1 result)

        Server switching is handled explicitly: each future game in the set uses
        the EMA serve rate of whoever serves that game (servers alternate each game).
        The probability resets to ~0.5 at the start of each new set, producing a
        chart that oscillates within each set and clearly shows who is dominating.

        Returns (p1_set_win, p2_set_win) — sum to 1.0.
        """
        s = self._state
        if not s:
            return 0.5, 0.5

        server   = int(s["serving_player"])   # 1 or 2
        is_deuce = bool(s["is_deuce"])

        # ── Layer 1: probability server wins the current in-progress game ──────
        if is_deuce:
            q = 1.0 - p_srv
            p_cur_game = (p_srv * p_srv) / (p_srv * p_srv + q * q)
        else:
            # Map tennis point scores (0/15/30/40) to recursion indices (0-3).
            if server == 1:
                si_pts = _PTS_IDX.get(int(s["points_p1"]), 0)
                sj_pts = _PTS_IDX.get(int(s["points_p2"]), 0)
            else:
                si_pts = _PTS_IDX.get(int(s["points_p2"]), 0)
                sj_pts = _PTS_IDX.get(int(s["points_p1"]), 0)
            p_cur_game = _p_game(p_srv, si_pts, sj_pts)

        # Convert to P1's perspective.
        p1_wins_cur_game = p_cur_game if server == 1 else (1.0 - p_cur_game)

        # ── Layer 2: probability P1 wins the current set ──────────────────────
        # Game-win probabilities from 0-0 for *future* games (uses EMA, stable).
        pg1 = _p_game(self._ema_p1, 0, 0)   # P(P1 wins game when P1 serves)
        pg2 = _p_game(self._ema_p2, 0, 0)   # P(P2 wins game when P2 serves)

        gi = int(s["games_p1"])
        gj = int(s["games_p2"])
        # After this game completes the OTHER player serves the next game.
        p1_serves_next_game = (server != 1)

        p1_wins_cur_set = (
            p1_wins_cur_game       * _p_set(gi + 1, gj, p1_serves_next_game, pg1, pg2)
            + (1.0 - p1_wins_cur_game) * _p_set(gi, gj + 1, p1_serves_next_game, pg1, pg2)
        )

        return float(p1_wins_cur_set), float(1.0 - p1_wins_cur_set)

    def reset(self) -> None:
        """Clear state and EMA. Call between matches when reusing the same instance."""
        self._state  = {}
        self._ema_p1 = _EMA_INIT
        self._ema_p2 = _EMA_INIT
