"""Tennis scoring data model and constants.

Tracks the evolving state of a tennis match at each point.
Used by MatchParser to record the game state BEFORE each point is played.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Raw point count → tennis score representation
# At deuce (both >= 3), score stays at 40; is_deuce/is_break_point capture the rest
TENNIS_POINTS = {0: 0, 1: 15, 2: 30}


def to_tennis_score(raw: int) -> int:
    """Convert raw point count (0,1,2,3,...) to tennis representation (0,15,30,40).

    Example:
        >>> to_tennis_score(0)
        0
        >>> to_tennis_score(1)
        15
        >>> to_tennis_score(2)
        30
        >>> to_tennis_score(3)   # deuce and beyond all map to 40
        40
    """
    return TENNIS_POINTS.get(raw, 40)


@dataclass
class GameState:
    """Full game state at a specific moment in a tennis match.

    This is the snapshot BEFORE a point is played — the feature vector
    for predicting who wins the next point.

    Example:
        >>> state = GameState(sets_p1=1, sets_p2=0, games_p1=3, games_p2=2,
        ...                   points_p1=2, points_p2=1, serving_player=1)
        >>> # Represents: P1 leads 1-0 in sets, 3-2 in games, 30-15 in points, P1 serving
    """

    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0  # games in current set
    games_p2: int = 0
    points_p1: int = 0  # raw count: 0,1,2,3,4,...
    points_p2: int = 0
    serving_player: int = 1  # 1 or 2
    set_num: int = 1
    game_num: int = 1  # game number within the set
    in_tiebreak: bool = False

    # Set history for score validation
    set_scores: List[tuple] = field(default_factory=list)


@dataclass
class ParseResult:
    """Output of MatchParser.parse() — points plus metadata for validation.

    Example:
        >>> result = ParseResult(
        ...     points=[{"sets_p1": 0, "points_p1": 0, "server_wins": 1, ...}],
        ...     set_scores=[(6, 4), (6, 1)],
        ... )
        >>> len(result.points)    # one dict per point in the match
        1
        >>> result.set_scores     # passed to ScoreValidator for verification
        [(6, 4), (6, 1)]
    """

    points: List[Dict]
    set_scores: List[Tuple[int, int]]
