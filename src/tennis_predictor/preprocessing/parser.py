"""Point-by-point string parser for Jeff Sackmann's tennis dataset.

Walks a compressed PBP string character by character, maintaining a GameState
and emitting one record per point with the full state snapshot BEFORE the point.

Format (from dataset_rules.txt):
  S = server won, R = returner won, A = ace, D = double fault
  ; = game end, . = set end (replaces ; for last game of a set)
  / = tiebreak serve change
"""

from typing import Dict, List, Optional

from .models import GameState, ParseResult, to_tennis_score


class MatchParser:
    """Parses a single match's compressed PBP string into point records.

    Each record captures the full game state BEFORE a point is played,
    plus the outcome (who won the point). Points are stored in tennis
    representation (0, 15, 30, 40).

    Pure parsing only — validation is handled by the pipeline via ScoreValidator.

    Example:
        >>> parser = MatchParser()
        >>> result = parser.parse(
        ...     match_id="test", player_1="Djokovic", player_2="Alcaraz",
        ...     pbp_sequence="SSSS;RRRR", tournament="Wimbledon", date="2023-07-16",
        ... )
        >>> len(result.points)       # 4 server wins + 4 returner wins = 8 points
        8
        >>> result.points[0]         # state BEFORE first point: 0-0, P1 serving
        {'match_id': 'test', ..., 'points_p1': 0, 'points_p2': 0, 'serving_player': 1, 'server_wins': 1}
        >>> result.set_scores        # no set completed yet (only 2 games played)
        []
    """

    def parse(
        self,
        match_id: str,
        player_1: str,
        player_2: str,
        pbp_sequence: str,
        tournament: str,
        date: str,
    ) -> ParseResult:
        """Parse a compressed PBP string into a list of game state records.

        Args:
            match_id: Unique match identifier.
            player_1: Name of player 1 (server at match start).
            player_2: Name of player 2.
            pbp_sequence: Compressed point sequence (e.g. "SSSS;RRRR;SRSR").
            tournament: Tournament name.
            date: Match date string.

        Returns:
            ParseResult with points and set_scores for downstream validation.

        Example:
            "SSSS;RRRR" → 8 point records:
              - Points 1-4: P1 serves at 0-0, 15-0, 30-0, 40-0 → all server_wins=1
              - ';' triggers _end_game → P1 wins game, score becomes 1-0, server switches to P2
              - Points 5-8: P2 serves at 0-0, 0-15, 0-30, 0-40 → all server_wins=0
              (returner P1 wins each point, so it's a break of serve)
        """
        points: List[Dict] = []
        state = GameState()
        last_point_winner: Optional[int] = None

        for char in pbp_sequence:
            point_winner = None
            is_ace = 0
            is_double_fault = 0

            if char == "S":
                point_winner = state.serving_player
            elif char == "R":
                point_winner = 3 - state.serving_player
            elif char == "A":
                point_winner = state.serving_player
                is_ace = 1
            elif char == "D":
                point_winner = 3 - state.serving_player
                is_double_fault = 1
            elif char == ";":
                self._end_game(state, last_point_winner)
                continue
            elif char == ".":
                # '.' replaces ';' for the last game of a set
                self._end_game(state, last_point_winner)
                continue
            elif char == "/":
                state.serving_player = 3 - state.serving_player
                state.in_tiebreak = True
                continue
            else:
                continue

            if point_winner is None:
                continue

            # Record game state BEFORE the point is played
            srv = state.serving_player

            # Derived features use raw counts for correct logic
            is_deuce = int(
                state.points_p1 >= 3
                and state.points_p2 >= 3
                and state.points_p1 == state.points_p2
            )

            if srv == 1:
                is_break_point = int(
                    state.points_p2 >= 3 and state.points_p2 > state.points_p1
                )
            else:
                is_break_point = int(
                    state.points_p1 >= 3 and state.points_p1 > state.points_p2
                )

            server_wins = 1 if point_winner == srv else 0

            point_record = {
                "match_id": match_id,
                "tournament": tournament,
                "date": date,
                "player_1": player_1,
                "player_2": player_2,
                "set_num": state.set_num,
                "game_num": state.game_num,
                "sets_p1": state.sets_p1,
                "sets_p2": state.sets_p2,
                "games_p1": state.games_p1,
                "games_p2": state.games_p2,
                "points_p1": to_tennis_score(state.points_p1),
                "points_p2": to_tennis_score(state.points_p2),
                "serving_player": srv,
                "in_tiebreak": int(state.in_tiebreak),
                "is_deuce": is_deuce,
                "is_break_point": is_break_point,
                "is_ace": is_ace,
                "is_double_fault": is_double_fault,
                "point_winner": point_winner,
                "server_wins": server_wins,
            }
            points.append(point_record)

            # Update state AFTER recording
            if point_winner == 1:
                state.points_p1 += 1
            else:
                state.points_p2 += 1

            last_point_winner = point_winner

        # Finalize the last game (no trailing ';' or '.' at end of match)
        if last_point_winner is not None and (
            state.points_p1 > 0 or state.points_p2 > 0
        ):
            self._end_game(state, last_point_winner)

        return ParseResult(points=points, set_scores=list(state.set_scores))

    @staticmethod
    def _end_game(state: GameState, last_point_winner: Optional[int]) -> None:
        """Handle game boundary: award game, check set completion, switch server.

        Called when ';' (game end), '.' (set end), or end-of-string is reached.

        Example:
            State before: games_p1=5, games_p2=4, serving_player=1, last_point_winner=1
            After _end_game:
              - games_p1 → 6 (P1 wins game), points reset to 0-0
              - serving_player → 2 (server switches)
              - Set won (6-4): sets_p1 += 1, games reset, set_scores appends (6, 4)
        """
        if last_point_winner == 1:
            state.games_p1 += 1
        elif last_point_winner == 2:
            state.games_p2 += 1

        state.points_p1 = 0
        state.points_p2 = 0
        state.game_num += 1

        # Switch server (not during tiebreak — '/' handles that)
        if not state.in_tiebreak:
            state.serving_player = 3 - state.serving_player

        # Check for set completion
        g1, g2 = state.games_p1, state.games_p2
        set_won = False

        if g1 >= 6 and g1 - g2 >= 2:
            set_won = True
        elif g2 >= 6 and g2 - g1 >= 2:
            set_won = True
        elif g1 == 7 or g2 == 7:
            set_won = True
            state.in_tiebreak = False

        if set_won:
            state.set_scores.append((state.games_p1, state.games_p2))
            if state.games_p1 > state.games_p2:
                state.sets_p1 += 1
            else:
                state.sets_p2 += 1
            state.games_p1 = 0
            state.games_p2 = 0
            state.set_num += 1
            state.game_num = 1

        # Check if tiebreak should start (6-6 in games)
        if state.games_p1 == 6 and state.games_p2 == 6:
            state.in_tiebreak = True
