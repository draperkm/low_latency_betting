"""Score validation: compare parsed set scores against expected match score.

The score column in Sackmann's CSV uses winner-loser format per set
(e.g. '6-4 6-1' means the winner won 6-4 in set 1 and 6-1 in set 2).
We convert to P1-P2 perspective before comparing.
"""

from typing import List, Optional, Tuple


class ScoreValidator:
    """Validates parsed set scores against the expected score string.

    Example:
        >>> v = ScoreValidator()
        >>> v.validate([(6, 4), (6, 1)], "6-4 6-1", match_winner=1)  # match
        None
        >>> v.validate([(6, 4), (6, 1)], "6-4 6-1", match_winner=2)  # mismatch (winner-loser flipped)
        "Set 1 mismatch: parsed (6, 4), expected (4, 6) ..."
    """

    def validate(
        self,
        set_scores: List[Tuple[int, int]],
        expected_score: str,
        match_winner: int,
    ) -> Optional[str]:
        """Compare parsed set scores against expected match score.

        Args:
            set_scores: List of (p1_games, p2_games) tuples from the parser.
            expected_score: Score string from CSV (winner-loser format, e.g. '6-4 3-6 7-5').
            match_winner: 1 or 2, indicating which player won the match.

        Returns:
            None if scores match, error message string if they don't.

        Example:
            Score "6-4 3-6 7-5" with match_winner=2 means P2 won.
            Converted to P1-P2: [(4, 6), (6, 3), (5, 7)].
            If parser produced [(4, 6), (6, 3), (5, 7)] → returns None (match).
            If parser produced [(4, 6), (3, 6), (5, 7)] → returns error string.
        """
        if not expected_score or not expected_score.strip():
            return None

        try:
            expected_sets = self._parse_score_string(expected_score, match_winner)
        except (ValueError, IndexError) as e:
            return f"Score parse error: {e} (score='{expected_score}')"

        if len(expected_sets) != len(set_scores):
            return (
                f"Set count mismatch: parsed {len(set_scores)} sets "
                f"{set_scores}, expected {len(expected_sets)} sets {expected_sets}"
            )

        for i, (parsed, expected) in enumerate(zip(set_scores, expected_sets)):
            if parsed != expected:
                return (
                    f"Set {i+1} mismatch: parsed {parsed}, expected {expected} "
                    f"(full: {set_scores} vs {expected_sets})"
                )

        return None

    def _parse_score_string(
        self, score: str, match_winner: int
    ) -> List[Tuple[int, int]]:
        """Parse score string from winner-loser format to (p1, p2) tuples.

        Handles tiebreak notation like '7-6(4)'.

        Example:
            "7-6(4) 3-6" with match_winner=1 → [(7, 6), (3, 6)]
            "7-6(4) 3-6" with match_winner=2 → [(6, 7), (6, 3)]
        """
        sets = []
        for set_str in score.strip().split():
            parts = set_str.split("-")
            if len(parts) != 2:
                continue

            # Handle tiebreak notation: 7-6(4) → winner got 7, loser got 6
            g_winner = int(parts[0].strip("()"))
            g_loser_str = parts[1].split("(")[0] if "(" in parts[1] else parts[1]
            g_loser = int(g_loser_str.strip("()"))

            # Convert from winner-loser to p1-p2
            if match_winner == 2:
                sets.append((g_loser, g_winner))
            else:
                sets.append((g_winner, g_loser))

        return sets
