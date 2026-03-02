"""CSV file discovery and match-level row iteration for Sackmann's dataset.

Handles file discovery (glob), CSV reading, and basic row validation.
Yields one dict per match row — the parser handles the PBP expansion.
"""

import csv
from pathlib import Path
from typing import Dict, Iterator, List


class SackmannLoader:
    """Discovers and loads match rows from Jeff Sackmann's CSV files.

    Supports two CSV schemas:
    - Full (ATP, WTA, CH, FU, ITF): pbp_id, date, tny_name, tour, draw,
      server1, server2, winner, pbp, score, adf_flag, wh_minutes
    - Compact (f_*, i_*, t_*): date, tny_name, tour, draw, server1,
      server2, winner, pbp, score, adf_flag (no pbp_id, no wh_minutes)

    Example:
        loader = SackmannLoader(Path("data/raw/tennis_pointbypoint"))
        for match in loader.iter_matches():
            print(match["player_1"], "vs", match["player_2"])
    """

    GLOB_PATTERN = "pbp_matches_*.csv"

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)

    def discover_files(self) -> List[Path]:
        """Find all Sackmann CSV files in the data directory.

        Example:
            >>> loader = SackmannLoader(Path("data/raw/tennis_pointbypoint"))
            >>> loader.discover_files()
            [Path('.../pbp_matches_atp_main_archive.csv'),
             Path('.../pbp_matches_atp_main_current.csv'),
             Path('.../pbp_matches_atp_qual_archive.csv'),
             Path('.../pbp_matches_atp_qual_current.csv')]
        """
        files = sorted(self._data_dir.glob(self.GLOB_PATTERN))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.GLOB_PATTERN}' in {self._data_dir}"
            )
        return files

    def iter_matches(self) -> Iterator[Dict]:
        """Yield one dict per valid match row across all CSV files.

        Skips rows with empty pbp, missing player names, or invalid winner.
        Each dict contains: match_id, player_1, player_2, pbp_sequence,
        tournament, date, match_winner, expected_score, adf_flag, source_file.

        Example:
            >>> loader = SackmannLoader(Path("data/raw/tennis_pointbypoint"))
            >>> match = next(loader.iter_matches())
            >>> match
            {'match_id': '2231275_Olivier_Rochus_Fabio_Fognini',
             'player_1': 'Olivier Rochus',
             'player_2': 'Fabio Fognini',
             'pbp_sequence': 'SSSS;RRRR;SSRRSS;...',
             'tournament': 'ATPStudenaCroatiaOpen-ATPUmag2011',
             'date': '28 Jul 11',
             'match_winner': 2,
             'expected_score': '6-4 6-1',
             'adf_flag': 0,
             'source_file': 'pbp_matches_atp_main_archive.csv'}
        """
        for csv_file in self.discover_files():
            yield from self._iter_file(csv_file)

    def _iter_file(self, csv_file: Path) -> Iterator[Dict]:
        """Yield match dicts from a single CSV file."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                pbp_sequence = row.get("pbp", "").strip()
                player_1 = row.get("server1", "").strip()
                player_2 = row.get("server2", "").strip()
                winner = row.get("winner", "").strip()

                # Skip invalid rows
                if not pbp_sequence or not player_1 or not player_2:
                    continue
                if winner not in ("1", "2"):
                    continue

                pbp_id = row.get("pbp_id", "").strip()
                date = row.get("date", "").strip()
                # Files without pbp_id use date + tournament as fallback
                id_prefix = pbp_id if pbp_id else f"{date}_{row.get('tny_name', '')}".strip()
                match_id = (
                    f"{id_prefix}_{player_1}_{player_2}".replace(" ", "_")[:120]
                )

                yield {
                    "match_id": match_id,
                    "player_1": player_1,
                    "player_2": player_2,
                    "pbp_sequence": pbp_sequence,
                    "tournament": row.get("tny_name", "").strip(),
                    "date": date,
                    "match_winner": int(winner),
                    "expected_score": row.get("score", "").strip(),
                    "adf_flag": int(row.get("adf_flag", "0")),
                    "source_file": csv_file.name,
                }
