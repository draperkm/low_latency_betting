Refactor: Notebook Logic → OOP Preprocessing Module
Context
The notebook 01_raw_data_to_training_set.ipynb contains debugged, working logic for:

Discovering and loading Sackmann CSV files
Parsing compressed PBP strings into game state records
Validating parsed scores against expected scores
Building a training-ready DataFrame and saving to CSV
The existing domain/, data/, features/ folders contain earlier code with known bugs
(. delimiter ignored, no end-of-match finalization, no score validation) and over-engineering
(Pydantic models for 2.8M records, unused fields, complex schemas). They should be replaced
with a single preprocessing/ module that mirrors the notebook's proven logic.

The notebook is not modified — it remains the ground truth reference implementation.
The OOP module is an independent extraction with zero imports from the notebook.

What Changes
Delete
src/tennis_predictor/domain/ — replaced by preprocessing/models.py
src/tennis_predictor/data/ — replaced by preprocessing/loader.py, parser.py, validator.py
src/tennis_predictor/features/ — not used in training pipeline; rebuild when serving context is needed
Create: src/tennis_predictor/preprocessing/

src/tennis_predictor/
├── __init__.py
├── preprocessing/
│   ├── __init__.py          # Public API exports
│   ├── models.py            # GameState dataclass + tennis scoring constants
│   ├── parser.py            # MatchParser: pbp string → list of point dicts
│   ├── validator.py         # ScoreValidator: parsed vs expected score
│   ├── loader.py            # SackmannLoader: CSV discovery + row iteration
│   └── pipeline.py          # TrainingPipeline: orchestrate full flow + save CSVs
├── notebooks/
│   └── 01_raw_data_to_training_set.ipynb  (UNTOUCHED)
Plus entry point: scripts/build_training_set.py

File Responsibilities (SRP)
models.py — Tennis scoring data model (one responsibility: state representation)

GameState dataclass: sets, games, points, server, tiebreak, set_scores
TENNIS_POINTS mapping: {0: 0, 1: 15, 2: 30} (3+ → 40)
to_tennis_score(raw: int) → int helper
parser.py — PBP string parsing (one responsibility: sequence → point records)

MatchParser class
parse(match_id, player_1, player_2, pbp_sequence, ...) → Tuple[List[Dict], Optional[str]]
Walks PBP string char by char (S/R/A/D/;/.//)
Records state BEFORE each point (feature vector)
Stores points in tennis representation (0, 15, 30, 40)
Computes is_deuce, is_break_point from raw counts
Computes server_wins binary label
Handles . as set delimiter (not ignored!)
Handles end-of-match finalization (no trailing delimiter)
Calls ScoreValidator to compare parsed vs expected score
_end_game(state, last_point_winner) — game boundary logic
validator.py — Score validation (one responsibility: correctness check)

ScoreValidator class
validate(set_scores, expected_score, match_winner) → Optional[str]
Parses score string (winner-loser format)
Converts to p1-p2 perspective
Handles tiebreak notation 7-6(4)
Returns None on match, error string on mismatch
loader.py — CSV I/O (one responsibility: file discovery + row reading)

SackmannLoader class
__init__(data_dir: Path)
discover_files() → List[Path] — glob for pbp_matches_atp_*.csv
iter_matches() → Iterator[Dict] — yield one dict per CSV row
Handles encoding, skips empty pbp/players
pipeline.py — Orchestration + CSV output (one responsibility: wire components, run pipeline, save results)

TrainingPipeline class
__init__(data_dir: Path, output_dir: Path)
run() → pd.DataFrame — full pipeline:
Discover files via SackmannLoader
Parse each match via MatchParser (returns points + validation error)
Track stats (parsed ok, validation pass/fail, sample errors)
Build DataFrame from all collected points
Print summary statistics (target distribution, feature stats)
Save CSVs:
output_dir/atp_full_game_states.csv — all 21 columns (metadata + features + labels)
output_dir/atp_training_dataset.csv — 10 feature columns + server_wins target only
Print saved file paths and sizes
Return the full DataFrame
TRAINING_FEATURES: class constant listing the 10 pre-point-state feature column names
TARGET = "server_wins": class constant
scripts/build_training_set.py — Entry point

Parses --data-dir and --output-dir from CLI (with sensible defaults)
Instantiates TrainingPipeline and calls run()
Notebook: UNTOUCHED
01_raw_data_to_training_set.ipynb stays exactly as-is. It is the ground truth.
The OOP module is a standalone extraction — no circular dependencies.

Verification
Run python scripts/build_training_set.py — should produce:
atp_full_game_states.csv (2,820,681 rows × 21 columns)
atp_training_dataset.csv (2,820,681 rows × 11 columns: 10 features + target)
Validation pass rate should be ~86.6% (same as notebook)
Server win rate should be ~63.2%