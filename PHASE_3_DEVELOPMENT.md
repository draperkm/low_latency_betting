# Phase 3 Development Plan

---

## Known Data Quality Issues to Fix

### 1. MatchParser has no match-termination condition

**Discovered**: 2026-03-01, during notebook 04 validation.

**Symptom**: Some matches in `full_game_states.csv` show set counts that are
impossible in real tennis (e.g. `sets_p1=5, sets_p2=1` for match
`3159244_Marin_Cilic_Sam_Querrey`, 499 points).

**Root cause**: `MatchParser` processes every point in a PBP string without
a termination condition. If the source PBP string in Sackmann's CSV contains
extra data past the logical match end (or if the source row is malformed),
the parser keeps consuming points and incrementing set counts past the real
winner.

The match `3159244_Marin_Cilic_Sam_Querrey` was almost certainly a best-of-3
that ended **2-0 at row 146**. Every point after that is noise from the raw
source PBP string.

**Evidence** (set transitions from parsed data):
```
row   0 → 0-0 sets  (match start)
row  84 → 1-0 sets  (Cilic wins set 1)
row 146 → 2-0 sets  (Cilic wins set 2)   ← real match end (best-of-3)
row 228 → 2-1 sets  (Querrey wins set 3) ← impossible after match end
row 311 → 3-1 sets                       ← impossible
row 388 → 4-1 sets                       ← impossible
row 465 → 5-1 sets                       ← impossible
```

**Phase 3 fix**: Add a match-termination condition to `MatchParser`.
Before yielding each point, check whether either player has already
reached the winning set count:
- Best-of-3: stop when `max(sets_p1, sets_p2) >= 2`
- Best-of-5: stop when `max(sets_p1, sets_p2) >= 3`

The match format (BO3 vs BO5) is available in the Sackmann match-level CSV
(`best_of` column). It should be passed into `MatchParser` at construction
so the termination threshold is known per match.

**Scale**: The current Phase 2 workaround is to filter affected matches in
`_build_catalogue()` using the all-zero reset detection. This catches a
different artifact (split PBP rows) but does NOT catch this over-run bug.
A Phase 3 fix in the parser would make the processed data correct by
construction, eliminating the need for downstream filtering.
