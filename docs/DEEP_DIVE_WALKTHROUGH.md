-> what is an ML system: what we know, and what we want to predict?

-> domain driven design considering ML best practices: decide the folder structure

-> nfl vs tennis: the aim of the project is to demonstrate a live odds update on a chart

-> understand the complexity of the dataset

-> first significant step: understanding and preparing the data: Sackmann data set (SRSSR). PASSING FROM RAW DATA -> TRAINING DATA (that's all the difficulty. Risk Analysts are the ones who take care of that) -> SOLUTION: parse raw data to create game states


# How Tennis Works (And How `parse_point_sequence` Interprets It)

## Tennis Structure (Top to Bottom)

```
MATCH (best-of-3 sets)
├── SET 1 (first to 6 games with 2-game lead)
│   ├── GAME 1 (first to 4 points)
│   │   ├── POINT 1
│   │   ├── POINT 2
│   │   ├── POINT 3
│   │   └── POINT 4 (game ends)
│   ├── GAME 2
│   └── ... (6+ games)
├── SET 2
└── SET 3
```

---

## Points Within a Game

In tennis, points are called by these scores:

```
Points won → Tennis Score
────────────────────────
0           → "0"
1           → "15"
2           → "30"
3           → "40"
4 (if ahead) → "Game"
```

**Special case: Deuce & Advantage**
- If both players reach 3 points (40-40) → **Deuce**
- Next point: **Advantage** to the leader
- If advantage player wins next → **Game**
- If other player wins → Back to **Deuce**

---

## The Encoded Format (What Jeff Sackmann Uses)

Each character represents **one point**:

```
Character → What Happened
────────────────────────────
'S'       → Server won the point
'R'       → Returner won the point
'A'       → Ace (server won, no rally)
'D'       → Double fault (returner won)
';'       → Game just ended (reset points 0-0)
'.'       → Set just ended (reset games 0-0)
'/'       → Tiebreak server switch
```

**Example:**
```
"S.R.S.S.R.R."

S         → Server wins point (0-0 → 1-0)
.         → Game ends (Server won 4-0)
R         → Returner wins point in new game (0-0 → 0-1)
.         → Game ends (Returner won 4-0)
... etc
```

---

## How `parse_point_sequence` Implements This

### 1. **Initialize Scorekeeper**

```python
score = TennisScore()
# serving_player = 1 (Player 1 serves first)
# points_p1 = 0, points_p2 = 0
# games_p1 = 0, games_p2 = 0
# sets_p1 = 0, sets_p2 = 0
```

### 2. **Loop Through Each Character**

```python
for char in pbp_sequence:
    if char == 'S':
        # Server won this point
        point_winner = score.serving_player  # (1 or 2)
    
    elif char == 'R':
        # Returner won this point
        point_winner = 3 - score.serving_player  # Toggle: 1→2, 2→1
    
    elif char == ';':
        # GAME ENDED
        # Determine who won the game (who had more points)
        if score.points_p1 > score.points_p2:
            score.games_p1 += 1
        else:
            score.games_p2 += 1
        
        # Reset for new game
        score.points_p1 = 0
        score.points_p2 = 0
        score.serving_player = 3 - score.serving_player  # Switch server
        
        # Check if SET is won (6 games with 2-game lead)
        if score.games_p1 >= 6 and score.games_p1 - score.games_p2 >= 2:
            score.sets_p1 += 1  # P1 won the set
            # Reset for new set
            score.games_p1 = 0
            score.games_p2 = 0
            score.set_num += 1
```

### 3. **Record the Point (BEFORE Updating Score)**

```python
point_record = {
    "set_num": score.set_num,          # Which set?
    "game_num": score.game_num,        # Which game in the set?
    "serving_player": score.serving_player,  # Who serves?
    "points_player_1": score.points_p1,      # Score BEFORE this point
    "points_player_2": score.points_p2,
    "sets_player_1": score.sets_p1,
    "sets_player_2": score.sets_p2,
    "games_player_1": score.games_p1,
    "games_player_2": score.games_p2,
    "point_winner": point_winner,      # WHO WON THIS POINT? (target label)
}
points.append(point_record)
```

### 4. **Update Score (AFTER Recording)**

```python
if point_winner == 1:
    score.points_p1 += 1
else:
    score.points_p2 += 1
```

---

## Step-by-Step Example: "S.RS.R."

```
Initial: TennisScore()
serving_player = 1, points_p1=0, points_p2=0, games_p1=0, games_p2=0

═══════════════════════════════════════════════════════════════

CHAR 'S' (Server wins)
  point_winner = 1 (serving_player)
  RECORD: {serving_player:1, points:0-0, winner:1}
  UPDATE: points_p1 = 1  ← Now it's 1-0

CHAR '.' (Game ended)
  games_p1 = 1 (Player 1 won, had 1 point vs 0)
  Reset: points_p1=0, points_p2=0
  Switch: serving_player = 2
  
═══════════════════════════════════════════════════════════════

CHAR 'R' (Returner wins, now Player 2 serves)
  point_winner = 3 - 2 = 1  (NOT serving_player)
  RECORD: {serving_player:2, points:0-0, winner:1}
  UPDATE: points_p1 = 1  ← Now it's 1-0 in game 2

CHAR 'S' (Server wins)
  point_winner = 2 (serving_player)
  RECORD: {serving_player:2, points:1-0, winner:2}
  UPDATE: points_p2 = 1  ← Now it's 1-1

CHAR '.' (Game ended)
  games_p2 = 1 (Player 2 won, had 1 point vs 1... WAIT, TIED!)
  
  ⚠️ ISSUE: Both had 1 point! This shouldn't end the game!
```

**This reveals a bug in the decoder** — the function assumes whoever has MORE points won, but that fails on tied games.

---

## ASCII Diagram: Full Match State Progression

```
┌─ MATCH START
│
├─ SET 1, GAME 1, Player 1 Serving
│  ├─ Point 1: S (1-0)
│  ├─ Point 2: R (1-1)
│  ├─ Point 3: S (2-1)
│  ├─ Point 4: S (3-1)
│  └─ Point 5: ; (Game ends, P1 won 4-1)
│
├─ SET 1, GAME 2, Player 2 Serving
│  ├─ Point 1: R (0-1)
│  └─ Point 2: . (Set ends)
│
└─ SET 2 (if match continues)
```

---

## Key Insight

The `parse_point_sequence` function:
1. **Expands** the compressed format into individual points
2. **Tracks state** (who serves, current score, games, sets)
3. **Records context** for each point (not just the winner, but the match state)
4. **Creates training data** where the model learns: "Given this match state, who won this point?"

This is perfect for ML because each row has:
- **Input**: `[serving_player, points, games, sets, surface, tournament]`
- **Output**: `[point_winner]` ← Label to predict


-> the use of notebooks for testing and debugging

-> montecarlo is the real interview value

-> PHILOSOPHY: this is just the first iteration. This is AGILE. Get a first working product and then improve it. By self-containing each step of the pipeline, we can improve each module indipendently, and teams can work in parallel 