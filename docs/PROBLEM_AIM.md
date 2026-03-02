## How to Predict a Point-by-Point Game
## does the training yeal a model that predict the probability of the server to win the current point?

## Dataset

The primary dataset for this project is **Jeff Sackmann's tennis_pointbypoint** repository on GitHub, which contains sequential point-by-point data for tens of thousands of professional tennis matches.

## Data Preparation: From Sequence to Game State

The raw Sackmann data records points as a sequence of outcomes within each match. Our preparation step transforms this sequential format into a **supervised learning dataset** where:

- **Each row represents a game state** at the moment a point is about to be played — encoding the current score (sets, games, points), who is serving, and any other contextual features available (e.g., player rankings, surface, head-to-head record).
- **The target variable (y)** is the **player who won that point** — framed as a binary classification (e.g., 1 if the server won the point, 0 otherwise).

By reconstructing the cumulative game state at every point in a match, we create a tabular dataset suitable for tree-based models. The agent must walk through the point sequence chronologically, maintaining a running scoreboard and emitting one row per point with the full state snapshot before the point was played.

## Modelling Approach

We train an **XGBoost classifier** to predict the point winner given a game state. The model learns which configurations of score, serve, momentum, and player characteristics are most predictive of the next point outcome.

### Feature Categories

| Category | Examples |
|----------|----------|
| **Score state** | Current set score, game score, point score, tiebreak indicator |
| **Serve context** | Who is serving, first or second serve |
| **Player attributes** | Rankings, ranking difference, head-to-head record |
| **Match context** | Surface type, tournament level, round |
| **Momentum proxies** | Points won in last N points, break point indicator, games won streak |

### Why XGBoost on Game States

Framing each point as an independent classification over a state vector lets us avoid the complexity of sequential or Bayesian recursive models. XGBoost naturally handles the non-linear interactions between features (e.g., serving at break point down a set behaves differently from serving at 40-0 up a set) and provides feature importance scores that reveal which game state dimensions matter most.