# DS 4320 Project 2: Predicting NHL Game Outcomes with NoSQL Data Architecture

This project implements a data pipeline to process NHL game data, clean it, and train a machine learning model to predict game outcomes. The pipeline is designed to handle the complexities of real-world data, including duplicates, null values, and irrelevant features. The final model's performance is evaluated against a baseline to demonstrate the value of the data processing steps.

| Spec | Value |
|------|-------|
| Name | Tristen Davin |
| NetID | pnr3kr |
| DOI | [Link](YOUR_ZENODO_DOI) |
| Press Release | [New Data Analysis Reveals How NHL In-Game Statistics Can Predict Game Outcomes](press_release.md) |
| Pipeline | [pipeline.ipynb](pipeline.ipynb) |
| License | [MIT](LICENSE) |

---

## Problem Definition

### General and Specific Problem

- **Initial General Problem**
Predicting sports game outcomes is a longstanding challenge in sports analytics, as the result of any given game depends on a complex combination of team performance, strategy, and variance.
- **Specific Problem**
Using NHL game-by-game statistics stored across five MongoDB collections, this project builds a binary classification model to predict whether the home or away team wins, using in-game performance metrics including shots on goal, power play goals, penalties, and goaltender save percentage.

### Rationale

The general problem was refined in three specific ways. First, the sport was narrowed to the NHL because hockey is a particularly high-variance, low-scoring sport where individual goaltender performance and special teams efficiency play an outsized role in determining outcomes, making it a compelling and nuanced classification problem. Second, the prediction target was fixed to binary win/loss rather than goal differential or exact score, because win/loss is the most practically meaningful outcome for analysts, fans, and betting markets, and framing it as binary classification makes the problem tractable for standard ML models. Third, the data source was fixed to a Kaggle NHL dataset loaded into MongoDB Atlas, because it provides consistent game-level and event-level statistics across multiple seasons in a format well-suited to the document model.

### Motivation

The NHL generates a wealth of detailed game-by-game statistics that make it a compelling domain for predictive modeling. Accurate game outcome prediction has real value for broadcasters, fantasy hockey platforms, sports analysts, and fans alike. For teams and coaching staffs, understanding which on-ice metrics most strongly predict winning can inform line combinations, power play strategy, and goaltender selection. Hockey is also a particularly interesting sport for prediction because of its relatively low-scoring nature — a single goal or goaltender performance can swing an outcome — making it a challenging and nuanced classification problem that goes beyond simple offensive statistics.

### Press Release

[New Data Analysis Reveals How NHL In-Game Statistics Can Predict Game Outcomes](press_release.md)

---

## Domain Exposition

### Terminology

| Term | Meaning | Why It Matters |
|------|---------|----------------|
| Save Percentage | The proportion of shots on goal that a goaltender stops | One of the strongest individual performance indicators for predicting game outcomes |
| Corsi For % | The percentage of total shot attempts taken by a team while on ice | A possession metric that reflects puck control and offensive pressure |
| Power Play | A situation where one team has more skaters due to an opponent's penalty | Power play efficiency is a key predictor of scoring and game outcomes |
| Shots on Goal | The number of shots directed at the opposing net that required a save or scored | Reflects offensive pressure and is one of the top features in the model |
| Penalty Kill | A team's ability to defend while shorthanded due to a penalty | Strong penalty killing reduces opponent power play opportunities |
| Home Ice Advantage | The tendency for home teams to win more often than away teams | The dataset shows ~55% home win rate, confirming this effect in the NHL |
| Regular Season | The 82-game schedule each NHL team plays before the playoffs | The dataset covers regular season games only |
| Goaltender Decision | Whether the goalie received a win or loss for their performance | Used as a feature to capture goaltender contribution |

### Domain Background

This project exists within the domain of sports analytics, specifically NHL hockey performance analysis. Hockey is unique among major North American sports due to its low-scoring nature, fast pace, and the outsized influence of goaltender performance on game outcomes. A single outstanding goaltending performance can overcome an otherwise dominant opponent, making prediction more nuanced than in higher-scoring sports like basketball or football. The field of hockey analytics has matured significantly over the past decade, moving beyond traditional box score statistics toward advanced possession metrics like Corsi and expected goals. This project sits at the intersection of sports analytics and machine learning, using team-level per-game statistics stored in a MongoDB document database to model and predict game outcomes across multiple NHL seasons.

### [Background Reading](background/)

### Background Summary

| Title | Brief Description | Link |
|-------|-------------------|------|
| Predicting Sport Event Outcomes Using Deep Learning | Peer-reviewed paper presenting a hybrid CNN-Transformer model for predicting sports outcomes, outperforming traditional ML methods | [background/01_deep_learning_sports.pdf](background/01_deep_learning_sports.pdf) |
| NHL Fantasy Picks, Props, Futures with EDGE Stats | NHL.com article covering current season player projections and advanced EDGE metrics | [background/02_nhl_edge_stats.pdf](background/02_nhl_edge_stats.pdf) |
| Ice Hockey - NHL, Teams, Rules (Britannica) | Comprehensive overview of ice hockey history, NHL structure, rules of play, and key statistics | [background/03_britannica_hockey.pdf](background/03_britannica_hockey.pdf) |
| Historical Perspectives and Current Directions in Hockey Analytics | Academic review of hockey analytics research covering metrics like Corsi, expected goals, and player valuation | [background/04_hockey_analytics_review.pdf](background/04_hockey_analytics_review.pdf) |
| A Brief History of Predicting Sports Outcomes | Overview of how sports prediction evolved from hunches and point spreads to Elo ratings and modern ML models | [background/05_history_sports_prediction.pdf](background/05_history_sports_prediction.pdf) |

---

## Data Creation

### Provenance

The raw data for this project was sourced from a publicly available NHL game dataset on Kaggle, accessed via the kagglehub Python library. The dataset was originally compiled from the NHL's official Stats API and contains detailed game-by-game records across multiple NHL regular seasons. Five collections were selected for loading into MongoDB Atlas: `game` (core game records including outcome, teams, score, and venue), `game_goalie_stats` (goaltender performance per game), `game_teams_stats` (team-level statistics per game including shots, power play goals, and penalties), and `team_info` (franchise and team metadata). These four collections were chosen because they contain the performance metrics most directly relevant to predicting game outcomes. Additional collections available in the dataset such as player biographical information and shift data were excluded due to storage constraints on the Atlas free tier.

### Code

| File | Collection | Description | Link |
|------|------------|-------------|------|
| `pipeline.ipynb` | `game` | Core game records including outcome, teams, and season (26,305 docs) | [pipeline.ipynb](pipeline.ipynb) |
| `pipeline.ipynb` | `game_goalie_stats` | Goaltender stats per game (56,656 docs) | [pipeline.ipynb](pipeline.ipynb) |
| `pipeline.ipynb` | `game_teams_stats` | Team stats per game (52,610 docs) | [pipeline.ipynb](pipeline.ipynb) |
| `pipeline.ipynb` | `team_info` | Team and franchise metadata (33 docs) | [pipeline.ipynb](pipeline.ipynb) |

### Bias Identification

Several sources of bias may have been introduced during data collection. First, the dataset only covers games and events that were officially recorded in the NHL's system, meaning any data entry errors or missing records from certain seasons could skew results. Second, the dataset likely has temporal bias — older seasons may be underrepresented or recorded with less detail than more recent ones, as data collection practices have improved over time. Third, using goalie stats at the game level means that games where multiple goalies played may not fully reflect typical game conditions, since only the primary goalie's stats are retained in the final merged dataset.

### Bias Mitigation

Temporal bias can be partially mitigated by documenting which seasons are included and limiting conclusions to that range rather than generalizing across all of NHL history. Missing or incomplete records were identified and filtered out during preprocessing by checking for null values across key fields. The multi-goalie issue was handled by selecting the goalie with the most time on ice per game, which is a standard approach in hockey analytics and ensures the most representative performance is captured.

### Rationale for Critical Decisions

The most important judgment call in this project was deciding to store data across five separate collections rather than embedding everything into a single document. While the MongoDB document model supports embedding related data directly inside each record, embedding all team stats, goalie stats, and goal events into each game document would have created extremely large and unwieldy documents. Storing them as separate collections linked by `game_id` keeps individual documents manageable while preserving the ability to join at query time. A second key decision involved which collections to include — due to storage constraints on the Atlas free tier, only collections most directly relevant to predicting game outcomes were loaded. A third key decision was choosing win/loss as the prediction target rather than goal differential or exact score, which frames the problem as binary classification and simplifies modeling while still producing a practically meaningful output.

---

## Metadata

### Implicit Schema Guidelines

Based on the documents observed across the four collections, the following schema guidelines should be followed. In the `game` collection, every document should contain `game_id`, `season`, `type`, `date_time_GMT`, `away_team_id`, `home_team_id`, `away_goals`, `home_goals`, `outcome`, and `venue` as required fields, where `game_id` is always stored as an integer and serves as the join key across all other collections. In `game_teams_stats`, every document should contain `game_id`, `team_id`, `HoA`, `won`, `goals`, `shots`, and `powerPlayGoals`, with `HoA` always stored as either `"home"` or `"away"` and all numeric fields stored as numbers rather than strings. In `game_goalie_stats`, every document should contain `game_id`, `player_id`, `team_id`, `decision`, `saves`, `shots`, and `savePercentage`, with all numeric fields stored as numbers to ensure aggregation pipelines work correctly. Finally, in `team_info`, every document should contain `team_id`, `teamName`, `abbreviation`, and `franchiseId` to allow filtering and joining by team.

### Data Summary

| Collection | Documents | Description |
|------------|-----------|-------------|
| `game` | 26,305 | One record per game with outcome, teams, score, and venue |
| `game_goalie_stats` | 56,656 | One record per goalie per game with saves, shots, and save percentage |
| `game_teams_stats` | 52,610 | One record per team per game with goals, shots, and power play stats |
| `team_info` | 33 | One record per NHL franchise with team name and abbreviation |
| **Total** | **135,604** | — |

### Data Dictionary

| Collection | Field | Data Type | Description | Example |
|------------|-------|-----------|-------------|---------|
| game | game_id | int | Unique game identifier, join key across collections | 2016020906 |
| game | season | int | Season in XXXXYYYY format (start year + end year) | 20162017 |
| game | type | string | Game type: R (regular), P (playoffs) | "R" |
| game | date_time_GMT | string | Game start time in GMT | "2017-02-25T22:00:00Z" |
| game | away_team_id | int | Away team identifier | 15 |
| game | home_team_id | int | Home team identifier | 18 |
| game | away_goals | int | Total goals scored by away team | 2 |
| game | home_goals | int | Total goals scored by home team | 5 |
| game | outcome | string | Game result and method | "home win REG" |
| game | venue | string | Arena name | "Bridgestone Arena" |
| game | venue_time_zone_offset | int | UTC offset of venue timezone | -5 |
| game_goals | play_id | string | Unique play identifier (game_id + play number) | "2015020314_272" |
| game_goals | strength | string | Strength state when goal was scored | "Even" |
| game_goals | gameWinningGoal | boolean | Whether this goal was the game winner | False |
| game_goals | emptyNet | boolean | Whether goal was scored on an empty net | True |
| game_goalie_stats | player_id | int | Unique goalie player identifier | 8471734 |
| game_goalie_stats | team_id | int | Team the goalie played for | 26 |
| game_goalie_stats | timeOnIce | int | Seconds played in the game | 3600 |
| game_goalie_stats | saves | int | Total saves made | 25 |
| game_goalie_stats | shots | int | Total shots faced | 26 |
| game_goalie_stats | decision | string | W (win), L (loss), or null if no decision | "W" |
| game_goalie_stats | savePercentage | float | Saves divided by shots faced × 100 | 96.15 |
| game_goalie_stats | powerPlaySavePercentage | float | Save percentage on power play shots | 75.0 |
| game_goalie_stats | evenStrengthSavePercentage | float | Save percentage on even strength shots | 100.0 |

### Uncertainty Quantification

| Collection | Field | Min | Max | Mean | Std Dev | Missing/NaN |
|------------|-------|-----|-----|------|---------|-------------|
| `game` | `away_goals` | 0 | 11 | 2.69 | 1.62 | 0 |
| `game` | `home_goals` | 0 | 12 | 2.96 | 1.69 | 0 |
| `game_goalie_stats` | `saves` | 0 | 85 | 25.20 | 8.65 | 0 |
| `game_goalie_stats` | `shots` | 0 | 88 | 27.68 | 8.83 | 0 |
| `game_goalie_stats` | `timeOnIce` | 0 | 9027 | 3369.15 | 734.19 | 0 |
| `game_goalie_stats` | `savePercentage` | 0.00 | 100.00 | 90.11 | 7.80 | 139 |
| `game_goalie_stats` | `powerPlaySavePercentage` | 0.00 | 100.00 | 84.26 | 22.67 | 4,743 |