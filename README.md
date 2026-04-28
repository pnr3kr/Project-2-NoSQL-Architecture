# DS 4320 Project 2: Predicting NHL Game Outcomes with NoSQL Data Architecture

This repository contains a fully constructed secondary dataset built using the document model in MongoDB Atlas, combining NHL game records, team statistics, goaltender performance, and team information data across four collections. The dataset is used to train and evaluate classification models that predict whether the home or away team wins an NHL game based on in-game performance metrics, including shots on goal, power play goals, and goaltender save percentage. A Random Forest classifier achieved 84% cross-validated accuracy after removing data leakage. The pipeline includes data collection, MongoDB-based document storage, feature engineering, cross-validated model comparison, and publication-quality visualizations of results.

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

- **Initial General Problem**:
Predicting sports game outcomes.

- **Specific Problem**:
Predicting the outcome (win/loss) of NHL games using a machine learning model based on in-game team performance statistics, including shots on goal, power play goals, goaltender save percentage, and penalty minutes, drawn from game records stored across four MongoDB collections.

### Motivation

The NHL generates a massive amount of detailed game-by-game statistics, making it a compelling domain for predictive modeling. Accurate game outcome prediction has real value for broadcasters, fantasy hockey platforms, sports analysts, and fans. For teams and coaching staffs, understanding which on-ice metrics most strongly predict winning can inform line combinations, power play strategy, and goaltender selection. Hockey is also a particularly interesting sport for prediction because of its low-scoring nature, with a single goal or goaltender performance swinging an outcome. This makes it a challenging classification problem that goes beyond simple offensive statistics.

### Rationale

Predicting sports outcomes in general is too vague to build a focused data pipeline around. Narrowing to the NHL allows us to identify a specific, consistent data source with standardized statistics across seasons. Focusing on team-level metrics rather than individual player stats keeps the problem manageable while still capturing the key factors that determine game outcomes. Choosing win/loss as the prediction target rather than exact score or goal differential frames this as a binary classification problem, which is well-suited for an initial modeling pipeline using a document database to store and query game records.

### Press Release

[New Data Analysis Reveals How NHL In-Game Statistics Can Predict Game Outcomes](press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|------------|
| Win/Loss (W/L) | The outcome of a game, used as the binary prediction target |
| Goals For (GF) | Total goals scored by a team in a game |
| Goals Against (GA) | Total goals scored against a team in a game |
| Shots on Goal (SOG) | Number of shots directed on net that would have scored if not saved |
| Save Percentage (SV%) | Proportion of shots on goal stopped by the goaltender |
| Power Play (PP) | A situation where one team has a numerical advantage due to opponent penalties |
| Power Play Percentage (PP%) | Rate at which a team scores during power play opportunities |
| Penalty Kill Percentage (PK%) | Rate at which a team prevents goals when shorthanded |
| Corsi For % (CF%) | Shot attempt differential, a measure of puck possession |
| Expected Goals (xG) | A model-based estimate of goal probability based on shot quality |
| Home/Away | Whether a team is playing at their home arena or on the road |
| Overtime (OT) | Extra period played when score is tied after regulation |

### Domain Background

The NHL is a professional ice hockey league consisting of 32 teams across the United States and Canada. Each team plays 82 regular-season games, generating an extensive dataset of per-game statistics. Hockey is unique among major sports due to its low-scoring nature, fast pace, and the outsized influence of goaltender performance, making prediction more complex than in higher-scoring sports like basketball or football.

### [Background Reading](background/)

### Background Summary

| Title | Description | Link |
|-------|-------------|------|
| Predicting Sport Event Outcomes Using Deep Learning | Peer-reviewed paper presenting a hybrid CNN-Transformer model for predicting sports outcomes, outperforming traditional ML methods with 55.5% accuracy | [Link](background/Predicting-Sport-Event-Outcomes-Using-Deep-Learning.pdf) |
| NHL Fantasy Picks, Props, Futures with EDGE Stats | NHL.com article covering current season player projections, advanced EDGE metrics, and futures predictions for awards and Stanley Cup | [Link](background/NHL-Fantasy-EDGE-stats.pdf) |
| Ice Hockey - NHL, Teams, Rules (Britannica) | Comprehensive overview of ice hockey history, NHL structure, rules of play, and key statistics and awards | [Link](background/Ice-hockey-NHL-Teams-Rules-Britannica.pdf) |
| Historical Perspectives and Current Directions in Hockey Analytics | Academic review of hockey analytics research covering metrics like Corsi, expected goals, plus-minus, and player valuation | [Link](background/Historical-Perspectives.pdf) |
| A Brief History of Predicting Sports Outcomes | Overview of how sports prediction evolved from hunches and point spreads to Elo ratings, Moneyball, and modern ML models | [Link](background/A-Brief-History-of-Predicting-Sports-Outcomes.pdf) |

---

## Data Creation

### Provenance

The raw data for this project was sourced from a publicly available NHL game dataset on Kaggle, accessed via the kagglehub Python library. The dataset was originally compiled from the NHL's official Stats API and contains detailed game-by-game records across multiple NHL regular seasons. Four collections were selected for loading into MongoDB Atlas: `game` (core game records including outcome, teams, score, and venue), `game_goalie_stats` (goaltender performance per game), `game_teams_stats` (team-level statistics per game including shots, power play goals, and penalties), and `team_info` (franchise and team metadata). These four collections were chosen because they contain the performance metrics most directly relevant to predicting game outcomes. Additional collections available in the dataset, such as player biographical information, shift data, and penalty records, were excluded due to storage constraints on the Atlas free tier.

### Code

| File | Collection | Description | Link |
|------|------------|-------------|------|
| `pipeline.ipynb` | `game` | Core game records including outcome, teams, and season (26,305 docs) | [Code](pipeline.ipynb) |
| `pipeline.ipynb` | `game_goalie_stats` | Goaltender stats per game (56,656 docs) | [Code](pipeline.ipynb) |
| `pipeline.ipynb` | `game_teams_stats` | Team stats per game, including shots, power play goals, and penalties (52,610 docs) | [Code](pipeline.ipynb) |
| `pipeline.ipynb` | `team_info` | Team and franchise metadata (33 docs) | [Code](pipeline.ipynb) |

### Bias Identification

Several sources of bias may have been introduced during data collection. First, the dataset only covers games and events that were officially recorded in the NHL's system, meaning any data entry errors or missing records from certain seasons could skew results. Second, the dataset likely has recency bias, as older seasons may be underrepresented or recorded with less detail than more recent ones. Additionally, data collection practices have improved over time. Third, using goalie stats at the game level means that games where multiple goalies played may not fully reflect typical game conditions, since only the primary goalie's stats are retained in the final merged dataset.

### Bias Mitigation

Recency bias can be partially mitigated by documenting which seasons are included and limiting conclusions to that range rather than generalizing across all of NHL history. Missing or incomplete records were identified and filtered out during preprocessing by checking for null values across key fields. The multi-goalie issue was handled by selecting the goalie with the most time on ice per game, which is a standard approach in hockey analytics and ensures the most representative performance is captured.

### Rationale

The most important judgment call in this project was deciding to store data across four separate collections rather than embedding everything into a single document. While MongoDB's document model supports embedding, time constraints during the project made constructing fully embedded documents impractical, so separate collections were used as a time-efficient alternative. A second key decision involved which collections to include; only the collections most relevant to predicting game outcomes were loaded, while collections such as penalty records, player biographical data, and shift data were excluded. A third key decision was choosing win/loss as the prediction target rather than goal differential or exact score, which frames the problem as binary classification and simplifies modeling while still producing a practically meaningful output.

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
| game_goalie_stats | player_id | int | Unique goalie player identifier | 8471734 |
| game_goalie_stats | team_id | int | Team the goalie played for | 26 |
| game_goalie_stats | timeOnIce | int | Seconds played in the game | 3600 |
| game_goalie_stats | saves | int | Total saves made | 25 |
| game_goalie_stats | shots | int | Total shots faced | 26 |
| game_goalie_stats | decision | string | W (win), L (loss), or null if no decision | "W" |
| game_goalie_stats | savePercentage | float | Saves divided by shots faced × 100 | 96.15 |
| game_goalie_stats | powerPlaySavePercentage | float | Save percentage on power play shots | 75.0 |
| game_goalie_stats | evenStrengthSavePercentage | float | Save percentage on even strength shots | 100.0 |
| game_teams_stats | game_id | int | Unique game identifier, join key | 2016020906 |
| game_teams_stats | team_id | int | Team identifier | 18 |
| game_teams_stats | HoA | string | Whether team is home or away | "home" |
| game_teams_stats | won | boolean | Whether this team won the game | True |
| game_teams_stats | goals | float | Total goals scored by the team | 5.0 |
| game_teams_stats | shots | float | Total shots on goal by the team | 32.0 |
| game_teams_stats | powerPlayGoals | float | Goals scored on the power play | 1.0 |
| game_teams_stats | powerPlayOpportunities | float | Number of power play chances | 3.0 |
| game_teams_stats | faceOffWinPercentage | float | Percentage of faceoffs won | 52.3 |
| game_teams_stats | pim | float | Penalty minutes accumulated | 8.0 |
| team_info | team_id | int | Unique team identifier | 18 |
| team_info | teamName | string | Full team name | "Predators" |
| team_info | shortName | string | City or region name | "Nashville" |
| team_info | abbreviation | string | Three letter team abbreviation | "NSH" |
| team_info | franchiseId | int | Unique franchise identifier | 34 |

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
| `game_teams_stats` | `goals` | 0 | 12 | 2.76 | 1.65 | 8 |
| `game_teams_stats` | `shots` | 0 | 88 | 29.77 | 6.88 | 8 |
| `game_teams_stats` | `powerPlayGoals` | 0 | 7 | 0.68 | 0.82 | 8 |
| `game_teams_stats` | `powerPlayOpportunities` | 0 | 16 | 3.77 | 1.89 | 8 |
| `game_teams_stats` | `pim` | 0 | 213 | 12.11 | 9.21 | 8 |
| `game_teams_stats` | `faceOffWinPercentage` | 0 | 79.20 | 49.96 | 7.34 | 22,148 |
