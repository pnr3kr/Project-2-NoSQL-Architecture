# Predicting NHL Game Outcomes with NoSQL Data Architecture

This repository contains a fully constructed secondary dataset built using the document model in MongoDB Atlas, combining NHL game records, team statistics, goaltender performance, and team information data across four collections. The dataset is used to train and evaluate classification models that predict whether the home or away team wins an NHL game based on in-game performance metrics, including shots on goal, power play goals, and goaltender save percentage. A Random Forest classifier reached 84% cross-validated accuracy, but a feature audit showed that every model input is an in-game statistic unavailable before puck drop, making that figure outcome reconstruction rather than prediction. The pipeline includes data collection, MongoDB-based document storage, feature engineering, cross-validated model comparison, and publication-quality visualizations of results.

| Spec | Value |
|------|-------|
| Name | Tristen Davin |
| NetID | pnr3kr |
| DOI | [Link](https://doi.org/10.5281/zenodo.19865471)|
| Press Release | [New Data Analysis Reveals How NHL In-Game Statistics Can Predict Game Outcomes](press_release.md) |
| Pipeline | [pipeline.ipynb](pipeline.ipynb) |
| License | [MIT](LICENSE) |

---

## Headline

**135,604 documents** across 4 collections → **26,305 games** → **23 features** → Random Forest at **84.1%** cross-validated accuracy.

**That 84.1% is not a prediction result, and finding out why is the most useful thing in this project.** All 23 model inputs are *in-game* box-score statistics — shots, penalty minutes, power play goals, goaltender save percentage. None are knowable before puck drop, so the model is reconstructing an outcome it can already see. Published work on sports outcome prediction sits around [55.5%](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQClg0IVXCmBRLUynQQ_xonaATKUnxeLOciIal3AYMCjBrY?e=ULRzb1); a genuine 84% would be a major result, and that gap is what prompted the audit below.

---

## Pipeline

```
Kaggle NHL dataset (originally NHL Stats API)
        │  kagglehub
        ▼
   MongoDB Atlas ──── 4 collections, 135,604 documents
        │  merge on game_id
        ▼
   df_raw ──────────── 26,305 games joined to team, goalie, and franchise records
        │  drop score columns, IDs, high-null columns
        ▼
   df_clean ────────── 23 numeric features, target = home_win
        │  80/20 split, 5-fold cross-validation
        ▼
   4 models compared ─ Random Forest best at 84.1%
        │  feature audit
        ▼
   Finding ─────────── all 23 features are post-game; result is leakage, not prediction
```

| Stage | What it does |
|-------|--------------|
| Load | Pulls the Kaggle NHL dataset via `kagglehub` and loads four collections into MongoDB Atlas |
| Join | Merges `game`, `game_teams_stats`, `game_goalie_stats`, and `team_info` on `game_id`; selects the highest-time-on-ice goalie per game |
| Clean | Drops high-null columns, removes score and identifier fields, checks for duplicates |
| Model | Trains and cross-validates Logistic Regression, Random Forest, Gradient Boosting, and SVM |
| Audit | Reviews every surviving feature for availability at prediction time |

**Stack:** Python · MongoDB Atlas · pandas · scikit-learn · matplotlib/seaborn

---

## Results

| Model | CV accuracy | Std dev |
|-------|-------------|---------|
| Random Forest | **0.8412** | 0.0030 |
| Gradient Boosting | 0.8001 | 0.0119 |
| SVM | 0.7989 | 0.0058 |
| Logistic Regression | 0.6641 | 0.0079 |

Held-out test set: 84% accuracy, 0.85 precision / 0.86 recall on home wins across 4,596 games.

### Why these numbers overstate the result

Two rounds of feature review happened, and only the second was sufficient.

**Round one** removed the obvious leaks — `home_goals`, `away_goals`, and identifier columns. Accuracy fell from ~96% to 84%, which looked like the problem was solved.

**Round two** examined what remained:

```
home_shots, home_pim, home_powerPlayOpportunities, home_powerPlayGoals,
away_shots, away_pim, away_powerPlayOpportunities, away_powerPlayGoals,
goalie_timeOnIce, goalie_assists, goalie_goals, goalie_pim, goalie_shots,
goalie_saves, goalie_powerPlaySaves, goalie_shortHandedSaves, goalie_evenSaves,
goalie_shortHandedShotsAgainst, goalie_evenShotsAgainst,
goalie_powerPlayShotsAgainst, goalie_savePercentage,
goalie_powerPlaySavePercentage, goalie_evenStrengthSavePercentage
```

Every one is measured *during* the game being predicted. Two are decisive on their own:

- `home_powerPlayGoals` and `away_powerPlayGoals` are goals. The score columns were dropped, but a component of the score was not.
- `goalie_savePercentage` with `goalie_shots` recovers goals allowed exactly, since `goals = shots × (1 − save%)`.

The score is therefore fully reconstructable from the surviving features.

A train/test split does not catch this. A held-out set detects **overfitting** — memorizing training rows — but leakage is a property of the features themselves and is present identically in train and test. Every metric looks healthy while the model stays impossible to run in advance: at 6pm you don't know how many shots a team will take at 7.

The joins confirm it structurally. All three merges are `on="game_id"`, and the notebook contains no `rolling()`, `shift()`, `expanding()`, or `cumsum` — the operations needed to build features from *prior* games.

### What a valid version requires

Every feature must be knowable at puck drop, which means computing team and goaltender form as of the morning of the game:

- Rolling team averages over the previous N games (shots, goals for/against, power play conversion)
- Season-to-date goaltender save percentage for the expected starter
- Rest days and back-to-back flags
- Home/away and travel distance
- Head-to-head history

Mechanically that is `sort_values('date_time_GMT')` → `groupby('team_id')` → `rolling(N)` → `shift(1)`, so no row can see its own game. Realistic accuracy for that design is roughly 60% against a home-ice baseline near 55% — modest, but real.

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

### [Background Reading](https://myuva-my.sharepoint.com/:f:/g/personal/pnr3kr_virginia_edu/IgBQN6j6hN_3QqN3vuppqCptARAztXwGbxyYGeKlOzv76j8?e=v9evWx)

### Background Summary

| Title | Description | Link |
|-------|-------------|------|
| Predicting Sport Event Outcomes Using Deep Learning | Peer-reviewed paper presenting a hybrid CNN-Transformer model for predicting sports outcomes, outperforming traditional ML methods with 55.5% accuracy | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQClg0IVXCmBRLUynQQ_xonaATKUnxeLOciIal3AYMCjBrY?e=ULRzb1) |
| NHL Fantasy Picks, Props, Futures with EDGE Stats | NHL.com article covering current season player projections, advanced EDGE metrics, and futures predictions for awards and Stanley Cup | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQCeiyOvV8zDRKQPsVE3e_9SAdZb61IzMnLit33ZUDH-IQw?e=HbogFj) |
| Ice Hockey - NHL, Teams, Rules (Britannica) | Comprehensive overview of ice hockey history, NHL structure, rules of play, and key statistics and awards | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQDwX8rexeTZTZt57MuGJqKpARLO-XWZHjbCLUJOjiDTRmM?e=rdY0jT) |
| Historical Perspectives and Current Directions in Hockey Analytics | Academic review of hockey analytics research covering metrics like Corsi, expected goals, plus-minus, and player valuation | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQA7-larw12tR7AOUZkmNguvAWjTnuqcYyh7X_1TEOicnis?e=qCACM1) |
| A Brief History of Predicting Sports Outcomes | Overview of how sports prediction evolved from hunches and point spreads to Elo ratings, Moneyball, and modern ML models | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/pnr3kr_virginia_edu/IQCiX1sc1x00R61ey4oJ_vTRAcNLENy4WoxLJBm2G8uQjSA?e=aDEnII) |

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

### Rationale

The most important judgment call in this project was deciding to store data across four separate collections rather than embedding everything into a single document. While MongoDB's document model supports embedding, time constraints during the project made constructing fully embedded documents impractical, so separate collections were used as a time-efficient alternative. A second key decision involved which collections to include; only the collections most relevant to predicting game outcomes were loaded, while collections such as penalty records, player biographical data, and shift data were excluded. A third key decision was choosing win/loss as the prediction target rather than goal differential or exact score, which frames the problem as binary classification and simplifies modeling while still producing a practically meaningful output.

### Bias Identification

Several sources of bias may have been introduced during data collection. First, the dataset only covers games and events that were officially recorded in the NHL's system, meaning any data entry errors or missing records from certain seasons could skew results. Second, the dataset likely has recency bias, as older seasons may be underrepresented or recorded with less detail than more recent ones. Additionally, data collection practices have improved over time. Third, using goalie stats at the game level means that games where multiple goalies played may not fully reflect typical game conditions, since only the primary goalie's stats are retained in the final merged dataset.

### Bias Mitigation

Recency bias can be partially mitigated by documenting which seasons are included and limiting conclusions to that range rather than generalizing across all of NHL history. Missing or incomplete records were identified and filtered out during preprocessing by checking for null values across key fields. The multi-goalie issue was handled by selecting the goalie with the most time on ice per game, which is a standard approach in hockey analytics and ensures the most representative performance is captured.

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
