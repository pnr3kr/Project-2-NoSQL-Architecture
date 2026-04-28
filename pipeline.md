# NHL Game Outcome Prediction Pipeline


```python
import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
```


```python
# Configure logging to write to pipeline.log
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
uri = os.getenv("MONGO_URI")

try:
    # Connect to MongoDB Atlas using stable API version
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    db = client["nhl"]
    logging.info("Successfully connected to MongoDB Atlas")
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    logging.error(f"Connection failed: {e}")
    raise
```

    Successfully connected to MongoDB Atlas!



```python
# Load CSVs into Atlas
csv_folder = "data/"

# Map CSV filenames to MongoDB collections
csv_files = {
    "game.csv":              "game",
    "game_goalie_stats.csv": "game_goalie_stats",
    "game_teams_stats.csv":  "game_teams_stats",
    "team_info.csv":         "team_info",
}

for filename, collection_name in csv_files.items():
    try:
        filepath = os.path.join(csv_folder, filename)
        
        # Skip files that don't exist locally
        if not os.path.exists(filepath):
            logging.warning(f"File not found, skipping: {filename}")
            print(f"Skipping {filename} - not found")
            continue

        # Load CSV and replace NaN with None for MongoDB compatibility
        df = pd.read_csv(filepath)
        records = df.where(pd.notnull(df), None).to_dict("records")

        # Insert records in chunks of 500 to avoid memory issues
        for i in range(0, len(records), 500):
            db[collection_name].insert_many(records[i:i+500])

        logging.info(f"Inserted {len(records)} records into '{collection_name}'")
        print(f"Inserted {len(records)} records into '{collection_name}'")

    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        print(f"Error loading {filename}: {e}")

print("Done!")
logging.info("Data loading complete")
```

## Data Preparation

Each collection is queried separately from MongoDB using `.find()` and merged 
in pandas using `game_id` as the join key. The original approach used MongoDB's 
`$lookup` aggregation to embed related documents directly into each game record, 
demonstrating the document model's embedding flexibility. However, this timed out 
on the Atlas free tier due to the size of the collections. Pandas merging was used 
as a practical adaptation while preserving the same logical structure.


```python
try:
    print("Querying collections from MongoDB...")
    logging.info("Starting data preparation - querying each collection separately")

    df_game = pd.DataFrame(list(db["game"].find()))
    print(f"Loaded {len(df_game)} games")

    df_goalie = pd.DataFrame(list(db["game_goalie_stats"].find()))
    print(f"Loaded {len(df_goalie)} goalie records")

    df_teams = pd.DataFrame(list(db["game_teams_stats"].find()))
    print(f"Loaded {len(df_teams)} team stat records")

    df_team_info = pd.DataFrame(list(db["team_info"].find()))
    print(f"Loaded {len(df_team_info)} team info records")

    # Deduplicate team stats keeping first occurrence of each game_id + HoA
    df_teams = df_teams.drop_duplicates(subset=["game_id", "HoA"])
    print(f"After deduplication: {len(df_teams)} team stat rows")

    # Rename columns to avoid conflicts
    home_cols = {col: f"home_{col}" for col in df_teams.columns if col != "game_id"}
    away_cols = {col: f"away_{col}" for col in df_teams.columns if col != "game_id"}

    df_home_teams = df_teams[df_teams["HoA"] == "home"].rename(columns=home_cols)
    df_away_teams = df_teams[df_teams["HoA"] == "away"].rename(columns=away_cols)

    # Aggregate goalie stats - keep primary goalie (most time on ice) per game
    df_goalie_agg = (df_goalie.sort_values("timeOnIce", ascending=False)
                              .groupby("game_id").first()
                              .reset_index()
                              .add_prefix("goalie_")
                              .rename(columns={"goalie_game_id": "game_id"}))
    print(f"Aggregated goalie stats: {len(df_goalie_agg)} rows")

    # Merge everything into one row per game
    df_raw = df_game.merge(df_home_teams, on="game_id", how="inner")
    print(f"After merging home team stats: {len(df_raw)} rows")

    df_raw = df_raw.merge(df_away_teams, on="game_id", how="inner")
    print(f"After merging away team stats: {len(df_raw)} rows")

    df_raw = df_raw.merge(df_goalie_agg, on="game_id", how="left")
    print(f"After merging goalie stats: {len(df_raw)} rows")

    df_raw = df_raw.merge(df_team_info, left_on="home_team_id_x", right_on="team_id", how="left")
    print(f"After merging team info: {len(df_raw)} rows")

    print(f"Final dataframe: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    logging.info(f"Data preparation complete: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    print(df_raw.head())

except Exception as e:
    logging.error(f"Data preparation failed: {e}")
    raise
```

    Querying collections from MongoDB...
    Loaded 26305 games
    Loaded 56656 goalie records
    Loaded 52610 team stat records
    Loaded 33 team info records
    After deduplication: 47470 team stat rows
    Aggregated goalie stats: 23721 rows
    After merging home team stats: 26305 rows
    After merging away team stats: 26305 rows
    After merging goalie stats: 26305 rows
    After merging team info: 26305 rows
    Final dataframe: 26305 rows, 76 columns
                          _id_x     game_id    season type         date_time_GMT  \
    0  69dc3f002eb6e0123805e417  2016020906  20162017    R  2017-02-25T22:00:00Z   
    1  69dc3f002eb6e0123805e418  2017021087  20172018    R  2018-03-15T23:30:00Z   
    2  69dc3f002eb6e0123805e41e  2015020646  20152016    R  2016-01-15T00:00:00Z   
    3  69dc3f002eb6e0123805e449  2017020059  20172018    R  2017-10-13T23:00:00Z   
    4  69dc3f002eb6e0123805e461  2015021171  20152016    R  2016-04-02T23:00:00Z   
    
       away_team_id_x  home_team_id_x  away_goals_x  home_goals_x       outcome  \
    0              15              18             2             5  home win REG   
    1               6              13             0             3  home win REG   
    2               3               2             1             3  home win REG   
    3               3              29             1             3  home win REG   
    4               1              14             1             3  home win REG   
    
       ... goalie_savePercentage goalie_powerPlaySavePercentage  \
    0  ...             92.307692                     100.000000   
    1  ...            100.000000                     100.000000   
    2  ...             97.142857                     100.000000   
    3  ...             97.368421                     100.000000   
    4  ...             96.153846                      92.307692   
    
      goalie_evenStrengthSavePercentage                     _id_y  team_id  \
    0                         91.666667  69dd17e32eb6e012381e372e     18.0   
    1                        100.000000  69dd17e32eb6e012381e373a     13.0   
    2                         95.833333  69dd17e32eb6e012381e3738      2.0   
    3                         96.153846  69dd17e32eb6e012381e373f     29.0   
    4                        100.000000  69dd17e32eb6e012381e3728     14.0   
    
      franchiseId     shortName      teamName abbreviation              link  
    0        34.0     Nashville     Predators          NSH  /api/v1/teams/18  
    1        33.0       Florida      Panthers          FLA  /api/v1/teams/13  
    2        22.0  NY Islanders     Islanders          NYI   /api/v1/teams/2  
    3        36.0      Columbus  Blue Jackets          CBJ  /api/v1/teams/29  
    4        31.0     Tampa Bay     Lightning          TBL  /api/v1/teams/14  
    
    [5 rows x 76 columns]



```python
# Check for nulls
print("Null counts:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Check outcome values
print("\nOutcome values:")
print(df_raw["outcome"].value_counts())

# Check for duplicate game_ids
print(f"\nDuplicate game_ids: {df_raw['game_id'].duplicated().sum()}")
```

    Null counts:
    home_rink_side_start                  1196
    home_head_coach                         14
    home_goals_y                             4
    home_shots                               4
    home_hits                             2464
    home_pim                                 4
    home_powerPlayOpportunities              4
    home_powerPlayGoals                      4
    home_faceOffWinPercentage            11074
    home_giveaways                        2464
    home_takeaways                        2464
    home_blocked                          2464
    home_startRinkSide                    1196
    away_head_coach                         14
    away_goals_y                             4
    away_shots                               4
    away_hits                             2464
    away_pim                                 4
    away_powerPlayOpportunities              4
    away_powerPlayGoals                      4
    away_faceOffWinPercentage            11074
    away_giveaways                        2464
    away_takeaways                        2464
    away_blocked                          2464
    away_startRinkSide                    1196
    goalie__id                              14
    goalie_player_id                        14
    goalie_team_id                          14
    goalie_timeOnIce                        14
    goalie_assists                          14
    goalie_goals                            14
    goalie_pim                              14
    goalie_shots                            14
    goalie_saves                            14
    goalie_powerPlaySaves                   14
    goalie_shortHandedSaves                 14
    goalie_evenSaves                        14
    goalie_shortHandedShotsAgainst          14
    goalie_evenShotsAgainst                 14
    goalie_powerPlayShotsAgainst            14
    goalie_decision                         24
    goalie_savePercentage                   14
    goalie_powerPlaySavePercentage         154
    goalie_evenStrengthSavePercentage       18
    _id_y                                   10
    team_id                                 10
    franchiseId                             10
    shortName                               10
    teamName                                10
    abbreviation                            10
    link                                    10
    dtype: int64
    
    Outcome values:
    outcome
    home win REG    11256
    away win REG     8897
    home win OT      2784
    away win OT      2716
    tbc win OT        628
    tbc win tbc        14
    away win tbc        6
    home win tbc        4
    Name: count, dtype: int64
    
    Duplicate game_ids: 2570



```python
try:
    # Drop duplicate game_ids keeping first occurrence
    df_clean = df_raw.drop_duplicates(subset=["game_id"])
    print(f"After dropping duplicates: {len(df_clean)} rows")
    logging.info(f"After dropping duplicates: {len(df_clean)} rows")

    # Filter to only clear home/away wins and create binary target variable
    df_clean = df_clean[df_clean["outcome"].str.contains("home win|away win")]
    df_clean["home_win"] = df_clean["outcome"].str.contains("home win").astype(int)
    print(f"After filtering outcomes: {len(df_clean)} rows")
    print(df_clean["home_win"].value_counts())
    logging.info(f"After filtering outcomes: {len(df_clean)} rows")

    # Drop high-null columns that would hurt model quality
    high_null_cols = [
        "home_faceOffWinPercentage", "away_faceOffWinPercentage",
        "home_hits", "away_hits",
        "home_giveaways", "away_giveaways",
        "home_takeaways", "away_takeaways",
        "home_blocked", "away_blocked",
        "home_rink_side_start", "away_startRinkSide", "home_startRinkSide"
    ]
    df_clean = df_clean.drop(columns=high_null_cols)
    print(f"After dropping high-null columns: {df_clean.shape}")
    logging.info(f"After dropping high-null columns: {df_clean.shape}")

    # Drop rows with nulls in key stat columns
    df_clean = df_clean.dropna(subset=[
        "home_goals_y", "away_goals_y",
        "home_shots", "away_shots",
        "home_powerPlayGoals", "away_powerPlayGoals",
        "goalie_savePercentage",
        "goalie_powerPlaySavePercentage",
        "goalie_evenStrengthSavePercentage"
    ])
    print(f"After dropping null rows: {len(df_clean)} rows")
    logging.info(f"After dropping null rows: {len(df_clean)} rows")

    # Check no nulls remain in key columns
    remaining_nulls = df_clean.isnull().sum()[df_clean.isnull().sum() > 0]
    print(f"\nRemaining nulls:\n{remaining_nulls}")

    print(f"\nFinal clean dataframe: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    logging.info(f"Data cleaning complete: {len(df_clean)} rows, {len(df_clean.columns)} columns")

except Exception as e:
    logging.error(f"Data cleaning failed: {e}")
    raise
```

    After dropping duplicates: 23735 rows
    After filtering outcomes: 23093 rows
    home_win
    1    12682
    0    10411
    Name: count, dtype: int64
    After dropping high-null columns: (23093, 64)
    After dropping null rows: 22978 rows
    
    Remaining nulls:
    Series([], dtype: int64)
    
    Final clean dataframe: 22978 rows, 64 columns



```python
# Uncertainty quantification for numerical features
print("=== Uncertainty Quantification ===\n")

# game collection
game_fields = ["away_goals", "home_goals"]
for field in game_fields:
    vals = pd.to_numeric(df_game[field], errors="coerce")
    print(f"game | {field}")
    print(f"  Min: {vals.min():.2f}, Max: {vals.max():.2f}, Mean: {vals.mean():.2f}, Std: {vals.std():.2f}, Missing: {vals.isna().sum()}")

print()

# game_goalie_stats collection
goalie_fields = ["saves", "shots", "timeOnIce", "savePercentage", "powerPlaySavePercentage"]
for field in goalie_fields:
    vals = pd.to_numeric(df_goalie[field], errors="coerce")
    print(f"game_goalie_stats | {field}")
    print(f"  Min: {np.nanmin(vals):.2f}, Max: {np.nanmax(vals):.2f}, Mean: {np.nanmean(vals):.2f}, Std: {np.nanstd(vals):.2f}, Missing/NaN: {vals.isna().sum() + np.sum(np.isnan(vals.dropna()))}")

print()

# game_teams_stats collection
teams_fields = ["goals", "shots", "powerPlayGoals", "powerPlayOpportunities", "pim", "faceOffWinPercentage"]
for field in teams_fields:
    vals = pd.to_numeric(df_teams[field], errors="coerce")
    print(f"game_teams_stats | {field}")
    print(f"  Min: {vals.min():.2f}, Max: {vals.max():.2f}, Mean: {vals.mean():.2f}, Std: {vals.std():.2f}, Missing: {vals.isna().sum()}")
```

    === Uncertainty Quantification ===
    
    game | away_goals
      Min: 0.00, Max: 11.00, Mean: 2.69, Std: 1.62, Missing: 0
    game | home_goals
      Min: 0.00, Max: 12.00, Mean: 2.96, Std: 1.69, Missing: 0
    
    game_goalie_stats | saves
      Min: 0.00, Max: 85.00, Mean: 25.20, Std: 8.65, Missing/NaN: 0
    game_goalie_stats | shots
      Min: 0.00, Max: 88.00, Mean: 27.68, Std: 8.83, Missing/NaN: 0
    game_goalie_stats | timeOnIce
      Min: 0.00, Max: 9027.00, Mean: 3369.15, Std: 734.19, Missing/NaN: 0
    game_goalie_stats | savePercentage
      Min: 0.00, Max: 100.00, Mean: 90.11, Std: 7.80, Missing/NaN: 139
    game_goalie_stats | powerPlaySavePercentage
      Min: 0.00, Max: 100.00, Mean: 84.26, Std: 22.67, Missing/NaN: 4743
    
    game_teams_stats | goals
      Min: 0.00, Max: 12.00, Mean: 2.76, Std: 1.65, Missing: 8
    game_teams_stats | shots
      Min: 0.00, Max: 88.00, Mean: 29.77, Std: 6.88, Missing: 8
    game_teams_stats | powerPlayGoals
      Min: 0.00, Max: 7.00, Mean: 0.68, Std: 0.82, Missing: 8
    game_teams_stats | powerPlayOpportunities
      Min: 0.00, Max: 16.00, Mean: 3.77, Std: 1.89, Missing: 8
    game_teams_stats | pim
      Min: 0.00, Max: 213.00, Mean: 12.11, Std: 9.21, Missing: 8
    game_teams_stats | faceOffWinPercentage
      Min: 0.00, Max: 79.20, Mean: 49.96, Std: 7.34, Missing: 22148



```python
# Select only numeric columns for correlation
numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()

# Remove non-feature columns
exclude = ["game_id", "season", "away_team_id_x", "home_team_id_x", 
           "away_goals_x", "home_goals_x", "venue_time_zone_offset",
           "home__id", "home_team_id_y", "away__id", "away_team_id_y",
           "goalie__id", "goalie_player_id", "goalie_team_id", "team_id", "franchiseId"]

feature_cols = [col for col in numeric_cols if col not in exclude]

# Compute correlation with target
corr = df_clean[feature_cols].corr()["home_win"].sort_values(ascending=False)
print(corr)

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df_clean[feature_cols].corr(), cmap="coolwarm", center=0, 
            xticklabels=True, yticklabels=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
```

    home_win                             1.000000
    home_goals_y                         0.596494
    home_powerPlayGoals                  0.251080
    away_pim                             0.075957
    home_powerPlayOpportunities          0.039419
    goalie_shortHandedShotsAgainst       0.011528
    home_shots                           0.010962
    home_pim                             0.008554
    goalie_shortHandedSaves              0.008516
    goalie_assists                       0.007108
    away_powerPlayOpportunities          0.006725
    goalie_pim                          -0.002715
    goalie_goals                        -0.007008
    goalie_powerPlaySavePercentage      -0.014791
    away_shots                          -0.018102
    goalie_timeOnIce                    -0.025045
    goalie_evenStrengthSavePercentage   -0.028683
    goalie_savePercentage               -0.031629
    goalie_powerPlayShotsAgainst        -0.042471
    goalie_powerPlaySaves               -0.045866
    goalie_evenShotsAgainst             -0.056834
    goalie_evenSaves                    -0.061725
    goalie_shots                        -0.071310
    goalie_saves                        -0.076901
    away_powerPlayGoals                 -0.235582
    away_goals_y                        -0.594871
    Name: home_win, dtype: float64



    
![png](pipeline_files/pipeline_9_1.png)
    



```python
# Remove leaky post-game stats and ID columns
exclude = ["game_id", "season", "away_team_id_x", "home_team_id_x",
           "away_goals_x", "home_goals_x", "home_goals_y", "away_goals_y",
           "venue_time_zone_offset", "home__id", "home_team_id_y",
           "away__id", "away_team_id_y", "goalie__id", "goalie_player_id",
           "goalie_team_id", "team_id", "franchiseId", "home_win"]

feature_cols = [col for col in df_clean.select_dtypes(include="number").columns
                if col not in exclude]

print(f"Training with {len(feature_cols)} features:")
print(feature_cols)

X = df_clean[feature_cols]
y = df_clean["home_win"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(random_state=42)
}

# Run cross validation on each model and pick the best
print("\nRunning cross validation on all models...")
logging.info("Starting model comparison with cross validation")

results = {}
for name, model in models.items():
    # Use scaled data for Logistic Regression and SVM
    X_cv = X_train_scaled if name in ["Logistic Regression", "SVM"] else X_train
    scores = cross_val_score(model, X_cv, y_train, cv=5, scoring="accuracy")
    results[name] = scores
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    logging.info(f"{name}: CV mean={scores.mean():.4f}, std={scores.std():.4f}")

# Select best model
best_name = max(results, key=lambda k: results[k].mean())
print(f"\nBest model: {best_name} with CV accuracy {results[best_name].mean():.4f}")
logging.info(f"Best model selected: {best_name}")

# Train best model on full training set and evaluate on test set
best_model = models[best_name]
X_tr = X_train_scaled if best_name in ["Logistic Regression", "SVM"] else X_train
X_te = X_test_scaled if best_name in ["Logistic Regression", "SVM"] else X_test

best_model.fit(X_tr, y_train)
y_pred = best_model.predict(X_te)

print(f"\nTest Set Results for {best_name}:")
print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))
logging.info(f"Test set evaluation complete for {best_name}")
```

    Training with 23 features:
    ['home_shots', 'home_pim', 'home_powerPlayOpportunities', 'home_powerPlayGoals', 'away_shots', 'away_pim', 'away_powerPlayOpportunities', 'away_powerPlayGoals', 'goalie_timeOnIce', 'goalie_assists', 'goalie_goals', 'goalie_pim', 'goalie_shots', 'goalie_saves', 'goalie_powerPlaySaves', 'goalie_shortHandedSaves', 'goalie_evenSaves', 'goalie_shortHandedShotsAgainst', 'goalie_evenShotsAgainst', 'goalie_powerPlayShotsAgainst', 'goalie_savePercentage', 'goalie_powerPlaySavePercentage', 'goalie_evenStrengthSavePercentage']
    
    Running cross validation on all models...
    Logistic Regression: mean=0.6641, std=0.0079
    Random Forest: mean=0.8412, std=0.0030
    Gradient Boosting: mean=0.8001, std=0.0119
    SVM: mean=0.7989, std=0.0058
    
    Best model: Random Forest with CV accuracy 0.8412
    
    Test Set Results for Random Forest:
                  precision    recall  f1-score   support
    
        Away Win       0.83      0.81      0.82      2056
        Home Win       0.85      0.86      0.86      2540
    
        accuracy                           0.84      4596
       macro avg       0.84      0.84      0.84      4596
    weighted avg       0.84      0.84      0.84      4596
    


## Analysis Rationale

The data preparation step originally used MongoDB's `$lookup` aggregation pipeline to embed goalie and team stats directly into each game document, demonstrating the document model's embedding flexibility. However, this approach timed out on the Atlas free tier due to the size of the collections. As a practical adaptation, each collection was queried separately using `.find()` and merged in pandas using `game_id` as the join key, producing the same logical structure.

During feature selection, `home_goals` and `away_goals` were initially included as features, but were removed after inspection revealed they caused data leakage.The model achieved 96% accuracy by essentially reading the final score to determine the winner, which is not a valid predictor for pre-outcome prediction. After removing these features, four models were compared using 5-fold cross validation: Logistic Regression (66%), Gradient Boosting (80%), SVM (80%), and Random Forest (84%). Random Forest was selected as the best model. Logistic Regression's lower accuracy suggests the relationships between features and game outcomes are non-linear. An 84% accuracy using in-game statistics is realistic for NHL prediction given the sport's inherently high variance nature.


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1 - Model comparison bar chart
model_names = list(results.keys())
model_means = [results[name].mean() for name in model_names]
model_stds = [results[name].std() for name in model_names]

colors = ["steelblue" if name != best_name else "darkorange" for name in model_names]
bars = axes[0].bar(model_names, model_means, yerr=model_stds, 
                   color=colors, edgecolor="black", capsize=5)
axes[0].set_title("Cross-Validation Accuracy by Model", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.5, 1.0)
axes[0].axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Baseline (50%)")
for bar, mean in zip(bars, model_means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
axes[0].legend()
axes[0].tick_params(axis="x", rotation=15)

# Plot 2 - Feature importance for best model
importances = pd.Series(best_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True).tail(15)

axes[1].barh(importances.index, importances.values, color="steelblue", edgecolor="black")
axes[1].set_title(f"Top 15 Feature Importances\n({best_name})", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Importance")

plt.suptitle("NHL Game Outcome Prediction Results", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/model_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("Visualization saved to results/model_results.png")
logging.info("Visualization saved to results/model_results.png")
```


    
![png](pipeline_files/pipeline_12_0.png)
    


    Visualization saved to results/model_results.png


## Visualization Rationale

Two visualizations were created to communicate the results of the analysis. The first is a bar chart comparing the cross-validation accuracy of all four models tested (Logistic Regression, Random Forest, Gradient Boosting, and SVM) with error bars showing the standard deviation across folds. This chart was chosen because it clearly communicates which model performed best and how consistent each model was across folds, making it easy for a non-technical audience to understand the model comparison. Random Forest is highlighted in orange to draw attention to the selected model.

The second visualization is a horizontal bar chart showing the top 15 most important features identified by the Random Forest model. This chart was chosen because feature importance is one of the most interpretable outputs of a Random Forest classifier and directly addresses the core question of which in-game statistics most strongly predict NHL game outcomes. Shots on goal and goaltender save percentage emerging as the top features aligns with domain knowledge about hockey, lending credibility to the model's findings.
