# 04 – Betting Models

✅ **Completed** — three prediction models (Underdog moneyline, Spread run line, Over/Under totals) implemented with probability + confidence outputs.

---

## Model Architecture Overview

```
                    ┌─────────────────────┐
                    │   Feature Pipeline   │
                    │  (shared features)   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Underdog  │  │   Spread   │  │   Over /   │
     │  Model     │  │   Model    │  │   Under    │
     │ (binary)   │  │ (binary)   │  │  (binary)  │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           ▼               ▼               ▼
     P(underdog win)  P(cover -1.5)   P(over total)
     + confidence     + confidence    + confidence
```

---

## 1. Shared Feature Engineering ✅ Completed

```python
# src/models/features.py
"""Feature engineering pipeline shared by all three models."""

import pandas as pd
import numpy as np
from typing import Optional


def compute_rolling_stats(
    df: pd.DataFrame,
    team_col: str,
    stat_cols: list[str],
    windows: list[int] = [10, 30],
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute rolling averages for team stats over recent games.
    
    Args:
        df: Game-level DataFrame sorted by date
        team_col: Column identifying the team
        stat_cols: Columns to compute rolling stats for
        windows: Rolling window sizes (in games)
        date_col: Date column for sorting
    
    Returns:
        DataFrame with additional rolling columns
    """
    df = df.sort_values(date_col)
    
    for window in windows:
        for col in stat_cols:
            new_col = f"{col}_roll_{window}"
            df[new_col] = (
                df.groupby(team_col)[col]
                .transform(lambda x: x.rolling(window, min_periods=3).mean())
            )
    return df


def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability.
    
    Examples:
        +150 → 0.400 (bet $100 to win $150)
        -110 → 0.524 (bet $110 to win $100)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_edge(predicted_prob: float, american_odds: int) -> float:
    """Calculate betting edge: model probability - implied probability.
    
    Positive edge = model thinks outcome is more likely than the line implies.
    """
    implied = implied_probability(american_odds)
    return predicted_prob - implied


def build_game_features(
    games_df: pd.DataFrame,
    team_batting_df: pd.DataFrame,
    team_pitching_df: pd.DataFrame,
    pitcher_stats_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the full feature matrix for a set of games.
    
    Takes game-level data and enriches with team stats, pitcher stats,
    and weather to create model-ready features.
    """
    features = games_df.copy()
    
    # ---- Team Batting Features (home & away) ----
    bat_cols = ["batting_avg", "team_obp", "team_slg", "team_ops",
                "team_wrc_plus", "team_hr", "runs_scored"]
    
    for prefix, team_col in [("away", "away_team_id"), ("home", "home_team_id")]:
        merged = features.merge(
            team_batting_df[["team_id", "season"] + bat_cols],
            left_on=[team_col, "season"],
            right_on=["team_id", "season"],
            how="left",
            suffixes=("", f"_{prefix}"),
        )
        for col in bat_cols:
            features[f"{prefix}_{col}"] = merged[col]
    
    # ---- Team Pitching Features ----
    pitch_cols = ["team_era", "team_fip", "team_whip", "team_k_per_9", "runs_allowed"]
    
    for prefix, team_col in [("away", "away_team_id"), ("home", "home_team_id")]:
        merged = features.merge(
            team_pitching_df[["team_id", "season"] + pitch_cols],
            left_on=[team_col, "season"],
            right_on=["team_id", "season"],
            how="left",
        )
        for col in pitch_cols:
            features[f"{prefix}_{col}"] = merged[col]
    
    # ---- Starting Pitcher Features ----
    sp_cols = ["era", "fip", "xfip", "whip", "k_per_9", "bb_per_9",
               "hard_hit_pct", "barrel_pct", "war"]
    
    for prefix, starter_col in [("away_sp", "away_starter_id"), ("home_sp", "home_starter_id")]:
        merged = features.merge(
            pitcher_stats_df[["player_id", "season"] + sp_cols],
            left_on=[starter_col, "season"],
            right_on=["player_id", "season"],
            how="left",
        )
        for col in sp_cols:
            features[f"{prefix}_{col}"] = merged[col]
    
    # ---- Weather Features ----
    if weather_df is not None:
        features = features.merge(
            weather_df[["game_id", "temp_f", "wind_mph", "wind_dir_deg",
                        "precip_prob_pct", "is_dome"]],
            on="game_id",
            how="left",
        )
        # Fill domed stadiums with neutral values
        features.loc[features["is_dome"] == True, "temp_f"] = 72.0
        features.loc[features["is_dome"] == True, "wind_mph"] = 0.0
    
    # ---- Derived Features ----
    
    # Run differential per game (season-level proxy for team strength)
    features["home_run_diff_per_game"] = (
        (features["home_runs_scored"] - features["home_runs_allowed"]) / 162
    )
    features["away_run_diff_per_game"] = (
        (features["away_runs_scored"] - features["away_runs_allowed"]) / 162
    )
    
    # Pitcher quality gap
    features["sp_era_gap"] = features["away_sp_era"] - features["home_sp_era"]
    features["sp_fip_gap"] = features["away_sp_fip"] - features["home_sp_fip"]
    features["sp_war_gap"] = features["home_sp_war"] - features["away_sp_war"]
    
    # Team quality gap
    features["ops_gap"] = features["home_team_ops"] - features["away_team_ops"]
    features["era_gap"] = features["away_team_era"] - features["home_team_era"]
    features["wrc_gap"] = features["home_team_wrc_plus"] - features["away_team_wrc_plus"]
    
    # Expected total runs (simple Pythagorean estimate)
    features["exp_total_runs"] = (
        (features["home_runs_scored"] + features["away_runs_scored"]) / 162 * 2
    )
    
    return features


# ---- Feature column lists for each model ----

SHARED_FEATURES = [
    # Home team batting
    "home_batting_avg", "home_team_obp", "home_team_slg", "home_team_ops",
    "home_team_wrc_plus",
    # Away team batting
    "away_batting_avg", "away_team_obp", "away_team_slg", "away_team_ops",
    "away_team_wrc_plus",
    # Home team pitching
    "home_team_era", "home_team_fip", "home_team_whip", "home_team_k_per_9",
    # Away team pitching
    "away_team_era", "away_team_fip", "away_team_whip", "away_team_k_per_9",
    # Starting pitchers
    "home_sp_era", "home_sp_fip", "home_sp_whip", "home_sp_k_per_9",
    "home_sp_hard_hit_pct", "home_sp_war",
    "away_sp_era", "away_sp_fip", "away_sp_whip", "away_sp_k_per_9",
    "away_sp_hard_hit_pct", "away_sp_war",
    # Gaps
    "sp_era_gap", "sp_fip_gap", "sp_war_gap",
    "ops_gap", "era_gap", "wrc_gap",
]

WEATHER_FEATURES = ["temp_f", "wind_mph", "wind_dir_deg", "precip_prob_pct"]

UNDERDOG_FEATURES = SHARED_FEATURES + WEATHER_FEATURES + [
    "home_run_diff_per_game", "away_run_diff_per_game",
]

SPREAD_FEATURES = UNDERDOG_FEATURES  # same features, different target

TOTAL_FEATURES = SHARED_FEATURES + WEATHER_FEATURES + [
    "exp_total_runs",
    "home_runs_scored", "away_runs_scored",
    "home_runs_allowed", "away_runs_allowed",
]
```

---

## 2. Underdog Model (Moneyline) ✅ Completed (Moneyline)

Predicts whether the **moneyline underdog** will win outright.

```python
# src/models/underdog_model.py
"""Underdog moneyline prediction model.

Target: Does the underdog win? (binary classification)
A positive edge vs. the line means we have a profitable bet.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
from .features import (
    UNDERDOG_FEATURES, implied_probability, calculate_edge,
    build_game_features,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def identify_underdog(row: pd.Series) -> dict:
    """Given a game row with odds, identify the underdog.
    
    Returns dict with underdog team, odds, and whether they won.
    """
    # In a moneyline market, the team with the + odds is the underdog
    # If home_ml > away_ml (both positive or home more positive), home is underdog
    home_ml = row.get("home_moneyline", 0)
    away_ml = row.get("away_moneyline", 0)
    
    if home_ml > away_ml:
        return {
            "underdog": "home",
            "underdog_odds": home_ml,
            "underdog_won": row.get("home_win", False),
        }
    else:
        return {
            "underdog": "away",
            "underdog_odds": away_ml,
            "underdog_won": not row.get("home_win", True),
        }


def train_underdog_model(
    features_df: pd.DataFrame,
    target_col: str = "underdog_won",
    feature_cols: list[str] = UNDERDOG_FEATURES,
    test_size: float = 0.2,
) -> dict:
    """Train the underdog prediction model.
    
    Uses time-series split to avoid look-ahead bias.
    
    Returns:
        dict with model, scaler, metrics, and feature importances
    """
    # Sort chronologically
    df = features_df.sort_values("date").dropna(subset=[target_col] + feature_cols)
    
    X = df[feature_cols].values
    y = df[target_col].astype(int).values
    
    # Time-based train/test split (no shuffling!)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)} games  |  Testing: {len(X_test)} games")
    print(f"Underdog win rate (train): {y_train.mean():.3f}")
    print(f"Underdog win rate (test):  {y_test.mean():.3f}")
    
    # Build pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )),
    ])
    
    # Train with early stopping via validation set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
    }
    
    print("\n--- Underdog Model Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Fav wins", "Dog wins"]))
    
    # Feature importance
    xgb_model = model.named_steps["xgb"]
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Features:")
    print(importances.head(10).to_string(index=False))
    
    # Save
    model_path = MODEL_DIR / "underdog_xgb_v1.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved → {model_path}")
    
    return {
        "model": model,
        "metrics": metrics,
        "importances": importances,
        "feature_cols": feature_cols,
    }


def predict_underdog(
    model_path: str,
    game_features: pd.DataFrame,
    feature_cols: list[str] = UNDERDOG_FEATURES,
    odds_col: str = "underdog_odds",
) -> pd.DataFrame:
    """Generate underdog predictions for upcoming games.
    
    Returns DataFrame with: game_id, predicted_prob, confidence,
    confidence_score, edge, pick_value.
    """
    model = joblib.load(model_path)
    
    X = game_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    
    results = game_features[["game_id", "away_team", "home_team", odds_col]].copy()
    results["predicted_prob"] = probs
    
    # Calculate edge vs. implied probability
    results["implied_prob"] = results[odds_col].apply(implied_probability)
    results["edge"] = results["predicted_prob"] - results["implied_prob"]
    
    # Confidence scoring
    results["confidence_score"] = _compute_confidence(
        results["predicted_prob"],
        results["edge"],
    )
    results["confidence"] = results["confidence_score"].apply(_confidence_label)
    
    # Pick value string
    results["pick_value"] = results.apply(
        lambda r: f"{'Home' if r.get('underdog') == 'home' else 'Away'} "
                  f"{r['underdog_odds']:+d}",
        axis=1,
    )
    
    return results


def _compute_confidence(prob: pd.Series, edge: pd.Series) -> pd.Series:
    """Combine probability strength and edge into a 0-1 confidence score.
    
    Higher confidence when:
    - Model probability diverges significantly from 0.5
    - Edge over the line is large
    """
    # Probability component (how sure the model is)
    prob_strength = (prob - 0.5).abs() * 2  # 0 at 50%, 1 at 0%/100%
    
    # Edge component (clipped to 0-0.20 range, scaled to 0-1)
    edge_strength = (edge.clip(0, 0.20) / 0.20)
    
    # Weighted combination
    confidence = 0.4 * prob_strength + 0.6 * edge_strength
    return confidence.clip(0, 1)


def _confidence_label(score: float) -> str:
    """Convert numeric confidence to category."""
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"
```

---

## 3. Spread Model (Run Line) ✅ Completed (Run Line -1.5)

Predicts whether the **favorite covers the -1.5 run line**.

```python
# src/models/spread_model.py
"""Run line (spread) prediction model.

Target: Does the favorite cover -1.5? (binary classification)
Standard MLB run line is -1.5 / +1.5.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
from .features import SPREAD_FEATURES, implied_probability, calculate_edge

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def prepare_spread_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the spread target variable.
    
    Target: 1 if favorite won by 2+ runs (covers -1.5), else 0.
    """
    df = df.copy()
    
    # Identify favorite (lower/more-negative moneyline)
    df["favorite_is_home"] = df["home_moneyline"] < df["away_moneyline"]
    
    # Calculate margin from favorite's perspective
    df["fav_margin"] = np.where(
        df["favorite_is_home"],
        df["home_score"] - df["away_score"],   # home is fav
        df["away_score"] - df["home_score"],   # away is fav
    )
    
    # Did the favorite cover -1.5?
    df["covers_spread"] = (df["fav_margin"] >= 2).astype(int)
    
    return df


def train_spread_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = SPREAD_FEATURES,
    test_size: float = 0.2,
) -> dict:
    """Train the run-line spread model."""
    
    df = prepare_spread_target(features_df)
    df = df.sort_values("date").dropna(subset=["covers_spread"] + feature_cols)
    
    X = df[feature_cols].values
    y = df["covers_spread"].values
    
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)} games  |  Testing: {len(X_test)} games")
    print(f"Cover rate (train): {y_train.mean():.3f}")
    print(f"Cover rate (test):  {y_test.mean():.3f}")
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=y_train.mean() / (1 - y_train.mean()),
            eval_metric="logloss",
            random_state=42,
        )),
    ])
    
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
    }
    
    print("\n--- Spread Model Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No cover", "Covers"]))
    
    # Save
    model_path = MODEL_DIR / "spread_xgb_v1.joblib"
    joblib.dump(model, model_path)
    
    return {"model": model, "metrics": metrics, "feature_cols": feature_cols}


def predict_spread(
    model_path: str,
    game_features: pd.DataFrame,
    feature_cols: list[str] = SPREAD_FEATURES,
    spread_odds_col: str = "spread_price",
) -> pd.DataFrame:
    """Generate spread predictions for upcoming games."""
    model = joblib.load(model_path)
    
    X = game_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    
    results = game_features[["game_id", "away_team", "home_team"]].copy()
    results["predicted_prob"] = probs
    
    # Edge vs. line
    if spread_odds_col in game_features.columns:
        results["implied_prob"] = game_features[spread_odds_col].apply(implied_probability)
        results["edge"] = results["predicted_prob"] - results["implied_prob"]
    else:
        results["edge"] = 0.0
    
    # Confidence
    results["confidence_score"] = _compute_spread_confidence(
        results["predicted_prob"], results["edge"]
    )
    results["confidence"] = results["confidence_score"].apply(
        lambda s: "high" if s >= 0.65 else ("medium" if s >= 0.35 else "low")
    )
    
    # The pick: which side of the spread
    results["pick_value"] = results.apply(
        lambda r: f"{'Home' if r['predicted_prob'] > 0.5 else 'Away'} -1.5"
                  if r["predicted_prob"] != 0.5 else "No pick",
        axis=1,
    )
    
    return results


def _compute_spread_confidence(prob, edge):
    """Spread-specific confidence: rewards strong probabilities + positive edge."""
    prob_strength = (prob - 0.5).abs() * 2
    edge_strength = (edge.clip(0, 0.15) / 0.15)
    return (0.5 * prob_strength + 0.5 * edge_strength).clip(0, 1)
```

---

## 4. Over/Under Model (Totals) ✅ Completed (Totals)

Predicts whether the game total will go **Over** or **Under** the posted total.

```python
# src/models/totals_model.py
"""Over/Under (totals) prediction model.

Target: Does the game go Over the posted total? (binary classification)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .features import TOTAL_FEATURES, implied_probability

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def prepare_totals_target(df: pd.DataFrame, total_col: str = "posted_total") -> pd.DataFrame:
    """Create the over/under target variable.
    
    Target: 1 if actual total > posted total (Over), 0 if Under.
    Pushes (exact match) are excluded.
    """
    df = df.copy()
    df["actual_total"] = df["away_score"] + df["home_score"]
    df["went_over"] = (df["actual_total"] > df[total_col]).astype(int)
    
    # Exclude pushes
    df = df[df["actual_total"] != df[total_col]]
    
    return df


def train_totals_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = TOTAL_FEATURES,
    test_size: float = 0.2,
    use_lightgbm: bool = False,
) -> dict:
    """Train the over/under prediction model.
    
    Weather features are especially important here:
    - High temp → more runs (balls carry farther)
    - Wind blowing out → more home runs
    - Wind blowing in → fewer home runs
    """
    df = features_df.sort_values("date").dropna(subset=["went_over"] + feature_cols)
    
    X = df[feature_cols].values
    y = df["went_over"].values
    
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)} games  |  Testing: {len(X_test)} games")
    print(f"Over rate (train): {y_train.mean():.3f}")
    print(f"Over rate (test):  {y_test.mean():.3f}")
    
    if use_lightgbm:
        classifier = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    else:
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])
    
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
    }
    
    print("\n--- Totals Model Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Under", "Over"]))
    
    # Feature importance
    clf = model.named_steps["clf"]
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Features for Totals:")
    print(importances.head(10).to_string(index=False))
    
    # Save
    suffix = "lgbm" if use_lightgbm else "xgb"
    model_path = MODEL_DIR / f"totals_{suffix}_v1.joblib"
    joblib.dump(model, model_path)
    
    return {"model": model, "metrics": metrics, "importances": importances}


def predict_totals(
    model_path: str,
    game_features: pd.DataFrame,
    feature_cols: list[str] = TOTAL_FEATURES,
) -> pd.DataFrame:
    """Generate over/under predictions for upcoming games."""
    model = joblib.load(model_path)
    
    X = game_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]  # P(Over)
    
    results = game_features[["game_id", "away_team", "home_team", "posted_total"]].copy()
    results["predicted_prob_over"] = probs
    results["predicted_prob_under"] = 1 - probs
    
    # Pick the side with higher probability
    results["pick_side"] = np.where(probs > 0.5, "Over", "Under")
    results["pick_prob"] = np.where(probs > 0.5, probs, 1 - probs)
    
    # Edge calculation
    results["edge"] = results.apply(
        lambda r: calculate_edge_totals(r), axis=1
    )
    
    # Confidence
    results["confidence_score"] = _compute_totals_confidence(
        results["pick_prob"], results["edge"]
    )
    results["confidence"] = results["confidence_score"].apply(
        lambda s: "high" if s >= 0.65 else ("medium" if s >= 0.35 else "low")
    )
    
    # Pick value string
    results["pick_value"] = results.apply(
        lambda r: f"{r['pick_side']} {r['posted_total']}", axis=1
    )
    
    return results


def calculate_edge_totals(row):
    """Calculate edge for totals bet."""
    if row["pick_side"] == "Over":
        implied = implied_probability(row.get("over_price", -110))
        return row["predicted_prob_over"] - implied
    else:
        implied = implied_probability(row.get("under_price", -110))
        return row["predicted_prob_under"] - implied


def _compute_totals_confidence(prob, edge):
    """Totals-specific confidence scoring."""
    prob_strength = (prob - 0.5).abs() * 2
    edge_strength = (edge.clip(0, 0.15) / 0.15)
    return (0.45 * prob_strength + 0.55 * edge_strength).clip(0, 1)
```

---

## 5. Ensemble / Meta-Model (Optional Enhancement) ✅ Added utilities

```python
# src/models/ensemble.py
"""Combine multiple model outputs for more robust predictions."""

import numpy as np
import pandas as pd


def ensemble_predictions(
    model_predictions: list[pd.DataFrame],
    weights: list[float] | None = None,
    prob_col: str = "predicted_prob",
) -> pd.DataFrame:
    """Weighted average of multiple model predictions.
    
    Args:
        model_predictions: List of DataFrames, each with game_id and prob_col
        weights: Optional weights (must sum to 1.0), defaults to equal weight
    """
    if weights is None:
        weights = [1.0 / len(model_predictions)] * len(model_predictions)
    
    assert abs(sum(weights) - 1.0) < 0.001, "Weights must sum to 1.0"
    
    # Use first DataFrame as base
    result = model_predictions[0][["game_id"]].copy()
    
    # Weighted average of probabilities
    weighted_prob = np.zeros(len(result))
    for df, w in zip(model_predictions, weights):
        weighted_prob += df[prob_col].values * w
    
    result["ensemble_prob"] = weighted_prob
    return result
```

---

## Requirements Addition

```text
# requirements.txt additions for modeling
xgboost>=2.0.0
lightgbm>=4.1.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## 6. Automated Model Training ✅

While the Streamlit **Models** tab provides a convenient interactive way to rebuild
the feature matrix and train the three classifiers, you don’t need to click
anything manually once the project is deployed. A simple CLI helper exists at

```text
scripts/train_models.py
```

Running it performs the same steps as the UI: it calls
`build_model_features()` for the desired year range and then invokes
`train_moneyline_model()`, `train_spread_model()` and `train_totals_model()` from
`src/models`. Models are serialized to the `models/` directory just as they are
when training interactively.

The daily ingestion workflow (`.github/workflows/ingestion.yml`) now includes a
post‑ingestion step that executes this script *only* during baseball season
(March–November). That means whenever new data lands, the serialized models are
re‑trained automatically without any human intervention. Outside of season the
workflow exits early and the models remain unchanged.

To run the training manually for debugging or an off‑cycle update:

```bash
python scripts/train_models.py --start-year 2020 --end-year 2025
```

The workflow log will display ROC‑AUC and other basic metrics for each model.

> **Next:** [05-model-evaluation.md](05-model-evaluation.md) – Measuring how well these models actually perform.
