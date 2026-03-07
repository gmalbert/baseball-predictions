"""Moneyline (home win) prediction model.

Predicts P(home team wins) for each game.

Without live odds data the "underdog edge" calculation is not possible, but
the probability estimates are still useful for:
  - Understanding which team has higher model-implied win probability
  - Comparing against posted moneyline odds when they become available
  - Backtesting historical accuracy

Target: home_win (1 = home team won, 0 = away team won)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import MONEYLINE_FEATURES, implied_probability, calculate_edge

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "moneyline_xgb_v1.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_moneyline_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = MONEYLINE_FEATURES,
    test_size: float = 0.2,
) -> dict:
    """Train the home-win (moneyline) prediction model.

    Uses a chronological train/test split to prevent look-ahead bias.

    Args:
        features_df: Output of build_model_features().
        feature_cols: Feature columns to use.
        test_size: Fraction of games reserved for testing (most-recent games).

    Returns:
        dict with keys: model, metrics, importances, feature_cols,
                        test_df (actual vs predicted for the test set).
    """
    df = features_df.sort_values("date").dropna(
        subset=["home_win"] + feature_cols
    )

    X = df[feature_cols].values
    y = df["home_win"].astype(int).values

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

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
        )),
    ])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "log_loss":    float(log_loss(y_test, y_prob)),
        "roc_auc":     float(roc_auc_score(y_test, y_prob)),
    }

    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.named_steps["xgb"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Attach predictions to the test slice for backtest display
    test_df = df.iloc[split_idx:][
        ["date", "hometeam", "visteam", "hruns", "vruns", "home_win"]
    ].copy().reset_index(drop=True)
    test_df["pred_prob"] = y_prob
    test_df["pred_win"]  = y_pred
    test_df["correct"]   = (test_df["pred_win"] == test_df["home_win"]).astype(int)

    joblib.dump(model, MODEL_PATH)

    return {
        "model":        model,
        "metrics":      metrics,
        "importances":  importances,
        "feature_cols": feature_cols,
        "test_df":      test_df,
        "train_size":   len(X_train),
        "test_size":    len(X_test),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_moneyline(
    model_or_path: "Pipeline | str | Path",
    game_features: pd.DataFrame,
    feature_cols: list[str] = MONEYLINE_FEATURES,
    home_ml_col: str | None = None,
    away_ml_col: str | None = None,
) -> pd.DataFrame:
    """Generate moneyline win-probability predictions for a set of games.

    Args:
        model_or_path: Trained pipeline or path to a saved .joblib file.
        game_features: DataFrame containing feature_cols and identifier columns.
        feature_cols:  Feature columns expected by the model.
        home_ml_col:   Optional column with home moneyline (American odds).
                       If provided, edge vs. the line is computed.
        away_ml_col:   Optional column with away moneyline.

    Returns:
        DataFrame with columns: hometeam, visteam, pred_home_win_prob,
        pred_away_win_prob, pick, [edge_home, edge_away] if odds provided.
    """
    if not isinstance(model_or_path, Pipeline):
        model_or_path = joblib.load(model_or_path)

    X = game_features[feature_cols].fillna(0).values
    probs_home = model_or_path.predict_proba(X)[:, 1]

    id_cols = [c for c in ("date", "hometeam", "visteam") if c in game_features.columns]
    results = game_features[id_cols].copy().reset_index(drop=True)
    results["pred_home_win_prob"] = probs_home.round(4)
    results["pred_away_win_prob"] = (1 - probs_home).round(4)
    results["pick"] = np.where(probs_home >= 0.5, "Home", "Away")

    if home_ml_col and home_ml_col in game_features.columns:
        results["edge_home"] = pd.Series(game_features[home_ml_col].values).apply(
            lambda odds: calculate_edge(float(probs_home[results.index]), odds)
            if pd.notna(odds) else np.nan
        )
    if away_ml_col and away_ml_col in game_features.columns:
        probs_away = 1 - probs_home
        results["edge_away"] = pd.Series(game_features[away_ml_col].values).apply(
            lambda odds: calculate_edge(float(probs_away[results.index]), odds)
            if pd.notna(odds) else np.nan
        )

    return results
