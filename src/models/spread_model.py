"""Run-line (spread) prediction model.

Predicts P(home team covers -1.5 runs) — i.e. wins by 2 or more runs.

In MLB, the standard run line is −1.5 (favorite) / +1.5 (underdog).
Without identified favorites we predict from the home team's perspective:
  1 = home team wins by 2+ runs  (would cover −1.5 if home were the fav)
  0 = home team wins by 0–1 or loses

Target: home_cover
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

from .features import SPREAD_FEATURES, implied_probability, calculate_edge

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "spread_xgb_v1.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_spread_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = SPREAD_FEATURES,
    test_size: float = 0.2,
) -> dict:
    """Train the run-line (spread) prediction model.

    Args:
        features_df: Output of build_model_features() — must contain home_cover.
        feature_cols: Feature columns to use.
        test_size: Fraction of games reserved for testing.

    Returns:
        dict with keys: model, metrics, importances, feature_cols, test_df.
    """
    df = features_df.sort_values("date").dropna(
        subset=["home_cover"] + feature_cols
    )

    X = df[feature_cols].values
    y = df["home_cover"].astype(int).values

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # scale_pos_weight compensates for class imbalance
    # (covers rarely exceed 50 % of games)
    pos_rate = y_train.mean()
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

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
            scale_pos_weight=spw,
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

    test_df = df.iloc[split_idx:][
        ["date", "hometeam", "visteam", "hruns", "vruns", "home_cover"]
    ].copy().reset_index(drop=True)
    test_df["pred_prob"] = y_prob
    test_df["pred_cover"] = y_pred
    test_df["correct"] = (test_df["pred_cover"] == test_df["home_cover"]).astype(int)
    test_df["home_margin"] = test_df["hruns"] - test_df["vruns"]

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

def predict_spread(
    model_or_path: "Pipeline | str | Path",
    game_features: pd.DataFrame,
    feature_cols: list[str] = SPREAD_FEATURES,
    spread_price_col: str | None = None,
) -> pd.DataFrame:
    """Generate run-line cover predictions for a set of games.

    Args:
        model_or_path: Trained pipeline or path to a saved .joblib file.
        game_features: DataFrame containing feature_cols.
        feature_cols:  Feature columns expected by the model.
        spread_price_col: Optional column with American odds for the -1.5 line.

    Returns:
        DataFrame with: hometeam, visteam, pred_cover_prob, pick_side,
        [edge] if odds provided.
    """
    if not isinstance(model_or_path, Pipeline):
        model_or_path = joblib.load(model_or_path)

    X = game_features[feature_cols].fillna(0).values
    probs_cover = model_or_path.predict_proba(X)[:, 1]

    id_cols = [c for c in ("date", "hometeam", "visteam") if c in game_features.columns]
    results = game_features[id_cols].copy().reset_index(drop=True)
    results["pred_cover_prob"]      = probs_cover.round(4)
    results["pred_no_cover_prob"]   = (1 - probs_cover).round(4)
    results["pick_side"]            = np.where(
        probs_cover >= 0.5, "Home −1.5", "Away +1.5"
    )

    if spread_price_col and spread_price_col in game_features.columns:
        results["edge"] = game_features[spread_price_col].apply(
            lambda odds: calculate_edge(float(probs_cover[results.index]), odds)
            if pd.notna(odds) else np.nan
        )

    return results
