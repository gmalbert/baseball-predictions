"""Over / Under (totals) prediction model.

Predicts P(actual total runs > expected total) for each game.

Without a live sportsbook feed, the posted total is replaced by an expected
total derived from each team's season scoring averages:

    exp_total = home_RS_G + away_RS_G

This makes the target:
    went_over = 1  if  actual_total > exp_total  else  0

The model supports both XGBoost (default) and LightGBM via use_lightgbm=True.
Weather features (temp, windspeed) are especially informative here.

Target: went_over
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

from .features import TOTALS_FEATURES, implied_probability

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_totals_model(
    features_df: pd.DataFrame,
    feature_cols: list[str] = TOTALS_FEATURES,
    test_size: float = 0.2,
    use_lightgbm: bool = False,
) -> dict:
    """Train the Over/Under prediction model.

    Args:
        features_df: Output of build_model_features() — must contain went_over.
        feature_cols: Feature columns to use (TOTALS_FEATURES by default).
        test_size: Fraction of games reserved for testing.
        use_lightgbm: Use LightGBM instead of XGBoost.

    Returns:
        dict with keys: model, metrics, importances, feature_cols, test_df.
    """
    df = features_df.sort_values("date").dropna(
        subset=["went_over"] + feature_cols
    )

    X = df[feature_cols].values
    y = df["went_over"].astype(int).values

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if use_lightgbm:
        from lightgbm import LGBMClassifier
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
        suffix = "lgbm"
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
        suffix = "xgb"

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
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

    clf = model.named_steps["clf"]
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    test_df = df.iloc[split_idx:][
        ["date", "hometeam", "visteam",
         "hruns", "vruns", "total_runs", "exp_total", "went_over"]
    ].copy().reset_index(drop=True)
    test_df["pred_prob_over"]  = y_prob
    test_df["pred_prob_under"] = 1 - y_prob
    test_df["pick_side"]       = np.where(y_prob >= 0.5, "Over", "Under")
    test_df["correct"]         = (
        (test_df["pick_side"] == "Over")  == (test_df["went_over"] == 1)
    ).astype(int)

    model_path = MODEL_DIR / f"totals_{suffix}_v1.joblib"
    joblib.dump(model, model_path)

    return {
        "model":        model,
        "metrics":      metrics,
        "importances":  importances,
        "feature_cols": feature_cols,
        "test_df":      test_df,
        "train_size":   len(X_train),
        "test_size":    len(X_test),
        "model_path":   model_path,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_totals(
    model_or_path: "Pipeline | str | Path",
    game_features: pd.DataFrame,
    feature_cols: list[str] = TOTALS_FEATURES,
    over_price_col: str | None = None,
    under_price_col: str | None = None,
) -> pd.DataFrame:
    """Generate Over/Under predictions for a set of games.

    Args:
        model_or_path: Trained pipeline or path to saved .joblib file.
        game_features: DataFrame containing feature_cols.
        feature_cols:  Feature columns expected by the model.
        over_price_col:  Optional American odds for the Over.
        under_price_col: Optional American odds for the Under.

    Returns:
        DataFrame with: hometeam, visteam, exp_total, pred_prob_over,
        pred_prob_under, pick_side, pick_prob, [edge] if odds provided.
    """
    if not isinstance(model_or_path, Pipeline):
        model_or_path = joblib.load(model_or_path)

    X = game_features[feature_cols].fillna(0).values
    probs_over = model_or_path.predict_proba(X)[:, 1]

    id_cols = [c for c in ("date", "hometeam", "visteam", "exp_total")
               if c in game_features.columns]
    results = game_features[id_cols].copy().reset_index(drop=True)
    results["pred_prob_over"]  = probs_over.round(4)
    results["pred_prob_under"] = (1 - probs_over).round(4)
    results["pick_side"]       = np.where(probs_over >= 0.5, "Over", "Under")
    results["pick_prob"]       = np.where(
        probs_over >= 0.5, probs_over, 1 - probs_over
    ).round(4)

    if over_price_col and over_price_col in game_features.columns:
        results["edge_over"] = game_features[over_price_col].apply(
            lambda odds: float(probs_over[results.index])
            - implied_probability(int(odds))
            if pd.notna(odds) else np.nan
        )

    return results
