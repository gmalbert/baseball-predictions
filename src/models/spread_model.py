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

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import SPREAD_FEATURES, calculate_edge
from .manifest import LoadedModel

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
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    df = features_df.sort_values("date").dropna(subset=["home_cover"] + feature_cols)

    X = df[feature_cols].values
    y = df["home_cover"].astype(int).values

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # scale_pos_weight compensates for class imbalance
    # (covers rarely exceed 50 % of games)
    pos_rate = y_train.mean()
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    from xgboost import XGBClassifier

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(
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
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    model.feature_cols_ = list(feature_cols)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    importances = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.named_steps["xgb"].feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[split_idx:][["date", "hometeam", "visteam", "hruns", "vruns", "home_cover"]]
        .copy()
        .reset_index(drop=True)
    )
    test_df["pred_prob"] = y_prob
    test_df["pred_cover"] = y_pred
    test_df["correct"] = (test_df["pred_cover"] == test_df["home_cover"]).astype(int)
    test_df["home_margin"] = test_df["hruns"] - test_df["vruns"]

    return {
        "model": model,
        "metrics": metrics,
        "importances": importances,
        "feature_cols": feature_cols,
        "test_df": test_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_spread(
    model_or_path: Pipeline | str | Path,
    game_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    spread_price_col: str | None = None,
    away_spread_price_col: str | None = None,
    home_point_col: str | None = None,
    away_point_col: str | None = None,
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
    loaded_bundle = None
    if not isinstance(model_or_path, Pipeline):
        artifact_path = Path(model_or_path)
        loaded_bundle = LoadedModel(artifact_path, artifact_path.with_suffix(".manifest.json"))
        model_or_path = loaded_bundle.estimator

    feature_cols = (
        [spec.name for spec in loaded_bundle.manifest.features]
        if loaded_bundle
        else list(getattr(model_or_path, "feature_cols_", feature_cols or SPREAD_FEATURES))
    )
    expected = getattr(model_or_path, "n_features_in_", None)
    if expected is not None and expected != len(feature_cols):
        raise ValueError(
            f"Model expects {expected} features but inference contract provides {len(feature_cols)}; retrain the model."
        )
    missing = [col for col in feature_cols if col not in game_features.columns]
    if missing:
        raise ValueError(f"Live feature schema missing {len(missing)} model columns: {missing}")

    if loaded_bundle:
        probs_cover = loaded_bundle.predict_probability(game_features).to_numpy()
    else:
        X = game_features[feature_cols].fillna(0).values
        probs_cover = model_or_path.predict_proba(X)[:, 1]

    id_cols = [c for c in ("game_id", "date", "hometeam", "visteam") if c in game_features.columns]
    results = game_features[id_cols].copy().reset_index(drop=True)
    results["pred_cover_prob"] = probs_cover.round(4)
    results["pred_no_cover_prob"] = (1 - probs_cover).round(4)
    results["pick_side"] = np.where(probs_cover >= 0.5, "Home -1.5", "Away +1.5")
    results["selection"] = np.where(probs_cover >= 0.5, "home", "away")
    results["pick_prob"] = np.where(probs_cover >= 0.5, probs_cover, 1 - probs_cover).round(4)

    if spread_price_col and spread_price_col in game_features.columns:
        results["edge_home"] = [
            calculate_edge(float(prob), odds) if pd.notna(odds) else np.nan
            for prob, odds in zip(probs_cover, game_features[spread_price_col].to_numpy())
        ]
    if away_spread_price_col and away_spread_price_col in game_features.columns:
        results["edge_away"] = [
            calculate_edge(float(prob), odds) if pd.notna(odds) else np.nan
            for prob, odds in zip(1 - probs_cover, game_features[away_spread_price_col].to_numpy())
        ]
    if "edge_home" in results:
        results["edge"] = results["edge_home"]
        if "edge_away" in results:
            results["edge"] = np.where(
                results["selection"] == "home", results["edge_home"], results["edge_away"]
            )
    if spread_price_col and spread_price_col in game_features.columns:
        away_values = (
            game_features[away_spread_price_col].to_numpy()
            if away_spread_price_col and away_spread_price_col in game_features
            else np.full(len(game_features), np.nan)
        )
        results["price_american"] = np.where(
            results["selection"] == "home", game_features[spread_price_col].to_numpy(), away_values
        )
    if home_point_col and home_point_col in game_features.columns:
        away_points = (
            game_features[away_point_col].to_numpy()
            if away_point_col and away_point_col in game_features
            else -game_features[home_point_col].to_numpy()
        )
        results["point"] = np.where(
            results["selection"] == "home", game_features[home_point_col].to_numpy(), away_points
        )

    return results
