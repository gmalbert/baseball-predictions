"""Pre-compute all aggregated datasets for the Streamlit app.

Run this script once after updating the retrosheet Parquet source files:

    python scripts/precompute_data.py

Saves ready-to-display DataFrames to data_files/processed/ so the
Streamlit app never needs to run expensive aggregations at runtime.
Users see instant page loads — all computation happens here, offline.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data_files" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

from retrosheet import (
    season_standings,
    season_team_batting,
    season_team_pitching,
    season_batting_leaders,
    season_pitching_leaders,
    load_gameinfo,
)
from src.models.features import build_model_features

MIN_YEAR = 2000
MAX_YEAR = 2025


def _save(df, name: str) -> None:
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"    -> {len(df):,} rows  ({path.stat().st_size / 1e6:.1f} MB)")


def run() -> None:
    import pandas as pd

    print(f"Pre-computing data for {MIN_YEAR}–{MAX_YEAR}…\n")

    print("  Standings…")
    _save(season_standings(MIN_YEAR, MAX_YEAR), "standings")

    print("  Team batting…")
    _save(season_team_batting(MIN_YEAR, MAX_YEAR), "team_batting")

    print("  Team pitching…")
    _save(season_team_pitching(MIN_YEAR, MAX_YEAR), "team_pitching")

    print("  Batting leaders…")
    _save(season_batting_leaders(MIN_YEAR, MAX_YEAR), "batting_leaders")

    print("  Pitching leaders…")
    _save(season_pitching_leaders(MIN_YEAR, MAX_YEAR), "pitching_leaders")

    print("  Model features…")
    feat_df = build_model_features(MIN_YEAR, MAX_YEAR)
    _save(feat_df, "model_features")

    # ── Train all three ML models ──────────────────────────────────────────
    from src.models.underdog_model import train_moneyline_model
    from src.models.spread_model import train_spread_model
    from src.models.totals_model import train_totals_model

    print("\n  Training moneyline model…")
    r_ml = train_moneyline_model(feat_df)
    print("  Training spread model…")
    r_sp = train_spread_model(feat_df)
    print("  Training totals model…")
    r_ou = train_totals_model(feat_df)

    # Save metrics (one row per model)
    metrics_rows = []
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        row = {"model": name, "train_size": r["train_size"], "test_size": r["test_size"]}
        row.update(r["metrics"])
        metrics_rows.append(row)
    print("  Saving model metrics…")
    _save(pd.DataFrame(metrics_rows), "model_metrics")

    # Save importances (all models, with model column)
    imps = []
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        df_i = r["importances"].copy()
        df_i["model"] = name
        imps.append(df_i)
    print("  Saving model importances…")
    _save(pd.concat(imps, ignore_index=True), "model_importances")

    # Save test DataFrames per model
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        print(f"  Saving {name} test_df…")
        _save(r["test_df"], f"{name}_test_df")

    # ── Walk-forward backtests ─────────────────────────────────────────────
    from xgboost import XGBClassifier
    from src.evaluation.backtester import walk_forward_backtest, BacktestResult
    from src.models.features import MONEYLINE_FEATURES, TOTALS_FEATURES

    def _train_xgb(X, y):
        clf = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        clf.fit(X.fillna(0), y)
        return clf

    def _predict_xgb(model, X):
        return model.predict_proba(X.fillna(0))[:, 1]

    ml_bt_cols = ["date", "gid", "home_win"] + [
        c for c in MONEYLINE_FEATURES if c in feat_df.columns
    ]
    ml_bt = feat_df[ml_bt_cols].rename(columns={"gid": "game_id"}).dropna()

    tot_bt_cols = ["date", "gid", "went_over"] + [
        c for c in TOTALS_FEATURES if c in feat_df.columns
    ]
    tot_bt = feat_df[tot_bt_cols].rename(columns={"gid": "game_id"}).dropna()

    print("  Running moneyline walk-forward backtest…")
    ml_bt_result = walk_forward_backtest(
        features_df=ml_bt, train_fn=_train_xgb, predict_fn=_predict_xgb,
        target_col="home_win", odds_col="home_ml", pick_type="underdog",
        model_name="moneyline", min_edge=0.02,
        train_window_games=1200, test_window_games=200, step_size=100,
    )
    print("  Running totals walk-forward backtest…")
    tot_bt_result = walk_forward_backtest(
        features_df=tot_bt, train_fn=_train_xgb, predict_fn=_predict_xgb,
        target_col="went_over", odds_col="total_line", pick_type="over_under",
        model_name="totals", min_edge=0.02,
        train_window_games=1200, test_window_games=200, step_size=100,
    )

    # Save backtest summaries
    summaries = [ml_bt_result.summary(), tot_bt_result.summary()]
    print("  Saving backtest summaries…")
    _save(pd.DataFrame(summaries), "backtest_summary")

    # Save all individual bets as a flat DataFrame
    def _bets_to_df(bt: BacktestResult) -> pd.DataFrame:
        if not bt.bets:
            return pd.DataFrame()
        return pd.DataFrame([{
            "model_name":      bt.model_name,
            "pick_type":       bt.pick_type,
            "game_id":         b.game_id,
            "date":            b.date,
            "predicted_prob":  b.predicted_prob,
            "confidence_score": b.confidence_score,
            "confidence":      b.confidence,
            "edge":            b.edge,
            "american_odds":   b.american_odds,
            "result":          b.result,
            "profit_units":    b.profit_units,
        } for b in bt.bets])

    bets_df = pd.concat(
        [_bets_to_df(ml_bt_result), _bets_to_df(tot_bt_result)],
        ignore_index=True,
    )
    if not bets_df.empty:
        print("  Saving backtest bets…")
        _save(bets_df, "backtest_bets")

    print("\nDone. Commit data_files/processed/*.parquet to git.")


if __name__ == "__main__":
    run()
