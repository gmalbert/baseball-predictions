# scripts/run_evaluation.py
"""Run full model evaluation pipeline.

Usage:
    python scripts/run_evaluation.py

The script trains all three models on the historical Retrosheet data,
then produces calibration reports, profitability breakdowns, and an
edge-filter analysis for each one.
"""

import sys
from pathlib import Path

# ensure repo root is on the path when running as a script
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from models.features import build_model_features
from models.underdog_model import train_moneyline_model
from models.spread_model import train_spread_model
from models.totals_model import train_totals_model

from evaluation.backtester import walk_forward_backtest, BacktestResult
from evaluation.calibration import calibration_report
from evaluation.profitability import (
    profitability_report,
    edge_filter_analysis,
    monthly_breakdown,
)
from src.evaluation.dashboard import generate_dashboard_data


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # ── 1. Build feature matrix ───────────────────────────────────────────────
    _section("BUILDING FEATURE MATRIX")
    features_df = build_model_features(2020, 2025)
    print(f"Feature matrix: {features_df.shape[0]:,} games × {features_df.shape[1]} columns")

    # ── 2. Train each model ───────────────────────────────────────────────────
    _section("TRAINING MONEYLINE MODEL")
    ml_result = train_moneyline_model(features_df)
    print(f"  AUC:      {ml_result['metrics']['roc_auc']:.4f}")
    print(f"  Accuracy: {ml_result['metrics']['accuracy']:.4f}")
    print(f"  Brier:    {ml_result['metrics']['brier_score']:.4f}")

    _section("TRAINING SPREAD MODEL")
    sp_result = train_spread_model(features_df)
    print(f"  AUC:      {sp_result['metrics']['roc_auc']:.4f}")
    print(f"  Accuracy: {sp_result['metrics']['accuracy']:.4f}")
    print(f"  Brier:    {sp_result['metrics']['brier_score']:.4f}")

    _section("TRAINING TOTALS MODEL")
    tot_result = train_totals_model(features_df)
    print(f"  AUC:      {tot_result['metrics']['roc_auc']:.4f}")
    print(f"  Accuracy: {tot_result['metrics']['accuracy']:.4f}")
    print(f"  Brier:    {tot_result['metrics']['brier_score']:.4f}")

    # ── 3. Calibration reports ────────────────────────────────────────────────
    _section("CALIBRATION ANALYSIS")

    ml_test = ml_result["test_df"]
    calibration_report(
        ml_test["home_win"].values,
        ml_test["pred_prob"].values,
        model_name="Moneyline",
    )

    sp_test = sp_result["test_df"]
    calibration_report(
        sp_test["home_cover"].values,
        sp_test["pred_prob"].values,
        model_name="Spread",
    )

    tot_test = tot_result["test_df"]
    calibration_report(
        tot_test["went_over"].values,
        tot_test["pred_prob_over"].values,
        model_name="Totals",
    )

    # ── 4. Walk-forward backtests ─────────────────────────────────────────────
    _section("WALK-FORWARD BACKTEST — Moneyline")
    from xgboost import XGBClassifier

    def _train_xgb(X: pd.DataFrame, y: pd.Series):
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(X.fillna(0), y)
        return clf

    def _predict_xgb(model, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X.fillna(0))[:, 1]

    from src.models.features import MONEYLINE_FEATURES, TOTALS_FEATURES, SPREAD_FEATURES

    # Moneyline backtest
    # feature matrix uses 'gid' not 'game_id'; no live odds yet so odds_col
    # falls back to the default -110 inside calculate_profit.
    ml_bt_cols = ["date", "gid", "home_win"] + [
        c for c in MONEYLINE_FEATURES if c in features_df.columns
    ]
    ml_bt_df = features_df[ml_bt_cols].rename(columns={"gid": "game_id"}).dropna()
    ml_backtest: BacktestResult = walk_forward_backtest(
        features_df=ml_bt_df,
        train_fn=_train_xgb,
        predict_fn=_predict_xgb,
        target_col="home_win",
        odds_col="home_ml",       # missing → default -110 inside backtester
        pick_type="underdog",
        model_name="xgb_moneyline_v1",
        min_edge=0.02,
        train_window_games=1200,
        test_window_games=200,
        step_size=100,
    )
    profitability_report(ml_backtest)
    edge_filter_analysis(ml_backtest)

    # Totals backtest
    _section("WALK-FORWARD BACKTEST — Totals")
    tot_bt_cols = ["date", "gid", "went_over"] + [
        c for c in TOTALS_FEATURES if c in features_df.columns
    ]
    tot_bt_df = features_df[tot_bt_cols].rename(columns={"gid": "game_id"}).dropna()
    tot_backtest: BacktestResult = walk_forward_backtest(
        features_df=tot_bt_df,
        train_fn=_train_xgb,
        predict_fn=_predict_xgb,
        target_col="went_over",
        odds_col="total_line",
        pick_type="over_under",
        model_name="xgb_totals_v1",
        min_edge=0.02,
        train_window_games=1200,
        test_window_games=200,
        step_size=100,
    )
    profitability_report(tot_backtest)
    edge_filter_analysis(tot_backtest)

    # ── 5. Dashboard data ─────────────────────────────────────────────────────
    _section("GENERATING DASHBOARD DATA")
    backtest_results = {
        "moneyline": ml_backtest,
        "totals": tot_backtest,
    }
    y_true_map = {
        "moneyline": ml_test["home_win"].tolist(),
        "totals": tot_test["went_over"].tolist(),
    }
    y_prob_map = {
        "moneyline": ml_test["pred_prob"].tolist(),
        "totals": tot_test["pred_prob_over"].tolist(),
    }
    dashboard = generate_dashboard_data(backtest_results, y_true_map, y_prob_map)

    print("\nLeaderboard:")
    for row in dashboard["leaderboard"]:
        print(
            f"  {row['model']:25s}  bets={row['total_bets']:4d}  "
            f"win_rate={row['win_rate']:.1%}  roi={row['roi']:+.1%}  "
            f"units={row['total_units']:+.1f}"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
