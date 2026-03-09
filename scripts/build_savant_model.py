"""Build Savant-enriched ML models from Monte Carlo feature rankings.

Reads mc_feature_ranking.csv, selects the top-N batter and pitcher Savant
columns, merges them as PA-weighted team-season aggregates onto the existing
Retrosheet game-level feature matrix, then trains and saves all three
betting models (moneyline, run line, totals).

Outputs written to data_files/processed/:
    savant_model_metrics.parquet
    savant_model_importances.parquet
    moneyline_savant_test_df.parquet
    spread_savant_test_df.parquet
    totals_savant_test_df.parquet

Model joblib files are written to models/ alongside the baseline models.

Usage:
    python scripts/build_savant_model.py
    python scripts/build_savant_model.py --n-bat 8 --n-pit 6
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

PROCESSED = ROOT / "data_files" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Baseline AUC from the Retrosheet-only models (used for delta display)
BASELINE_AUC = {
    "moneyline": 0.6253,
    "spread":    0.6304,
    "totals":    0.6157,
}


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_savant_enriched_features(
    n_bat: int = 8,
    n_pit: int = 6,
    min_year: int = 2020,
    max_year: int = 2025,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Combine Retrosheet game features with top Savant aggregates.

    Steps:
      1. Read mc_feature_ranking.csv and pick the top n_bat / n_pit columns.
      2. Use _load_savant_data() (from monte_carlo_features) to get batter +
         pitcher DataFrames with the Retrosheet team code already attached.
      3. Aggregate to team-season level (PA-weighted mean).
      4. Merge home & away flavours onto build_model_features() output.

    Returns:
        enriched_df   — game-level DataFrame with all baseline + Savant cols
        bat_cols_used — list of batter Savant column names that were merged
        pit_cols_used — list of pitcher Savant column names that were merged
    """
    ranking_path = PROCESSED / "mc_feature_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            "mc_feature_ranking.csv not found. "
            "Run 'python scripts/monte_carlo_features.py --trials 1000' first."
        )

    ranking = pd.read_csv(ranking_path)
    top_bat = ranking[ranking["type"] == "batter"].head(n_bat)["feature"].tolist()
    top_pit = ranking[ranking["type"] == "pitcher"].head(n_pit)["feature"].tolist()

    print(f"  Top {n_bat} batter features : {top_bat}")
    print(f"  Top {n_pit} pitcher features: {top_pit}")

    # Reuse helpers from monte_carlo_features (already tested + working)
    from scripts.monte_carlo_features import _load_savant_data, _aggregate_team_season

    print("Loading base Retrosheet game features...")
    from src.models.features import build_model_features
    games = build_model_features(min_year, max_year)

    print("Loading Savant CSVs and building team-season aggregates...")
    bat_df, pit_df = _load_savant_data()

    bat_agg = _aggregate_team_season(bat_df, top_bat, "bat")
    pit_agg = _aggregate_team_season(pit_df, top_pit, "pit")

    enriched = games.copy()
    bat_cols_used: list[str] = []
    pit_cols_used: list[str] = []

    if not bat_agg.empty:
        bat_cols_used = [c for c in top_bat if f"bat_{c}" in bat_agg.columns]

        home_bat = bat_agg.rename(columns={"team": "hometeam"}).rename(
            columns={f"bat_{c}": f"home_bat_{c}" for c in bat_cols_used}
        )
        away_bat = bat_agg.rename(columns={"team": "visteam"}).rename(
            columns={f"bat_{c}": f"away_bat_{c}" for c in bat_cols_used}
        )
        home_cols = ["season", "hometeam"] + [f"home_bat_{c}" for c in bat_cols_used]
        away_cols = ["season", "visteam"]  + [f"away_bat_{c}" for c in bat_cols_used]
        enriched = (
            enriched
            .merge(home_bat[home_cols], on=["season", "hometeam"], how="left")
            .merge(away_bat[away_cols], on=["season", "visteam"],  how="left")
        )

    if not pit_agg.empty:
        pit_cols_used = [c for c in top_pit if f"pit_{c}" in pit_agg.columns]

        home_pit = pit_agg.rename(columns={"team": "hometeam"}).rename(
            columns={f"pit_{c}": f"home_pit_{c}" for c in pit_cols_used}
        )
        away_pit = pit_agg.rename(columns={"team": "visteam"}).rename(
            columns={f"pit_{c}": f"away_pit_{c}" for c in pit_cols_used}
        )
        home_cols = ["season", "hometeam"] + [f"home_pit_{c}" for c in pit_cols_used]
        away_cols = ["season", "visteam"]  + [f"away_pit_{c}" for c in pit_cols_used]
        enriched = (
            enriched
            .merge(home_pit[home_cols], on=["season", "hometeam"], how="left")
            .merge(away_pit[away_cols], on=["season", "visteam"],  how="left")
        )

    n_rows_with_savant = enriched[
        [f"home_bat_{c}" for c in bat_cols_used[:1]]
    ].notna().all(axis=1).sum() if bat_cols_used else 0
    print(f"  Game rows with Savant data: {n_rows_with_savant:,} / {len(enriched):,}")

    return enriched, bat_cols_used, pit_cols_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_bat: int = 8, n_pit: int = 6) -> None:
    from src.models.features import MONEYLINE_FEATURES, SPREAD_FEATURES, TOTALS_FEATURES
    from src.models.underdog_model import train_moneyline_model
    from src.models.spread_model import train_spread_model
    from src.models.totals_model import train_totals_model

    print(f"\n=== Building Savant-enriched models (top {n_bat} bat + {n_pit} pit) ===\n")

    enriched, bat_cols, pit_cols = build_savant_enriched_features(n_bat, n_pit)

    # Build Savant column names as they appear in the enriched game DataFrame
    savant_game_cols = (
        [f"home_bat_{c}" for c in bat_cols] +
        [f"away_bat_{c}" for c in bat_cols] +
        [f"home_pit_{c}" for c in pit_cols] +
        [f"away_pit_{c}" for c in pit_cols]
    )
    savant_game_cols = [c for c in savant_game_cols if c in enriched.columns]

    ml_features = [c for c in MONEYLINE_FEATURES + savant_game_cols if c in enriched.columns]
    sp_features = [c for c in SPREAD_FEATURES    + savant_game_cols if c in enriched.columns]
    ou_features = [c for c in TOTALS_FEATURES    + savant_game_cols if c in enriched.columns]

    print(f"  Feature counts — ML: {len(ml_features)}  SP: {len(sp_features)}  OU: {len(ou_features)}")

    print("\nTraining moneyline model (Savant-enriched)...")
    r_ml = train_moneyline_model(enriched, feature_cols=ml_features)
    delta_ml = r_ml["metrics"]["roc_auc"] - BASELINE_AUC["moneyline"]
    print(f"  AUC {r_ml['metrics']['roc_auc']:.4f}  ({delta_ml:+.4f} vs baseline {BASELINE_AUC['moneyline']:.4f})")

    print("Training spread model (Savant-enriched)...")
    r_sp = train_spread_model(enriched, feature_cols=sp_features)
    delta_sp = r_sp["metrics"]["roc_auc"] - BASELINE_AUC["spread"]
    print(f"  AUC {r_sp['metrics']['roc_auc']:.4f}  ({delta_sp:+.4f} vs baseline {BASELINE_AUC['spread']:.4f})")

    print("Training totals model (Savant-enriched)...")
    r_ou = train_totals_model(enriched, feature_cols=ou_features)
    delta_ou = r_ou["metrics"]["roc_auc"] - BASELINE_AUC["totals"]
    print(f"  AUC {r_ou['metrics']['roc_auc']:.4f}  ({delta_ou:+.4f} vs baseline {BASELINE_AUC['totals']:.4f})")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_rows = []
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        baseline = BASELINE_AUC[name]
        row = {
            "model":                name,
            "train_size":           r["train_size"],
            "test_size":            r["test_size"],
            "n_bat_features":       n_bat,
            "n_pit_features":       n_pit,
            "savant_bat_features":  ",".join(bat_cols),
            "savant_pit_features":  ",".join(pit_cols),
            "baseline_roc_auc":     baseline,
            "roc_auc_delta":        r["metrics"]["roc_auc"] - baseline,
        }
        row.update(r["metrics"])
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_parquet(PROCESSED / "savant_model_metrics.parquet", index=False)
    print("\n  Saved savant_model_metrics.parquet")

    # ── Save importances ──────────────────────────────────────────────────
    imps = []
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        df_i = r["importances"].copy()
        df_i["model"] = name
        imps.append(df_i)
    pd.concat(imps, ignore_index=True).to_parquet(
        PROCESSED / "savant_model_importances.parquet", index=False
    )
    print("  Saved savant_model_importances.parquet")

    # ── Save test DataFrames ───────────────────────────────────────────────
    for name, r in [("moneyline", r_ml), ("spread", r_sp), ("totals", r_ou)]:
        r["test_df"].to_parquet(PROCESSED / f"{name}_savant_test_df.parquet", index=False)
    print("  Saved per-model savant test DataFrames")

    print("\n=== Done. Savant-enriched models saved. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Savant-enriched betting models")
    parser.add_argument("--n-bat", type=int, default=8, help="Top N batter Savant features to use")
    parser.add_argument("--n-pit", type=int, default=6, help="Top N pitcher Savant features to use")
    args = parser.parse_args()
    main(n_bat=args.n_bat, n_pit=args.n_pit)
