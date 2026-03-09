"""Monte Carlo feature selection for Savant leaderboard columns.

Randomly samples subsets of Savant-derived features, trains lightweight
models for each of the three bet targets (moneyline, spread, totals),
and records which features appear most often in top-performing combos.

The output is a ranked list of features by "selection frequency in top-N
trials" — a practical way to find the best combination without exhaustive
search over 100+ columns.

Usage:
    python scripts/monte_carlo_features.py
    python scripts/monte_carlo_features.py --trials 2000 --top-pct 10
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_files" / "raw"
RESULTS_DIR = ROOT / "data_files" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Targets we evaluate against ──────────────────────────────────────────
TARGETS = {
    "moneyline": "home_win",
    "spread": "home_cover",
    "totals": "went_over",
}

# ── Savant columns eligible for Monte Carlo sampling ─────────────────────
# We exclude identifiers, counts (low signal-to-noise), and highly
# correlated duplicates.  Rate stats are preferred over raw counts.

BATTER_CANDIDATE_FEATURES: list[str] = [
    # Expected stats
    "xba", "xslg", "xwoba", "xobp", "xiso",
    "wobacon", "xwobacon", "bacon", "xbacon",
    "xbadiff", "xslgdiff", "wobadiff",
    # Bat tracking
    "avg_swing_speed", "fast_swing_rate",
    "squared_up_contact", "squared_up_swing",
    "avg_swing_length", "swords",
    "attack_angle", "ideal_angle_rate",
    # Batted ball quality
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "barrel_batted_rate", "solidcontact_percent",
    "flareburner_percent", "poorlyunder_percent",
    "poorlytopped_percent", "poorlyweak_percent",
    "hard_hit_percent", "avg_best_speed", "avg_hyper_speed",
    # Plate discipline
    "k_percent", "bb_percent",
    "z_swing_percent", "z_swing_miss_percent",
    "oz_swing_percent", "oz_swing_miss_percent",
    "oz_contact_percent", "iz_contact_percent",
    "meatball_swing_percent", "meatball_percent",
    "edge_percent", "whiff_percent", "swing_percent",
    "f_strike_percent",
    # Distribution
    "pull_percent", "straightaway_percent", "opposite_percent",
    "groundballs_percent", "flyballs_percent",
    "linedrives_percent", "popups_percent",
    # Speed
    "sprint_speed", "hp_to_1b",
]

PITCHER_CANDIDATE_FEATURES: list[str] = [
    # Expected stats (pitcher-allowed)
    "xba", "xslg", "xwoba", "xobp", "xiso", "xera",
    "wobacon", "xwobacon", "xbadiff", "xslgdiff", "wobadiff",
    # Batted ball quality allowed
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "barrel_batted_rate", "solidcontact_percent",
    "hard_hit_percent", "avg_best_speed", "avg_hyper_speed",
    # Plate discipline induced
    "k_percent", "bb_percent",
    "z_swing_percent", "z_swing_miss_percent",
    "oz_swing_percent", "oz_swing_miss_percent",
    "oz_contact_percent", "iz_contact_percent",
    "meatball_percent", "edge_percent",
    "whiff_percent", "swing_percent", "f_strike_percent",
    # Distribution
    "groundballs_percent", "flyballs_percent",
    "linedrives_percent", "popups_percent",
    # Arsenal
    "velocity", "ff_avg_speed", "ff_avg_spin",
    "ff_avg_break_x", "ff_avg_break_z",
    "sl_avg_speed", "sl_avg_spin",
    "ch_avg_speed", "ch_avg_spin",
    "cu_avg_speed", "cu_avg_spin",
    "release_extension", "arm_angle",
]


# ── Data loading & merging ───────────────────────────────────────────────

# Retrosheet 3-letter franchise codes → full team name used by build_model_features().
# The "team" column in batting.parquet / gameinfo.parquet uses these codes; the
# processed standings/pitching/batting parquets use the full names below.
RETRO_TO_TEAM_NAME: dict[str, str] = {
    "ANA": "Angels",
    "ARI": "Diamondbacks",
    "ATL": "Braves",
    "BAL": "Orioles",
    "BOS": "Red Sox",
    "CHA": "White Sox",
    "CHN": "Cubs",
    "CIN": "Reds",
    "CLE": "Guardians",   # processed data uses Guardians for all years
    "COL": "Rockies",
    "DET": "Tigers",
    "HOU": "Astros",
    "KCA": "Royals",
    "LAN": "Dodgers",
    "MIA": "Marlins",
    "MIL": "Brewers",
    "MIN": "Twins",
    "NYA": "Yankees",
    "NYN": "Mets",
    "OAK": "Athletics",
    "ATH": "Athletics",   # Sacramento Athletics (2025+)
    "PHI": "Phillies",
    "PIT": "Pirates",
    "SDN": "Padres",
    "SEA": "Mariners",
    "SFN": "Giants",
    "SLN": "Cardinals",
    "TBA": "Rays",
    "TEX": "Rangers",
    "TOR": "Blue Jays",
    "WAS": "Nationals",
}

_PLAYER_TEAM_MAP: pd.DataFrame | None = None  # cached after first build


def _build_player_team_map() -> pd.DataFrame:
    """Build (mlbam_id, year, team) map from Retrosheet + Chadwick register.

    Retrosheet batting/pitching parquets have (id=retrosheet_id, team, date).
    The Chadwick register maps key_retro → key_mlbam.
    Combining them gives us MLBAM player_id → Retrosheet team code per season.
    """
    global _PLAYER_TEAM_MAP
    if _PLAYER_TEAM_MAP is not None:
        return _PLAYER_TEAM_MAP

    logger.info("Building player → team map from Retrosheet + Chadwick...")
    retro_dir = ROOT / "data_files" / "retrosheet"

    # Pull retrosheet IDs + teams from batting AND pitching parquets
    frames = []
    for fname in ("batting.parquet", "pitching.parquet"):
        fpath = retro_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath, columns=["id", "team", "date"])
        df["season"] = df["date"].astype(str).str[:4].astype(int, errors="ignore")
        df = df[df["season"] >= 2020][["id", "team", "season"]].drop_duplicates()
        frames.append(df)

    if not frames:
        logger.warning("No Retrosheet parquets found; skipping player-team map.")
        _PLAYER_TEAM_MAP = pd.DataFrame(columns=["player_id", "year", "team"])
        return _PLAYER_TEAM_MAP

    retro_players = pd.concat(frames, ignore_index=True).drop_duplicates()
    retro_players.columns = ["retro_id", "team", "season"]

    # Load chadwick register to map retro_id → mlbam_id
    from pybaseball import chadwick_register
    chad = chadwick_register()[["key_retro", "key_mlbam"]].dropna(subset=["key_retro", "key_mlbam"])
    chad["key_mlbam"] = chad["key_mlbam"].astype(int)

    merged = retro_players.merge(chad, left_on="retro_id", right_on="key_retro", how="inner")
    result = merged[["key_mlbam", "season", "team"]].rename(
        columns={"key_mlbam": "player_id", "season": "year"}
    ).drop_duplicates()

    # Translate Retrosheet 3-letter codes → full team names used by build_model_features()
    result["team"] = result["team"].map(RETRO_TO_TEAM_NAME).fillna(result["team"])

    logger.info("  Player-team map: %d unique (player, year, team) rows", len(result))
    _PLAYER_TEAM_MAP = result
    return result


def _load_savant_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2020-2025 Savant CSVs and attach Retrosheet team codes."""
    bat_path = RAW_DIR / "batting" / "savant_batter_2020_2025.csv"
    pit_path = RAW_DIR / "pitching" / "savant_pitcher_2020_2025.csv"

    if not bat_path.exists() or not pit_path.exists():
        raise FileNotFoundError(
            "Run scripts/fetch_savant_leaderboards.py first to download CSVs."
        )

    bat = pd.read_csv(bat_path)
    pit = pd.read_csv(pit_path)

    # Normalise column names (strip quotes/whitespace from Savant header)
    for df in (bat, pit):
        df.columns = [c.strip().strip('"').strip() for c in df.columns]
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Attach Retrosheet team codes via chadwick cross-reference
    team_map = _build_player_team_map()
    if not team_map.empty:
        bat = bat.merge(team_map, on=["player_id", "year"], how="left")
        pit = pit.merge(team_map, on=["player_id", "year"], how="left")
        logger.info(
            "Team join coverage — batters: %.1f%%  pitchers: %.1f%%",
            bat["team"].notna().mean() * 100,
            pit["team"].notna().mean() * 100,
        )
    else:
        bat["team"] = np.nan
        pit["team"] = np.nan

    return bat, pit


def _load_game_data() -> pd.DataFrame:
    """Load the game-level feature matrix from Retrosheet.

    Re-uses the existing build_model_features() which produces:
      - team/SP features from Retrosheet
      - targets: home_win, home_cover, went_over
    """
    from src.models.features import build_model_features
    return build_model_features(2020, 2025)


def _aggregate_team_season(
    player_df: pd.DataFrame,
    candidate_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """PA-weighted average of player stats → one row per (season, team).

    Returns columns like   ``{prefix}_{col}``  for each candidate col present.
    The ``team`` column must already be attached by ``_load_savant_data()``.
    """
    if "team" not in player_df.columns or player_df["team"].isna().all():
        logger.warning("No usable team column in Savant data; skipping aggregation.")
        return pd.DataFrame()

    available = [c for c in candidate_cols if c in player_df.columns]
    if not available:
        return pd.DataFrame()

    # Drop rows without a team assignment for aggregation
    df = player_df.dropna(subset=["team"])

    records = []
    for (year, team), grp in df.groupby(["year", "team"]):
        pa = grp["pa"] if "pa" in grp.columns else pd.Series(1.0, index=grp.index)
        row: dict = {"season": int(year), "team": team}
        for col in available:
            valid = grp.dropna(subset=[col])
            if len(valid) > 0:
                vpa = pa.loc[valid.index]
                denom = vpa.sum()
                row[f"{prefix}_{col}"] = (valid[col] * vpa).sum() / denom if denom > 0 else np.nan
            else:
                row[f"{prefix}_{col}"] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def build_enriched_features(
    games: pd.DataFrame,
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    bat_cols: list[str],
    pit_cols: list[str],
) -> pd.DataFrame:
    """Join Savant team-level aggregates onto Retrosheet game rows.

    For each game we attach:
      home_bat_{col}, away_bat_{col}   – team batting Savant stats
      home_pit_{col}, away_pit_{col}   – starting pitcher Savant stats

    Pitcher stats are joined at the individual level (the SP for that game)
    rather than team-level where possible.
    """
    # Team-level batting aggregates
    bat_agg = _aggregate_team_season(bat_df, bat_cols, "bat")

    # Pitcher aggregates (team-level for now; SP-level join planned later)
    pit_agg = _aggregate_team_season(pit_df, pit_cols, "pit")

    enriched = games.copy()

    # Join home batting
    if not bat_agg.empty:
        home_bat = bat_agg.rename(columns={"team": "hometeam"})
        home_bat = home_bat.rename(columns={
            c: f"home_{c}" for c in home_bat.columns if c.startswith("bat_")
        })
        enriched = enriched.merge(home_bat, on=["season", "hometeam"], how="left")

        away_bat = bat_agg.rename(columns={"team": "visteam"})
        away_bat = away_bat.rename(columns={
            c: f"away_{c}" for c in away_bat.columns if c.startswith("bat_")
        })
        enriched = enriched.merge(away_bat, on=["season", "visteam"], how="left")

    # Join home/away pitching
    if not pit_agg.empty:
        home_pit = pit_agg.rename(columns={"team": "hometeam"})
        home_pit = home_pit.rename(columns={
            c: f"home_{c}" for c in home_pit.columns if c.startswith("pit_")
        })
        enriched = enriched.merge(home_pit, on=["season", "hometeam"], how="left")

        away_pit = pit_agg.rename(columns={"team": "visteam"})
        away_pit = away_pit.rename(columns={
            c: f"away_{c}" for c in away_pit.columns if c.startswith("pit_")
        })
        enriched = enriched.merge(away_pit, on=["season", "visteam"], how="left")

    return enriched


# ── Monte Carlo trial engine ─────────────────────────────────────────────

def _run_single_trial(
    games: pd.DataFrame,
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    baseline_features: list[str],
    rng: np.random.Generator,
    n_bat: int = 6,
    n_pit: int = 4,
) -> dict:
    """Run one random feature-subset trial across all 3 targets.

    Steps:
      1. Sample n_bat batter cols + n_pit pitcher cols at random.
      2. Build enriched features (baseline Retrosheet + sampled Savant).
      3. Evaluate with 3-fold TimeSeriesSplit for each target.
      4. Return {target: roc_auc} + the column lists.
    """
    # Sample random subsets
    bat_available = [c for c in BATTER_CANDIDATE_FEATURES if c in bat_df.columns]
    pit_available = [c for c in PITCHER_CANDIDATE_FEATURES if c in pit_df.columns]

    n_bat_actual = min(n_bat, len(bat_available))
    n_pit_actual = min(n_pit, len(pit_available))

    bat_sample = list(rng.choice(bat_available, size=n_bat_actual, replace=False))
    pit_sample = list(rng.choice(pit_available, size=n_pit_actual, replace=False))

    # Build the enriched dataset
    enriched = build_enriched_features(games, bat_df, pit_df, bat_sample, pit_sample)

    # Savant feature column names in the enriched frame
    savant_cols = []
    for col in bat_sample:
        savant_cols.extend([f"home_bat_{col}", f"away_bat_{col}"])
    for col in pit_sample:
        savant_cols.extend([f"home_pit_{col}", f"away_pit_{col}"])

    # Only keep Savant columns that actually got joined
    savant_cols = [c for c in savant_cols if c in enriched.columns]
    if not savant_cols:
        return {}

    all_features = baseline_features + savant_cols

    # Drop rows with missing targets or features
    subset = enriched.dropna(subset=all_features + list(TARGETS.values()), how="any")
    if len(subset) < 500:
        return {}

    X = subset[all_features].values
    results: dict = {
        "bat_cols": bat_sample,
        "pit_cols": pit_sample,
        "n_rows": len(subset),
    }

    # TimeSeriesSplit evaluation for each target
    tscv = TimeSeriesSplit(n_splits=3)
    for model_name, target_col in TARGETS.items():
        y = subset[target_col].values
        aucs = []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("xgb", XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )),
            ])
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]

            if len(np.unique(y_te)) > 1:
                aucs.append(roc_auc_score(y_te, proba))

        results[f"{model_name}_auc"] = float(np.mean(aucs)) if aucs else 0.0

    return results


def run_monte_carlo(
    n_trials: int = 500,
    top_pct: float = 10.0,
    n_bat: int = 6,
    n_pit: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full Monte Carlo feature search.

    Args:
        n_trials: Number of random trials.
        top_pct: Keep the top X% of trials by mean ROC-AUC.
        n_bat: Number of batter Savant columns to sample per trial.
        n_pit: Number of pitcher Savant columns to sample per trial.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with one row per feature, ranked by frequency in top trials.
    """
    rng = np.random.default_rng(seed)

    logger.info("Loading game data (Retrosheet)...")
    games = _load_game_data()

    logger.info("Loading Savant leaderboards...")
    bat_df, pit_df = _load_savant_data()

    # Baseline features from the existing Retrosheet pipeline
    from src.models.features import ALL_FEATURE_COLS
    baseline_features = [c for c in ALL_FEATURE_COLS if c in games.columns]

    logger.info(
        "Running %d Monte Carlo trials  (sample %d bat + %d pit cols each)...",
        n_trials, n_bat, n_pit,
    )

    all_results = []
    for i in range(n_trials):
        if (i + 1) % 50 == 0:
            logger.info("  Trial %d / %d", i + 1, n_trials)

        result = _run_single_trial(
            games, bat_df, pit_df, baseline_features, rng, n_bat, n_pit,
        )
        if result:
            all_results.append(result)

    if not all_results:
        logger.error("No valid trials completed. Check data availability.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Mean AUC across all 3 targets
    auc_cols = [c for c in results_df.columns if c.endswith("_auc")]
    results_df["mean_auc"] = results_df[auc_cols].mean(axis=1)

    # Identify top trials
    cutoff = np.percentile(results_df["mean_auc"], 100 - top_pct)
    top_trials = results_df[results_df["mean_auc"] >= cutoff]

    logger.info(
        "Top %d%% cutoff: mean_auc >= %.4f  (%d trials)",
        int(top_pct), cutoff, len(top_trials),
    )

    # Count how often each feature appears in top trials
    bat_counter: Counter = Counter()
    pit_counter: Counter = Counter()
    for _, row in top_trials.iterrows():
        for col in row["bat_cols"]:
            bat_counter[col] += 1
        for col in row["pit_cols"]:
            pit_counter[col] += 1

    # Build ranked summary
    rows = []
    for col, count in bat_counter.most_common():
        rows.append({
            "feature": col,
            "type": "batter",
            "top_trial_appearances": count,
            "appearance_rate": count / len(top_trials),
        })
    for col, count in pit_counter.most_common():
        rows.append({
            "feature": col,
            "type": "pitcher",
            "top_trial_appearances": count,
            "appearance_rate": count / len(top_trials),
        })

    ranking_df = pd.DataFrame(rows).sort_values(
        "appearance_rate", ascending=False,
    ).reset_index(drop=True)

    # ── Summary stats ────────────────────────────────────────────────────
    logger.info("\n=== MONTE CARLO FEATURE RANKING ===")
    logger.info("Trials: %d  |  Valid: %d  |  Top %.0f%%: %d",
                n_trials, len(results_df), top_pct, len(top_trials))
    logger.info("Baseline AUC (no Savant):  moneyline=%.4f  spread=%.4f  totals=%.4f",
                *_baseline_auc(games, baseline_features))
    for target in TARGETS:
        col = f"{target}_auc"
        logger.info(
            "  %s — median: %.4f  top-10%%: %.4f",
            target, results_df[col].median(), top_trials[col].median(),
        )

    logger.info("\nTop 20 features by appearance in top trials:")
    for _, r in ranking_df.head(20).iterrows():
        logger.info(
            "  %-30s  %s  rate=%.1f%%",
            r["feature"], r["type"], r["appearance_rate"] * 100,
        )

    # Save results
    results_path = RESULTS_DIR / "mc_feature_trials.parquet"
    results_df.to_parquet(results_path, index=False)
    ranking_path = RESULTS_DIR / "mc_feature_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logger.info("\nResults saved to:\n  %s\n  %s", results_path, ranking_path)

    return ranking_df


def _baseline_auc(
    games: pd.DataFrame,
    baseline_features: list[str],
) -> tuple[float, float, float]:
    """Quick baseline AUC with existing Retrosheet features only."""
    subset = games.dropna(
        subset=baseline_features + list(TARGETS.values()), how="any",
    )
    X = subset[baseline_features].values
    tscv = TimeSeriesSplit(n_splits=3)
    aucs = {}
    for model_name, target_col in TARGETS.items():
        y = subset[target_col].values
        fold_aucs = []
        for train_idx, test_idx in tscv.split(X):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("xgb", XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, random_state=42,
                )),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                fold_aucs.append(roc_auc_score(y[test_idx], proba))
        aucs[model_name] = float(np.mean(fold_aucs)) if fold_aucs else 0.0

    return aucs["moneyline"], aucs["spread"], aucs["totals"]


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo feature selection for Savant columns",
    )
    parser.add_argument("--trials", type=int, default=500, help="Number of random trials")
    parser.add_argument("--top-pct", type=float, default=10.0, help="Top percent to analyse")
    parser.add_argument("--n-bat", type=int, default=6, help="Batter features per trial")
    parser.add_argument("--n-pit", type=int, default=4, help="Pitcher features per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ranking = run_monte_carlo(
        n_trials=args.trials,
        top_pct=args.top_pct,
        n_bat=args.n_bat,
        n_pit=args.n_pit,
        seed=args.seed,
    )

    if not ranking.empty:
        print("\n\nFinal Feature Ranking:")
        print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
