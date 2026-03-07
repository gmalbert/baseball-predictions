"""Feature engineering pipeline shared by all three betting models.

Builds a game-level feature matrix from Retrosheet CSV data.
All three models (moneyline, spread, totals) draw from the same matrix.

Data sources used:
    gameinfo.csv   – one row per game; scores, weather, context
    teamstats.csv  – per-game batting/pitching lines aggregated to season
    pitching.csv   – individual pitcher lines; p_gs==1 identifies the starter

Important note on lookahead bias:
    We join same-season team and pitcher stats (full-season aggregates).
    This is intentional for a backtesting/analysis tool. A live-deployment
    version would instead use expanding stats computed through game_date - 1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running this file directly (e.g. python src/models/features.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrosheet import (
    load_gameinfo,
    season_standings,
    season_team_pitching,
    season_team_batting,
)

# ---------------------------------------------------------------------------
# Feature column lists used by each model
# ---------------------------------------------------------------------------

_TEAM_FEATURES = [
    "home_WPct", "away_WPct", "WPct_diff",
    "home_PythWPct", "away_PythWPct", "PythWPct_diff",
    "home_RS_G", "home_RA_G", "away_RS_G", "away_RA_G",
    "home_RD_G", "away_RD_G",
    # pitching
    "home_ERA", "away_ERA", "ERA_diff",
    "home_WHIP", "away_WHIP", "WHIP_diff",
    "home_K9", "away_K9",
    # batting
    "home_BA", "away_BA",
    "home_SLG", "away_SLG",
]

_SP_FEATURES = [
    "home_sp_ERA", "away_sp_ERA", "sp_ERA_gap",
    "home_sp_WHIP", "away_sp_WHIP",
    "home_sp_K9", "away_sp_K9",
]

_CONTEXT_FEATURES = [
    "temp", "windspeed", "is_day",
]

MONEYLINE_FEATURES: list[str] = (
    _TEAM_FEATURES + _SP_FEATURES + _CONTEXT_FEATURES
)

SPREAD_FEATURES: list[str] = MONEYLINE_FEATURES  # same inputs, different target

TOTALS_FEATURES: list[str] = (
    _TEAM_FEATURES + _SP_FEATURES + _CONTEXT_FEATURES + ["exp_total"]
)

ALL_FEATURE_COLS: list[str] = TOTALS_FEATURES  # superset


# ---------------------------------------------------------------------------
# Odds / edge utilities (used when real lines are available)
# ---------------------------------------------------------------------------

def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied (break-even) probability.

    Examples:
        +150  →  100 / (150 + 100)  = 0.400
        -110  →  110 / (110 + 100)  = 0.524
    """
    if american_odds >= 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if american_odds >= 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def calculate_edge(predicted_prob: float, american_odds: int) -> float:
    """Calculate betting edge: model probability minus implied probability.

    Positive edge means the model thinks the outcome is more likely than
    the sportsbook's line implies — a theoretically +EV bet.
    """
    return predicted_prob - implied_probability(american_odds)


# ---------------------------------------------------------------------------
# Starting-pitcher season stats per game
# ---------------------------------------------------------------------------

def _load_sp_season_stats(min_year: int, max_year: int) -> pd.DataFrame:
    """Return a DataFrame with home and away starting pitcher stats per game.

    Columns: gid, home_sp_ERA, home_sp_WHIP, home_sp_K9,
                  away_sp_ERA, away_sp_WHIP, away_sp_K9
    """
    raw_dir = Path(__file__).resolve().parents[2] / "data_files" / "retrosheet"

    needed_cols = [
        "gid", "id", "team", "stattype", "vishome", "gametype",
        "p_gs", "p_ipouts", "p_er", "p_k", "p_w", "p_h", "date",
    ]
    pitch = pd.read_parquet(raw_dir / "pitching.parquet")
    cols = [c for c in needed_cols if c in pitch.columns]
    pitch = pitch[cols]
    pitch = pitch[pitch["p_gs"] == 1.0].copy()

    # Parse date / season
    pitch["season"] = pd.to_numeric(
        pitch["date"].astype(str).str[:4], errors="coerce"
    )
    pitch = pitch[(pitch["season"] >= min_year) & (pitch["season"] <= max_year)]

    for c in ("p_ipouts", "p_er", "p_k", "p_w", "p_h"):
        pitch[c] = pd.to_numeric(pitch[c], errors="coerce")

    pitch["ip"] = pitch["p_ipouts"] / 3

    # Season aggregates per pitcher
    sp_agg = pitch.groupby(["season", "id"]).agg(
        total_ip=("ip", "sum"),
        total_er=("p_er", "sum"),
        total_k=("p_k", "sum"),
        total_bb=("p_w", "sum"),
        total_h=("p_h", "sum"),
    ).reset_index()
    ip_s = sp_agg["total_ip"].where(sp_agg["total_ip"] > 0)
    sp_agg["sp_ERA"] = (9 * sp_agg["total_er"] / ip_s).round(2)
    sp_agg["sp_WHIP"] = ((sp_agg["total_h"] + sp_agg["total_bb"]) / ip_s).round(3)
    sp_agg["sp_K9"] = (9 * sp_agg["total_k"] / ip_s).round(2)

    # Each game row now has vishome='h' (home starter) or 'v' (away starter)
    # Merge season stats back onto the game-level starter rows
    pitch = pitch.merge(
        sp_agg[["season", "id", "sp_ERA", "sp_WHIP", "sp_K9"]],
        on=["season", "id"],
        how="left",
    )

    home_sp = pitch[pitch["vishome"] == "h"][["gid", "sp_ERA", "sp_WHIP", "sp_K9"]].copy()
    home_sp.columns = ["gid", "home_sp_ERA", "home_sp_WHIP", "home_sp_K9"]

    away_sp = pitch[pitch["vishome"] == "v"][["gid", "sp_ERA", "sp_WHIP", "sp_K9"]].copy()
    away_sp.columns = ["gid", "away_sp_ERA", "away_sp_WHIP", "away_sp_K9"]

    # Some games have multiple rows per side (doubleheaders).
    # Keep first occurrence for each gid/role.
    home_sp = home_sp.drop_duplicates(subset="gid")
    away_sp = away_sp.drop_duplicates(subset="gid")

    sp_per_game = home_sp.merge(away_sp, on="gid", how="outer")
    sp_per_game["sp_ERA_gap"] = (
        sp_per_game["away_sp_ERA"] - sp_per_game["home_sp_ERA"]
    )
    return sp_per_game


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_model_features(min_year: int = 2020, max_year: int = 2025) -> pd.DataFrame:
    """Build the complete game-level feature matrix for ML models.

    One row per game. Includes:
      - Team season-level batting and pitching stats for home and away
      - Starting pitcher season-level ERA / WHIP / K9
      - Game context (temperature, wind, day/night)
      - Three binary target columns: home_win, home_cover, went_over

    Returns:
        DataFrame with ALL_FEATURE_COLS + target columns + identifiers.
    """
    # ── 1. Load raw game info ────────────────────────────────────────────────
    gi = load_gameinfo(min_year, max_year)

    # ── 2. Season-level team stats ───────────────────────────────────────────
    stnd = season_standings(min_year, max_year)[
        ["season", "team", "WPct", "PythWPct",
         "RS_per_G", "RA_per_G", "RD_per_G"]
    ]
    tpitch = season_team_pitching(min_year, max_year)[
        ["season", "team", "ERA", "WHIP", "K9"]
    ]
    tbat = season_team_batting(min_year, max_year)[
        ["season", "team", "BA", "SLG"]
    ]

    team_stats = (
        stnd
        .merge(tpitch, on=["season", "team"], how="left")
        .merge(tbat,   on=["season", "team"], how="left")
    )

    # ── 3. Join home team stats ──────────────────────────────────────────────
    home_stats = team_stats.rename(
        columns={
            "team": "hometeam",
            "WPct": "home_WPct", "PythWPct": "home_PythWPct",
            "RS_per_G": "home_RS_G", "RA_per_G": "home_RA_G",
            "RD_per_G": "home_RD_G",
            "ERA": "home_ERA", "WHIP": "home_WHIP", "K9": "home_K9",
            "BA": "home_BA", "SLG": "home_SLG",
        }
    )
    away_stats = team_stats.rename(
        columns={
            "team": "visteam",
            "WPct": "away_WPct", "PythWPct": "away_PythWPct",
            "RS_per_G": "away_RS_G", "RA_per_G": "away_RA_G",
            "RD_per_G": "away_RD_G",
            "ERA": "away_ERA", "WHIP": "away_WHIP", "K9": "away_K9",
            "BA": "away_BA", "SLG": "away_SLG",
        }
    )

    feat = (
        gi
        .merge(home_stats, on=["season", "hometeam"], how="left")
        .merge(away_stats, on=["season", "visteam"],  how="left")
    )

    # ── 4. Differential features ─────────────────────────────────────────────
    feat["WPct_diff"]     = feat["home_WPct"]     - feat["away_WPct"]
    feat["PythWPct_diff"] = feat["home_PythWPct"] - feat["away_PythWPct"]
    feat["RS_advantage"]  = feat["home_RS_G"]     - feat["away_RS_G"]
    feat["RA_advantage"]  = feat["away_RA_G"]     - feat["home_RA_G"]
    feat["ERA_diff"]      = feat["away_ERA"]       - feat["home_ERA"]
    feat["WHIP_diff"]     = feat["away_WHIP"]      - feat["home_WHIP"]

    # ── 5. Starting pitcher features ─────────────────────────────────────────
    sp_df = _load_sp_season_stats(min_year, max_year)
    feat = feat.merge(sp_df, on="gid", how="left")

    # Fill missing SP stats with team average (graceful fallback)
    for side, team_era, team_whip, team_k9 in [
        ("home", "home_ERA", "home_WHIP", "home_K9"),
        ("away", "away_ERA", "away_WHIP", "away_K9"),
    ]:
        feat[f"{side}_sp_ERA"]  = feat[f"{side}_sp_ERA"].fillna(feat[team_era])
        feat[f"{side}_sp_WHIP"] = feat[f"{side}_sp_WHIP"].fillna(feat[team_whip])
        feat[f"{side}_sp_K9"]   = feat[f"{side}_sp_K9"].fillna(feat[team_k9])

    feat["sp_ERA_gap"] = feat["away_sp_ERA"] - feat["home_sp_ERA"]

    # ── 6. Game context features ─────────────────────────────────────────────
    feat["temp"]      = pd.to_numeric(feat["temp"],      errors="coerce")
    feat["windspeed"] = pd.to_numeric(feat["windspeed"], errors="coerce")
    feat["is_day"]    = (feat["daynight"] == "d").astype(float)

    # Fill weather with neutral medians where missing
    feat["temp"]      = feat["temp"].fillna(feat["temp"].median())
    feat["windspeed"] = feat["windspeed"].fillna(0.0)

    # ── 7. Expected total (used as surrogate "posted total" for O/U model) ───
    # Simple estimate: average of both teams' runs-per-game scored
    feat["exp_total"] = feat["home_RS_G"] + feat["away_RS_G"]

    # ── 8. Targets ───────────────────────────────────────────────────────────
    feat["home_win"]   = (feat["wteam"] == feat["hometeam"]).astype(int)
    feat["home_cover"] = ((feat["hruns"] - feat["vruns"]) >= 2).astype(int)
    feat["went_over"]  = (feat["total_runs"] > feat["exp_total"]).astype(int)

    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Building feature matrix (2020–2025)…")
    df = build_model_features(2020, 2025)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTarget rates:")
    print(f"  home_win   = {df['home_win'].mean():.3f}")
    print(f"  home_cover = {df['home_cover'].mean():.3f}")
    print(f"  went_over  = {df['went_over'].mean():.3f}")
    missing = df[ALL_FEATURE_COLS].isnull().mean()
    worst = missing.nlargest(5)
    print(f"\nMissing % (top 5 cols):\n{worst}")
