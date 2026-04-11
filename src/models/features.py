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
from datetime import date as _date

_CUR_YEAR = _date.today().year

# Allow running this file directly (e.g. python src/models/features.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrosheet import (
    load_gameinfo,
    season_standings,
    season_team_pitching,
    season_team_batting,
)
from src.models.extra_features import (
    rest_days_features,
    fielding_features,
    kb_rate_features,
    lob_features,
    weather_interaction_features,
    umpire_features,
    umpire_position_features,
    pythagorean_diff_features,
    baserunning_features,
    bullpen_fatigue_features,
    sp_vs_opp_features,
    daynight_split_features,
    platoon_features,
    # New: docs/14 recommendations
    team_consistency,
    woba_team_features,
    fip_sp_features,
    savant_team_features,
    savant_sp_features,
    park_factor_features,
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
    # extra: pythagorean diff (7.1)
    "home_pyth_diff", "away_pyth_diff",
    # extra: wOBA with year-specific FG weights (priority 1/13)
    "home_team_wOBA", "away_team_wOBA",
    # extra: team scoring consistency (priority 4)
    "home_con_r", "away_con_r", "home_con_ra", "away_con_ra",
    # extra: Savant team batting quality (priority 3, 7, 11, 12) — optional
    "home_team_barrel_pct", "away_team_barrel_pct",
    "home_team_exit_velo",  "away_team_exit_velo",
    "home_team_sprint_speed", "away_team_sprint_speed",
    "home_team_oaa",        "away_team_oaa",
    "home_team_xwoba",      "away_team_xwoba",
    "home_team_xwoba_diff", "away_team_xwoba_diff",
]

_SP_FEATURES = [
    "home_sp_ERA", "away_sp_ERA", "sp_ERA_gap",
    "home_sp_WHIP", "away_sp_WHIP",
    "home_sp_K9", "away_sp_K9",
    # extra: SP vs opponent history (2.3) — NaN when < 3 prior starts
    "home_sp_vs_opp_ERA", "away_sp_vs_opp_ERA",
    "home_sp_vs_opp_K9",  "away_sp_vs_opp_K9",
    # extra: FIP with year-specific cFIP (priority 5)
    "home_sp_FIP", "away_sp_FIP",
    # extra: Savant SP quality (priority 3, 8) — optional (NaN when not in CSV)
    "home_sp_xwoba",         "away_sp_xwoba",
    "home_sp_wobadiff",      "away_sp_wobadiff",
    "home_sp_barrel_allowed","away_sp_barrel_allowed",
    "home_sp_whiff_pct",     "away_sp_whiff_pct",
    "home_sp_edge_pct",      "away_sp_edge_pct",
]

_CONTEXT_FEATURES = [
    "temp", "windspeed", "is_day",
    # extra: weather interactions (6.2)
    "wind_out", "wind_in", "dome_flag", "temp_cold", "temp_hot", "overcast_flag",
    # extra: umpire (6.3)
    "ump_runs_avg", "ump_above_avg_flag",
    "ump_home_games", "ump_home_over_mean", "ump_home_trend",
    # extra: umpire positions (6.3b)
    "ump1b_runs_avg", "ump2b_runs_avg", "ump3b_runs_avg",
    # extra: rest days (3.3)
    "home_days_rest", "away_days_rest",
    "home_back_to_back", "away_back_to_back", "is_doubleheader",
]

_MATCHUP_FEATURES = [
    # extra: plate discipline (5.1)
    "home_K_rate", "away_K_rate", "home_BB_rate", "away_BB_rate",
    "home_K_BB_ratio", "away_K_BB_ratio",
    # extra: day/night splits (3.2)
    "home_day_WPct", "away_day_WPct", "home_night_WPct", "away_night_WPct",
    # extra: fielding (4.1)
    "home_errors_per_g", "away_errors_per_g", "home_dp_rate", "away_dp_rate",
    "home_def_efficiency", "away_def_efficiency",
    # extra: baserunning (7.2)
    "home_sb_success_rate", "away_sb_success_rate", "home_sb_rate", "away_sb_rate",
    # extra: bullpen fatigue (7.3)
    "home_bullpen_ip_3d", "away_bullpen_ip_3d",
    "home_pen_arms_3d",  "away_pen_arms_3d",
    # extra: platoon advantage (8.1)
    "home_platoon_adv", "away_platoon_adv", "platoon_adv_gap",
    # extra: matchup K delta (8.2 — derived below)
    "matchup_k_delta",
    # extra: handedness-split park factors (priority 6)
    "pf_basic_R", "pf_basic_L", "pf_hr_R", "pf_hr_L",
]

_TOTALS_EXTRA = [
    # extra: LOB (5.2)
    "home_lob_per_g", "away_lob_per_g",
]

MONEYLINE_FEATURES: list[str] = (
    _TEAM_FEATURES + _SP_FEATURES + _CONTEXT_FEATURES + _MATCHUP_FEATURES
)

SPREAD_FEATURES: list[str] = MONEYLINE_FEATURES  # same inputs, different target

TOTALS_FEATURES: list[str] = (
    _TEAM_FEATURES + _SP_FEATURES + _CONTEXT_FEATURES + _MATCHUP_FEATURES
    + _TOTALS_EXTRA + ["exp_total"]
)

ALL_FEATURE_COLS: list[str] = list(dict.fromkeys(
    TOTALS_FEATURES
))  # deduplicated superset


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

def build_model_features(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
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
    feat["exp_total"] = feat["home_RS_G"] + feat["away_RS_G"]

    # ── 8. Targets ───────────────────────────────────────────────────────────
    feat["home_win"]   = (feat["wteam"] == feat["hometeam"]).astype(int)
    feat["home_cover"] = ((feat["hruns"] - feat["vruns"]) >= 2).astype(int)
    feat["went_over"]  = (feat["total_runs"] > feat["exp_total"]).astype(int)

    # ── 9. Extra features (docs/11-feature-engineering-roadmap.md) ───────────

    # 3.3 Rest days / doubleheaders
    feat = feat.merge(rest_days_features(min_year, max_year), on="gid", how="left")

    # 4.1 Fielding quality
    _fld = fielding_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _fld.rename(columns={
            "team": team_col,
            "errors_per_g": f"{side}_errors_per_g",
            "dp_rate":       f"{side}_dp_rate",
            "def_efficiency": f"{side}_def_efficiency",
        })
        feat = feat.merge(tmp, on=["season", team_col], how="left")

    # 5.1 K/BB plate discipline
    _kb = kb_rate_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _kb.rename(columns={
            "team":      team_col,
            "K_rate":    f"{side}_K_rate",
            "BB_rate":   f"{side}_BB_rate",
            "K_BB_ratio": f"{side}_K_BB_ratio",
        })
        feat = feat.merge(tmp, on=["season", team_col], how="left")

    # 5.2 LOB per game
    _lob = lob_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _lob.rename(columns={"team": team_col, "lob_per_g": f"{side}_lob_per_g"})
        feat = feat.merge(tmp, on=["season", team_col], how="left")

    # 6.2 Weather interaction
    feat = feat.merge(weather_interaction_features(min_year, max_year), on="gid", how="left")
    for col in ("wind_out", "wind_in", "dome_flag", "temp_cold", "temp_hot", "overcast_flag"):
        feat[col] = feat[col].fillna(0.0)

    # 6.3 Umpire tendencies (plate umpire)
    feat = feat.merge(umpire_features(min_year, max_year), on="gid", how="left")
    league_runs = feat["ump_runs_avg"].median()
    feat["ump_runs_avg"]       = feat["ump_runs_avg"].fillna(league_runs)
    feat["ump_above_avg_flag"] = feat["ump_above_avg_flag"].fillna(0.0)
    feat["ump_home_games"]     = feat["ump_home_games"].fillna(0).astype(int)
    feat["ump_home_over_mean"] = feat["ump_home_over_mean"].fillna(0.0)
    feat["ump_home_trend"]     = feat["ump_home_trend"].fillna(0.0)

    # 6.3b Umpire position tendencies (1B/2B/3B)
    feat = feat.merge(umpire_position_features(min_year, max_year), on="gid", how="left")
    for _uc in ("ump1b_runs_avg", "ump2b_runs_avg", "ump3b_runs_avg"):
        feat[_uc] = feat[_uc].fillna(league_runs)

    # 7.1 Pythagorean differential
    _pd = pythagorean_diff_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _pd.rename(columns={"team": team_col, "pyth_diff": f"{side}_pyth_diff"})
        feat = feat.merge(tmp, on=["season", team_col], how="left")
    feat["home_pyth_diff"] = feat["home_pyth_diff"].fillna(0.0)
    feat["away_pyth_diff"] = feat["away_pyth_diff"].fillna(0.0)

    # 7.2 Base-running efficiency
    _br = baserunning_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _br.rename(columns={
            "team":           team_col,
            "sb_success_rate": f"{side}_sb_success_rate",
            "sb_rate":         f"{side}_sb_rate",
        })
        feat = feat.merge(tmp, on=["season", team_col], how="left")

    # 7.3 Bullpen workload & fatigue
    feat = feat.merge(bullpen_fatigue_features(min_year, max_year), on="gid", how="left")
    for col in ("home_bullpen_ip_3d", "away_bullpen_ip_3d",
                "home_pen_arms_3d",  "away_pen_arms_3d"):
        feat[col] = feat[col].fillna(0.0)

    # 2.3 SP vs. opponent history
    _svo = sp_vs_opp_features(min_year, max_year)
    feat = feat.merge(_svo, on="gid", how="left")
    # Fill with season SP ERA/K9 when no opponent history
    feat["home_sp_vs_opp_ERA"] = feat["home_sp_vs_opp_ERA"].fillna(feat["home_sp_ERA"])
    feat["away_sp_vs_opp_ERA"] = feat["away_sp_vs_opp_ERA"].fillna(feat["away_sp_ERA"])
    feat["home_sp_vs_opp_K9"]  = feat["home_sp_vs_opp_K9"].fillna(feat["home_sp_K9"])
    feat["away_sp_vs_opp_K9"]  = feat["away_sp_vs_opp_K9"].fillna(feat["away_sp_K9"])

    # 3.2 Day/night splits
    _dn = daynight_split_features(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _dn.rename(columns={
            "team":      team_col,
            "day_WPct":  f"{side}_day_WPct",
            "night_WPct": f"{side}_night_WPct",
        })
        feat = feat.merge(tmp, on=["season", team_col], how="left")
    # Fill with overall WPct when splits unavailable
    feat["home_day_WPct"]   = feat["home_day_WPct"].fillna(feat["home_WPct"])
    feat["away_day_WPct"]   = feat["away_day_WPct"].fillna(feat["away_WPct"])
    feat["home_night_WPct"] = feat["home_night_WPct"].fillna(feat["home_WPct"])
    feat["away_night_WPct"] = feat["away_night_WPct"].fillna(feat["away_WPct"])

    # 8.1 Platoon advantage
    feat = feat.merge(platoon_features(min_year, max_year), on="gid", how="left")
    feat["home_platoon_adv"] = feat["home_platoon_adv"].fillna(0.5)
    feat["away_platoon_adv"] = feat["away_platoon_adv"].fillna(0.5)
    feat["platoon_adv_gap"]  = feat["platoon_adv_gap"].fillna(0.0)

    # 8.2 Matchup K delta — high-K offense vs high-K pitching amplifies strikeouts
    feat["matchup_k_delta"] = (
        feat["home_K_rate"].fillna(0) - feat["away_K_rate"].fillna(0)
    ).round(4)

    # ── 10. New features from docs/14 recommendations ────────────────────────

    # Priority 4: Team scoring consistency (con_r / con_ra)
    _con = team_consistency(min_year, max_year)
    for side, team_col in (("home", "hometeam"), ("away", "visteam")):
        tmp = _con.rename(columns={
            "team":   team_col,
            "con_r":  f"{side}_con_r",
            "con_ra": f"{side}_con_ra",
        })
        feat = feat.merge(tmp, on=["season", team_col], how="left")
    feat["home_con_r"]  = feat["home_con_r"].fillna(0.5)
    feat["away_con_r"]  = feat["away_con_r"].fillna(0.5)
    feat["home_con_ra"] = feat["home_con_ra"].fillna(0.5)
    feat["away_con_ra"] = feat["away_con_ra"].fillna(0.5)

    # Priority 1/13: Team wOBA with year-specific FG weights
    _woba = woba_team_features(min_year, max_year)
    if not _woba.empty:
        for side, team_col in (("home", "hometeam"), ("away", "visteam")):
            tmp = _woba.rename(columns={"team": team_col, "team_wOBA": f"{side}_team_wOBA"})
            feat = feat.merge(tmp, on=["season", team_col], how="left")
        lg_woba = feat["home_team_wOBA"].median()
        feat["home_team_wOBA"] = feat["home_team_wOBA"].fillna(lg_woba)
        feat["away_team_wOBA"] = feat["away_team_wOBA"].fillna(lg_woba)

    # Priority 5: SP FIP with year-specific cFIP constant
    _fip = fip_sp_features(min_year, max_year)
    if not _fip.empty:
        feat = feat.merge(_fip, on="gid", how="left")
        feat["home_sp_FIP"] = feat["home_sp_FIP"].fillna(feat.get("home_sp_ERA", feat["home_ERA"]))
        feat["away_sp_FIP"] = feat["away_sp_FIP"].fillna(feat.get("away_sp_ERA", feat["away_ERA"]))

    # Priority 3, 7, 11, 12: Savant team batting quality (optional — no-op if CSV absent)
    _sv_team = savant_team_features(min_year, max_year)
    if not _sv_team.empty:
        for side, team_col in (("home", "hometeam"), ("away", "visteam")):
            sv_cols = [c for c in _sv_team.columns if c not in ("season", "team")]
            tmp_rename = {c: f"{side}_{c}" for c in sv_cols}
            tmp = _sv_team.rename(columns={"team": team_col, **tmp_rename})
            feat = feat.merge(tmp, on=["season", team_col], how="left")
        _sv_fill = {
            "home_team_barrel_pct": 8.0, "away_team_barrel_pct": 8.0,
            "home_team_exit_velo":  88.5, "away_team_exit_velo":  88.5,
            "home_team_sprint_speed": 27.0, "away_team_sprint_speed": 27.0,
            "home_team_oaa": 0.0, "away_team_oaa": 0.0,
            "home_team_xwoba": 0.320, "away_team_xwoba": 0.320,
            "home_team_xwoba_diff": 0.0, "away_team_xwoba_diff": 0.0,
        }
        for col, default in _sv_fill.items():
            if col in feat.columns:
                feat[col] = feat[col].fillna(default)

    # Priority 3, 8: Savant SP quality (optional — no-op if CSV absent)
    _sv_sp = savant_sp_features(min_year, max_year)
    if not _sv_sp.empty:
        feat = feat.merge(_sv_sp, on="gid", how="left")
        _sp_savant_cols = [
            "home_sp_xwoba", "away_sp_xwoba",
            "home_sp_wobadiff", "away_sp_wobadiff",
            "home_sp_barrel_allowed", "away_sp_barrel_allowed",
            "home_sp_whiff_pct", "away_sp_whiff_pct",
            "home_sp_edge_pct", "away_sp_edge_pct",
        ]
        _sp_savant_fill = {
            "home_sp_xwoba": 0.320, "away_sp_xwoba": 0.320,
            "home_sp_wobadiff": 0.0, "away_sp_wobadiff": 0.0,
            "home_sp_barrel_allowed": 8.0, "away_sp_barrel_allowed": 8.0,
            "home_sp_whiff_pct": 24.0, "away_sp_whiff_pct": 24.0,
            "home_sp_edge_pct": 43.0, "away_sp_edge_pct": 43.0,
        }
        for col, default in _sp_savant_fill.items():
            if col in feat.columns:
                feat[col] = feat[col].fillna(default)

    # Priority 6: Handedness-split park factors
    _pf = park_factor_features(min_year, max_year)
    if not _pf.empty:
        feat = feat.merge(_pf, on="gid", how="left")
    for pf_col in ("pf_basic_R", "pf_basic_L", "pf_hr_R", "pf_hr_L"):
        if pf_col not in feat.columns:
            feat[pf_col] = 100.0
        else:
            feat[pf_col] = feat[pf_col].fillna(100.0)

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
