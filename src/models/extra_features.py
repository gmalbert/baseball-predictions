"""Extra feature engineering functions for the Betting Cleanup MLB dashboard.

Each function returns a DataFrame suitable for merging into the main feature
matrix produced by :func:`features.build_model_features`.

New features implemented here (keyed to docs/11-feature-engineering-roadmap.md):
    2.3  SP vs. Opponent History
    3.2  Day/Night Splits
    3.3  Rest Days & Doubleheaders
    4.1  Team Fielding Quality
    5.1  K/BB Plate Discipline Rates
    5.2  LOB & Scoring Efficiency
    6.2  Weather Interaction Features
    6.3  Umpire Home-Plate Tendency
    6.3b Umpire Base Position Tendencies
    7.1  Pythagorean Win% Differential
    7.2  Base-Running Efficiency
    7.3  Bullpen Workload & Fatigue
    8.1  Platoon Advantage (handedness)
    8.2  Matchup K/BB Delta (derived in features.py after merging)

Data sources:
    data_files/retrosheet/gameinfo.parquet / gameinfo.csv
    data_files/retrosheet/teamstats.parquet / teamstats.csv
    data_files/retrosheet/pitching.parquet
    data_files/retrosheet/allplayers.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from retrosheet import season_standings, TEAM_NAMES  # noqa: E402


def _code_to_name(code: str) -> str:
    """Convert Retrosheet 3-letter team code to the canonical full name."""
    return TEAM_NAMES.get(str(code).upper(), code)

_RETRO = Path(__file__).resolve().parents[2] / "data_files" / "retrosheet"

# Retrosheet wind-direction codes (all lowercase)
# "To" = wind blowing toward outfield = ball carries = HR-friendly
_WIND_OUT_CODES = {"tolas", "tolf", "tocf", "torf", "out"}
# "From" = wind blowing in from outfield = suppresses HRs
_WIND_IN_CODES = {"froml", "fromc", "fromr", "fromlf", "fromcf", "fromrf", "in"}

# Known fully-enclosed / retractable-roof parks by Retrosheet site code
_DOME_SITES = {
    "PHO01",  # Chase Field (retractable)
    "HOU03",  # Minute Maid Park (retractable)
    "SEA03",  # T-Mobile Park (retractable)
    "MIL06",  # American Family Field (retractable)
    "TOR02",  # Rogers Centre (dome)
    "TAM01",  # Tropicana Field (dome)
    "RAY01",
    "MIA02",  # loanDepot park (retractable)
    "LAS01",  # Oakland/Sacramento (indoor)
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_teamstats(min_year: int, max_year: int) -> pd.DataFrame:
    """Load teamstats parquet with season + full team name (matches gameinfo naming)."""
    ts = pd.read_parquet(_RETRO / "teamstats.parquet")
    ts["season"] = (pd.to_numeric(ts["date"], errors="coerce") // 10000).astype("int64")
    for col in ts.columns:
        if col not in ("gid", "team", "stattype", "date", "vishome", "opp",
                       "gametype", "win", "loss", "tie"):
            ts[col] = pd.to_numeric(ts[col], errors="coerce")
    ts = ts[(ts["season"] >= min_year) & (ts["season"] <= max_year)].copy()
    ts["team_full"] = ts["team"].map(_code_to_name)
    return ts


def _load_teamstats_csv(min_year: int, max_year: int) -> pd.DataFrame:
    """Load teamstats for extra columns (lob etc.); reads parquet if CSV unavailable."""
    wanted = {
        "gid", "team", "stattype", "date", "vishome", "opp", "win", "loss",
        "lob", "d_po", "d_a", "d_e", "d_dp",
        "b_pa", "b_k", "b_w", "b_sb", "b_cs",
    }
    path_csv = _RETRO / "teamstats.csv"
    if path_csv.exists():
        ts = pd.read_csv(
            path_csv,
            usecols=lambda c: c in wanted,
            dtype=str,
            low_memory=False,
        )
    else:
        ts = pd.read_parquet(_RETRO / "teamstats.parquet")
        ts = ts[[c for c in ts.columns if c in wanted]]
    # 'stattype' is dropped from lean parquets (all rows are already 'value').
    if "stattype" in ts.columns:
        ts = ts[ts["stattype"] == "value"].copy()
    else:
        ts = ts.copy()
    for col in ts.columns:
        if col not in ("gid", "team", "stattype", "date", "vishome", "opp"):
            ts[col] = pd.to_numeric(ts[col], errors="coerce")
    ts["season"] = pd.to_numeric(ts["date"].astype(str).str[:4], errors="coerce")
    ts = ts[(ts["season"] >= min_year) & (ts["season"] <= max_year)].copy()
    ts["team_full"] = ts["team"].map(_code_to_name)
    return ts


def _load_gameinfo_csv(min_year: int, max_year: int,
                       extra_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Load gameinfo; reads parquet if CSV unavailable (CSV is gitignored, parquet is not)."""
    base_cols = ["gid", "visteam", "hometeam", "date", "number",
                 "daynight", "vruns", "hruns", "wteam", "season"]
    wanted = set(base_cols + (extra_cols or []))
    path_csv = _RETRO / "gameinfo.csv"
    if path_csv.exists():
        gi = pd.read_csv(
            path_csv,
            usecols=lambda c: c in wanted,
            dtype=str,
            low_memory=False,
        )
    else:
        gi = pd.read_parquet(_RETRO / "gameinfo.parquet")
        gi = gi[[c for c in gi.columns if c in wanted]]
    gi["season"] = pd.to_numeric(gi["season"], errors="coerce")
    gi["date"]   = pd.to_datetime(gi["date"].astype(str), format="%Y%m%d", errors="coerce")
    gi["number"] = pd.to_numeric(gi["number"], errors="coerce").fillna(0).astype(int)
    gi["vruns"]  = pd.to_numeric(gi.get("vruns"), errors="coerce")
    gi["hruns"]  = pd.to_numeric(gi.get("hruns"), errors="coerce")
    return gi[(gi["season"] >= min_year) & (gi["season"] <= max_year)].copy()


# ---------------------------------------------------------------------------
# 3.3 — Rest Days & Doubleheaders
# ---------------------------------------------------------------------------

def rest_days_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Days-of-rest and back-to-back flags for each team per game.

    Returns DataFrame with columns:
        gid, is_doubleheader,
        home_days_rest, home_back_to_back,
        away_days_rest, away_back_to_back
    """
    gi = _load_gameinfo_csv(min_year, max_year)
    gi["is_doubleheader"] = (gi["number"] == 2).astype(int)

    def _team_rest(team_col: str, prefix: str) -> pd.DataFrame:
        tg = (
            gi[["gid", team_col, "date"]]
            .rename(columns={team_col: "team"})
            .sort_values(["team", "date", "gid"])
        )
        tg["prev_date"]  = tg.groupby("team")["date"].shift(1)
        tg["days_rest"]  = (tg["date"] - tg["prev_date"]).dt.days.fillna(7).clip(0, 14)
        tg["back_to_back"] = (tg["days_rest"] <= 1).astype(int)
        return tg[["gid", "days_rest", "back_to_back"]].rename(
            columns={"days_rest": f"{prefix}_days_rest",
                     "back_to_back": f"{prefix}_back_to_back"}
        )

    home_rest = _team_rest("hometeam", "home")
    away_rest = _team_rest("visteam",  "away")
    return (
        gi[["gid", "is_doubleheader"]]
        .merge(home_rest, on="gid", how="left")
        .merge(away_rest, on="gid", how="left")
        .drop_duplicates("gid")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4.1 — Team Fielding Quality
# ---------------------------------------------------------------------------

def fielding_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season-level fielding metrics per team.

    Returns: season, team, errors_per_g, dp_rate, def_efficiency
    """
    ts = _load_teamstats(min_year, max_year)
    grp = ts.groupby(["season", "team_full"]).agg(
        games      =("gid",   "count"),
        total_errors=("d_e",  "sum"),
        total_dp    =("d_dp", "sum"),
        total_po    =("d_po", "sum"),
        total_a     =("d_a",  "sum"),
    ).reset_index().rename(columns={"team_full": "team"})
    g = grp["games"].clip(lower=1)
    grp["errors_per_g"] = (grp["total_errors"] / g).round(3)
    grp["dp_rate"]      = (grp["total_dp"]      / g).round(3)
    chances = (grp["total_po"] + grp["total_a"] + grp["total_errors"]).clip(lower=1)
    grp["def_efficiency"] = ((grp["total_po"] + grp["total_a"]) / chances).round(4)
    return grp[["season", "team", "errors_per_g", "dp_rate", "def_efficiency"]]


# ---------------------------------------------------------------------------
# 5.1 — Plate Discipline: K/BB Rates
# ---------------------------------------------------------------------------

def kb_rate_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season K%, BB%, K/BB ratio for each team's batting lineup.

    Returns: season, team, K_rate, BB_rate, K_BB_ratio
    """
    ts = _load_teamstats(min_year, max_year)
    grp = ts.groupby(["season", "team_full"]).agg(
        total_pa =("b_pa", "sum"),
        total_k  =("b_k",  "sum"),
        total_bb =("b_w",  "sum"),
    ).reset_index().rename(columns={"team_full": "team"})
    pa = grp["total_pa"].clip(lower=1)
    grp["K_rate"]     = (grp["total_k"] / pa).round(4)
    grp["BB_rate"]    = (grp["total_bb"] / pa).round(4)
    grp["K_BB_ratio"] = (grp["total_k"] / grp["total_bb"].clip(lower=1)).round(3)
    return grp[["season", "team", "K_rate", "BB_rate", "K_BB_ratio"]]


# ---------------------------------------------------------------------------
# 5.2 — LOB & Scoring Efficiency
# ---------------------------------------------------------------------------

def lob_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season LOB per game (reads teamstats CSV which has lob column).

    Returns: season, team, lob_per_g
    """
    ts = _load_teamstats_csv(min_year, max_year)
    grp = ts.groupby(["season", "team_full"]).agg(
        games     =("gid",  "count"),
        total_lob =("lob",  "sum"),
    ).reset_index().rename(columns={"team_full": "team"})
    grp["lob_per_g"] = (grp["total_lob"] / grp["games"].clip(lower=1)).round(2)
    return grp[["season", "team", "lob_per_g"]]


# ---------------------------------------------------------------------------
# 6.2 — Weather Interaction Features
# ---------------------------------------------------------------------------

def weather_interaction_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Derived weather features from gameinfo CSV.

    Returns: gid, wind_out, wind_in, dome_flag, temp_cold, temp_hot,
             overcast_flag
    """
    _weather_cols = {"gid", "site", "fieldcond", "winddir", "windspeed",
                     "precip", "sky", "temp", "season"}
    path_csv = _RETRO / "gameinfo.csv"
    if path_csv.exists():
        gi = pd.read_csv(
            path_csv,
            usecols=lambda c: c in _weather_cols,
            dtype=str,
            low_memory=False,
        )
    else:
        gi = pd.read_parquet(_RETRO / "gameinfo.parquet")
        gi = gi[[c for c in gi.columns if c in _weather_cols]]
    gi["season"] = pd.to_numeric(gi["season"], errors="coerce")
    gi = gi[(gi["season"] >= min_year) & (gi["season"] <= max_year)].copy()

    winddir = gi["winddir"].fillna("unknown").str.lower().str.strip()
    gi["wind_out"] = winddir.isin(_WIND_OUT_CODES).astype(float)
    gi["wind_in"]  = winddir.isin(_WIND_IN_CODES).astype(float)

    site      = gi["site"].fillna("")
    fieldcond = gi["fieldcond"].fillna("").str.lower()
    gi["dome_flag"] = (site.isin(_DOME_SITES) | (fieldcond == "dome")).astype(float)

    temp = pd.to_numeric(gi["temp"], errors="coerce")
    gi["temp_cold"] = (temp < 50).astype(float)
    gi["temp_hot"]  = (temp > 90).astype(float)

    sky = gi["sky"].fillna("unknown").str.lower()
    gi["overcast_flag"] = sky.isin({"overcast", "cloudy"}).astype(float)

    # Dome overrides environmental conditions
    dome = gi["dome_flag"] == 1
    gi.loc[dome, ["wind_out", "wind_in", "temp_cold", "temp_hot", "overcast_flag"]] = 0.0

    return (
        gi[["gid", "wind_out", "wind_in", "dome_flag",
            "temp_cold", "temp_hot", "overcast_flag"]]
        .drop_duplicates("gid")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6.3 — Umpire Home-Plate Tendency
# ---------------------------------------------------------------------------

def umpire_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Expanding-window historical runs/game for each home-plate umpire.

    Uses data from up to 3 seasons before min_year for warm-up.
    Avoids lookahead: each game uses only prior umpire history.

    Returns: gid, ump_runs_avg, ump_above_avg_flag
    """
    warmup_year = max(min_year - 3, 2015)
    _ump_cols = {"gid", "date", "umphome", "vruns", "hruns", "season"}
    path_csv = _RETRO / "gameinfo.csv"
    if path_csv.exists():
        gi = pd.read_csv(
            path_csv,
            usecols=lambda c: c in _ump_cols,
            dtype=str,
            low_memory=False,
        )
    else:
        gi = pd.read_parquet(_RETRO / "gameinfo.parquet")
        gi = gi[[c for c in gi.columns if c in _ump_cols]]
    if "umphome" not in gi.columns:
        # umphome absent from parquet until build_parquet_data.py is re-run locally
        return pd.DataFrame(columns=[
            "gid", "ump_runs_avg", "ump_above_avg_flag",
            "ump_home_games", "ump_home_over_mean", "ump_home_trend",
        ])
    gi["season"]     = pd.to_numeric(gi["season"], errors="coerce")
    gi               = gi[gi["season"] >= warmup_year].copy()
    gi["date"]       = pd.to_datetime(gi["date"].astype(str), format="%Y%m%d", errors="coerce")
    gi["vruns"]      = pd.to_numeric(gi["vruns"], errors="coerce")
    gi["hruns"]      = pd.to_numeric(gi["hruns"], errors="coerce")
    gi["total_runs"] = gi["vruns"] + gi["hruns"]
    gi               = gi.sort_values("date").reset_index(drop=True)

    league_mean = gi["total_runs"].mean()

    # Expanding lagged mean per umpire (no lookahead)
    gi["ump_runs_avg"] = (
        gi.groupby("umphome")["total_runs"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
        .fillna(league_mean)
        .round(2)
    )
    gi["ump_above_avg_flag"] = (gi["ump_runs_avg"] > league_mean).astype(float)

    # Games umpired before this game (sample-size signal)
    gi["ump_home_games"] = (
        gi.groupby("umphome")["total_runs"]
        .expanding()
        .count()
        .reset_index(level=0, drop=True)
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # Delta vs league average (positive = over-average run environment)
    gi["ump_home_over_mean"] = (gi["ump_runs_avg"] - league_mean).round(2)

    # Trend: rolling-30 lagged mean minus prior rolling-30 (shifted 30 more)
    _roll30 = (
        gi.groupby("umphome")["total_runs"]
        .transform(lambda x: x.rolling(30, min_periods=10).mean().shift(1))
    )
    _roll30_prior = (
        gi.groupby("umphome")["total_runs"]
        .transform(lambda x: x.rolling(30, min_periods=10).mean().shift(31))
    )
    gi["ump_home_trend"] = (_roll30 - _roll30_prior).round(2).fillna(0.0)

    return (
        gi[gi["season"] >= min_year][[
            "gid", "ump_runs_avg", "ump_above_avg_flag",
            "ump_home_games", "ump_home_over_mean", "ump_home_trend",
        ]]
        .drop_duplicates("gid")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6.3b — Umpire Base Position Tendencies
# ---------------------------------------------------------------------------

def umpire_position_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Expanding-window historical runs/game for each base umpire position (1B/2B/3B).

    Requires ump1b/ump2b/ump3b columns in gameinfo (added in build_parquet_data.py).
    Gracefully returns league-mean fill if the columns are absent.
    Avoids lookahead: each game uses only prior history for that umpire.

    Returns: gid, ump1b_runs_avg, ump2b_runs_avg, ump3b_runs_avg
    """
    warmup_year = max(min_year - 3, 2015)
    _cols = {"gid", "date", "ump1b", "ump2b", "ump3b", "vruns", "hruns", "season"}
    path_csv = _RETRO / "gameinfo.csv"
    if path_csv.exists():
        gi = pd.read_csv(
            path_csv,
            usecols=lambda c: c in _cols,
            dtype=str,
            low_memory=False,
        )
    else:
        gi = pd.read_parquet(_RETRO / "gameinfo.parquet")
        gi = gi[[c for c in gi.columns if c in _cols]]

    pos_cols = [c for c in ("ump1b", "ump2b", "ump3b") if c in gi.columns]
    _out_cols = ["gid", "ump1b_runs_avg", "ump2b_runs_avg", "ump3b_runs_avg"]
    if not pos_cols:
        return pd.DataFrame(columns=_out_cols)

    gi["season"]     = pd.to_numeric(gi["season"], errors="coerce")
    gi               = gi[gi["season"] >= warmup_year].copy()
    gi["date"]       = pd.to_datetime(gi["date"].astype(str), format="%Y%m%d", errors="coerce")
    gi["vruns"]      = pd.to_numeric(gi["vruns"], errors="coerce")
    gi["hruns"]      = pd.to_numeric(gi["hruns"], errors="coerce")
    gi["total_runs"] = gi["vruns"] + gi["hruns"]
    gi               = gi.sort_values("date").reset_index(drop=True)

    league_mean = gi["total_runs"].mean()
    result = gi[["gid", "season"]].copy()

    for pos in pos_cols:
        result[f"{pos}_runs_avg"] = (
            gi.groupby(pos)["total_runs"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
            .fillna(league_mean)
            .round(2)
        )

    # Ensure all three output columns exist even if source data is partial
    for pos in ("ump1b", "ump2b", "ump3b"):
        col = f"{pos}_runs_avg"
        if col not in result.columns:
            result[col] = float(league_mean)

    return (
        result[result["season"] >= min_year][_out_cols]
        .drop_duplicates("gid")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 7.1 — Pythagorean Win% Differential
# ---------------------------------------------------------------------------

def pythagorean_diff_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Actual WPct minus Pythagorean WPct per team per season.

    Positive = over-performing (regression expected).
    Negative = under-performing (improvement expected).

    Returns: season, team, pyth_diff
    """
    stnd = season_standings(min_year, max_year)
    stnd["WPct"]      = pd.to_numeric(stnd["WPct"],      errors="coerce")
    stnd["PythWPct"]  = pd.to_numeric(stnd["PythWPct"],  errors="coerce")
    stnd = stnd.dropna(subset=["WPct", "PythWPct"]).copy()
    stnd["pyth_diff"] = (stnd["WPct"] - stnd["PythWPct"]).round(4)
    return stnd[["season", "team", "pyth_diff"]]


# ---------------------------------------------------------------------------
# 7.2 — Base-Running Efficiency
# ---------------------------------------------------------------------------

def baserunning_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Stolen base success rate and per-game rate.

    Returns: season, team, sb_success_rate, sb_rate
    """
    ts = _load_teamstats(min_year, max_year)
    grp = ts.groupby(["season", "team_full"]).agg(
        games    =("gid",  "count"),
        total_sb =("b_sb", "sum"),
        total_cs =("b_cs", "sum"),
    ).reset_index().rename(columns={"team_full": "team"})
    attempts = (grp["total_sb"] + grp["total_cs"]).clip(lower=1)
    grp["sb_success_rate"] = (grp["total_sb"] / attempts).round(3)
    grp["sb_rate"]         = (grp["total_sb"] / grp["games"].clip(lower=1)).round(3)
    return grp[["season", "team", "sb_success_rate", "sb_rate"]]


# ---------------------------------------------------------------------------
# 7.3 — Bullpen Workload & Fatigue
# ---------------------------------------------------------------------------

def bullpen_fatigue_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Total relief IP and unique relievers used per team in the 3 days BEFORE each game.

    Returns: gid, home_bullpen_ip_3d, away_bullpen_ip_3d,
                  home_pen_arms_3d, away_pen_arms_3d
    """
    p = pd.read_parquet(_RETRO / "pitching.parquet")
    p = p[p["p_gs"] != 1.0].copy()  # relief appearances only
    needed = {"gid", "id", "team", "vishome", "p_ipouts", "date"}
    p = p[[c for c in needed if c in p.columns]].copy()
    p["date"]   = pd.to_datetime(p["date"].astype(str), format="%Y%m%d", errors="coerce")
    p["ip"]     = pd.to_numeric(p["p_ipouts"], errors="coerce").fillna(0) / 3
    p["season"] = p["date"].dt.year
    p = p[(p["season"] >= min_year) & (p["season"] <= max_year)]

    # Aggregate to game-day relief totals per team/side
    game_rel = (
        p.groupby(["team", "vishome", "gid", "date"])
        .agg(relief_ip=("ip", "sum"), relief_arms=("id", "nunique"))
        .reset_index()
        .sort_values(["team", "vishome", "date"])
    )

    def _rolling_3d(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        # closed='left' excludes the current game date
        df["bullpen_ip_3d"]  = df["relief_ip"].rolling("3D",   closed="left").sum().fillna(0)
        df["pen_arms_3d"]    = df["relief_arms"].rolling("3D", closed="left").sum().fillna(0)
        return df.reset_index()

    # Avoid groupby.apply() entirely — pandas 3.x drops groupby key columns,
    # and the include_groups workaround still drops both 'team' and 'vishome'.
    # A simple loop + concat is version-safe and equally fast for this table size.
    parts: list[pd.DataFrame] = []
    for (team, vh), grp in game_rel.groupby(["team", "vishome"]):
        result = _rolling_3d(grp.copy())
        result["team"] = team
        result["vishome"] = vh
        parts.append(result)
    game_rel = pd.concat(parts, ignore_index=True)

    home = (
        game_rel[game_rel["vishome"] == "h"][["gid", "bullpen_ip_3d", "pen_arms_3d"]]
        .rename(columns={"bullpen_ip_3d": "home_bullpen_ip_3d",
                         "pen_arms_3d":   "home_pen_arms_3d"})
        .drop_duplicates("gid")
    )
    away = (
        game_rel[game_rel["vishome"] == "v"][["gid", "bullpen_ip_3d", "pen_arms_3d"]]
        .rename(columns={"bullpen_ip_3d": "away_bullpen_ip_3d",
                         "pen_arms_3d":   "away_pen_arms_3d"})
        .drop_duplicates("gid")
    )
    return home.merge(away, on="gid", how="outer").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.3 — SP vs. Opponent History
# ---------------------------------------------------------------------------

def sp_vs_opp_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Cumulative SP ERA and K/9 against the specific opponent BEFORE this game.

    Requires ≥3 prior starts vs. that opponent; otherwise NaN (handled by
    downstream fillna with the SP's season ERA / K9).

    Returns: gid, home_sp_vs_opp_ERA, home_sp_vs_opp_K9,
                  away_sp_vs_opp_ERA, away_sp_vs_opp_K9
    """
    warmup = max(min_year - 5, 2015)
    p = pd.read_parquet(_RETRO / "pitching.parquet")
    needed = {"gid", "id", "team", "vishome", "opp", "p_gs",
              "p_ipouts", "p_er", "p_k", "date"}
    p = p[[c for c in needed if c in p.columns]].copy()
    p = p[p["p_gs"] == 1.0].copy()
    p["date"]   = pd.to_datetime(p["date"].astype(str), format="%Y%m%d", errors="coerce")
    p["season"] = p["date"].dt.year
    p = p[p["season"] >= warmup]
    for col in ("p_ipouts", "p_er", "p_k"):
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
    p["ip"] = p["p_ipouts"] / 3
    p = p.sort_values(["id", "opp", "date"])

    # Cumulative totals before this appearance (shift(1) = excludes current)
    for col, new in [("ip", "cum_ip"), ("p_er", "cum_er"), ("p_k", "cum_k")]:
        p[new] = p.groupby(["id", "opp"])[col].cumsum().shift(1).fillna(0)
    p["cum_starts"] = p.groupby(["id", "opp"]).cumcount()  # prior starts count

    mask   = (p["cum_starts"] >= 3) & (p["cum_ip"] >= 3)
    ip_s   = p["cum_ip"].clip(lower=1)
    p["sp_vs_opp_ERA"] = np.where(mask, (9 * p["cum_er"] / ip_s).round(2), np.nan)
    p["sp_vs_opp_K9"]  = np.where(mask, (9 * p["cum_k"]  / ip_s).round(2), np.nan)

    target = p[p["season"] >= min_year]
    home = (
        target[target["vishome"] == "h"][["gid", "sp_vs_opp_ERA", "sp_vs_opp_K9"]]
        .rename(columns={"sp_vs_opp_ERA": "home_sp_vs_opp_ERA",
                         "sp_vs_opp_K9":  "home_sp_vs_opp_K9"})
        .drop_duplicates("gid")
    )
    away = (
        target[target["vishome"] == "v"][["gid", "sp_vs_opp_ERA", "sp_vs_opp_K9"]]
        .rename(columns={"sp_vs_opp_ERA": "away_sp_vs_opp_ERA",
                         "sp_vs_opp_K9":  "away_sp_vs_opp_K9"})
        .drop_duplicates("gid")
    )
    return home.merge(away, on="gid", how="outer").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.2 — Day/Night Splits
# ---------------------------------------------------------------------------

def daynight_split_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season W% split by day vs. night games.

    Returns: season, team, day_WPct, night_WPct
    """
    gi = _load_gameinfo_csv(min_year, max_year)
    gi["dn"] = gi["daynight"].fillna("n").str.lower().str.strip()

    records = []
    for team_col in ("visteam", "hometeam"):
        tmp = gi[["season", team_col, "dn", "wteam"]].rename(
            columns={team_col: "team"}
        ).copy()
        tmp["won"] = (tmp["wteam"] == tmp["team"]).astype(int)
        records.append(tmp[["season", "team", "dn", "won"]])

    df = pd.concat(records, ignore_index=True)
    grp = df.groupby(["season", "team", "dn"]).agg(
        games=("won", "count"), wins=("won", "sum")
    ).reset_index()
    grp["WPct"] = (grp["wins"] / grp["games"].clip(lower=1)).round(3)
    # Map Retrosheet codes to full names to match gameinfo/features.py convention
    grp["team"] = grp["team"].map(_code_to_name)

    # gameinfo.csv uses 'day' and 'night' (full words)
    day_   = (
        grp[grp["dn"] == "day"][["season", "team", "WPct"]]
        .rename(columns={"WPct": "day_WPct"})
    )
    night_ = (
        grp[grp["dn"] == "night"][["season", "team", "WPct"]]
        .rename(columns={"WPct": "night_WPct"})
    )
    return day_.merge(night_, on=["season", "team"], how="outer")


# ---------------------------------------------------------------------------
# 8.1 — Platoon Advantage (handedness)
# ---------------------------------------------------------------------------

def platoon_features(min_year: int, max_year: int) -> pd.DataFrame:
    """SP throw-arm and team lineup handedness → platoon advantage score.

    platoon_adv = fraction of batters with OPPOSITE hand to opposing SP.
    E.g. away SP is L (LHP) → home right-handed batters have the advantage
         → home_platoon_adv = fraction of home batters who bat R.

    Returns: gid, home_sp_throws_L, away_sp_throws_L,
                  home_pct_left_bat, away_pct_left_bat,
                  home_platoon_adv, away_platoon_adv, platoon_adv_gap
    """
    ap = pd.read_parquet(_RETRO / "allplayers.parquet")

    # Fraction of left-handed batters per team per season (batters only: g_p == 0 ≥ half)
    bat_grp = ap.groupby(["season", "team"]).apply(
        lambda d: pd.Series({
            "pct_left_bat":  (d["bat"] == "L").mean(),
            "pct_right_bat": (d["bat"] == "R").mean(),
        })
    ).reset_index()

    # Starting pitcher throw arm per game
    p = pd.read_parquet(_RETRO / "pitching.parquet")
    p = p[p["p_gs"] == 1.0].copy()
    p["season"] = pd.to_numeric(p["date"].astype(str).str[:4], errors="coerce")
    p = p[(p["season"] >= min_year) & (p["season"] <= max_year)]
    p = p.merge(
        ap[["id", "season", "throw"]].drop_duplicates(subset=["id", "season"]),
        on=["id", "season"], how="left",
    )
    p["sp_throws_L"] = (p["throw"] == "L").astype(float)

    home_t = (
        p[p["vishome"] == "h"][["gid", "season", "team", "sp_throws_L"]]
        .rename(columns={"sp_throws_L": "home_sp_throws_L", "team": "hometeam"})
        .drop_duplicates("gid")
    )
    away_t = (
        p[p["vishome"] == "v"][["gid", "season", "team", "sp_throws_L"]]
        .rename(columns={"sp_throws_L": "away_sp_throws_L", "team": "visteam"})
        .drop_duplicates("gid")
    )

    result = home_t.merge(away_t, on=["gid", "season"], how="outer")
    result = result.merge(
        bat_grp[["season", "team", "pct_left_bat"]].rename(
            columns={"team": "hometeam", "pct_left_bat": "home_pct_left_bat"}),
        on=["season", "hometeam"], how="left",
    ).merge(
        bat_grp[["season", "team", "pct_left_bat"]].rename(
            columns={"team": "visteam", "pct_left_bat": "away_pct_left_bat"}),
        on=["season", "visteam"], how="left",
    )

    # Home batters face AWAY SP; away batters face HOME SP
    hl = result["home_pct_left_bat"].fillna(0.5)
    al = result["away_pct_left_bat"].fillna(0.5)
    result["home_platoon_adv"] = np.where(
        result["away_sp_throws_L"] == 1,
        1 - hl,   # right-bat advantage vs LHP
        hl,       # left-bat advantage vs RHP
    ).round(3)
    result["away_platoon_adv"] = np.where(
        result["home_sp_throws_L"] == 1,
        1 - al,
        al,
    ).round(3)
    result["platoon_adv_gap"] = (
        result["home_platoon_adv"] - result["away_platoon_adv"]
    ).round(3)

    keep = [
        "gid", "home_sp_throws_L", "away_sp_throws_L",
        "home_pct_left_bat", "away_pct_left_bat",
        "home_platoon_adv", "away_platoon_adv", "platoon_adv_gap",
    ]
    return result[[c for c in keep if c in result.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers for Savant-based features (docs/14 priorities 3, 5, 7, 8, 11, 12, 13)
# ---------------------------------------------------------------------------

_RAW_DIR = Path(__file__).resolve().parents[2] / "data_files" / "raw"


def _load_savant_batter_csv(min_year: int, max_year: int) -> pd.DataFrame:
    """Load the raw Savant batter CSV(s) for the requested year range."""
    csv_files = sorted((_RAW_DIR / "batting").glob("savant_batter_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
            df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
            frames.append(df)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_savant_pitcher_csv(min_year: int, max_year: int) -> pd.DataFrame:
    """Load the raw Savant pitcher CSV(s) for the requested year range."""
    csv_files = sorted((_RAW_DIR / "pitching").glob("savant_pitcher_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
            df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
            frames.append(df)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_savant_batter_team_agg(min_year: int, max_year: int) -> pd.DataFrame:
    """Aggregate Savant batter stats to (season, team) via Chadwick cross-reference.

    Uses retrosheet batting rows (which have the Retrosheet player ID and team) to
    anchor which players belong to each team, then joins the Savant player_id through
    the Chadwick Bureau registry.

    Returns DataFrame with columns:
        season, team, team_barrel_pct, team_exit_velo, team_sprint_speed,
        team_oaa, team_xwoba, team_xwoba_diff
    """
    sv = _load_savant_batter_csv(min_year, max_year)
    if sv.empty:
        return pd.DataFrame()

    needed_sv = ["player_id", "year", "barrel_batted_rate", "exit_velocity_avg",
                 "sprint_speed", "n_outs_above_average", "xwoba", "wobadiff"]
    sv = sv[[c for c in needed_sv if c in sv.columns]].copy()
    sv["player_id"] = pd.to_numeric(sv["player_id"], errors="coerce")

    # Retrosheet batting: gid, id (retro ID), team, date
    bat = pd.read_parquet(_RETRO / "batting.parquet",
                          columns=["id", "team", "date"] if True else None)
    bat = bat[["id", "team", "date"]].copy()
    bat["season"] = pd.to_numeric(bat["date"].astype(str).str[:4], errors="coerce")
    bat = bat[(bat["season"] >= min_year) & (bat["season"] <= max_year)]
    bat = bat[["id", "team", "season"]].drop_duplicates()

    # Chadwick: retro_id → mlbam player_id
    try:
        from src.ingestion.chadwick import load_player_registry
        registry = load_player_registry()
        id_map = (
            registry[["key_retro", "key_mlbam"]]
            .dropna(subset=["key_retro", "key_mlbam"])
            .copy()
        )
        id_map["key_mlbam"] = pd.to_numeric(id_map["key_mlbam"], errors="coerce")
        id_map = id_map.dropna(subset=["key_mlbam"])
        id_map["key_mlbam"] = id_map["key_mlbam"].astype(int)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    # Join Retrosheet players → MLBAM IDs
    bat = bat.merge(
        id_map.rename(columns={"key_retro": "id", "key_mlbam": "player_id"}),
        on="id", how="inner",
    )

    # Join Savant stats
    merged = bat.merge(
        sv, left_on=["player_id", "season"], right_on=["player_id", "year"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame()

    # Aggregate to (season, team) — mean per player who qualified that year
    agg_dict: dict[str, tuple] = {}
    for col, out_col in [
        ("barrel_batted_rate", "team_barrel_pct"),
        ("exit_velocity_avg",  "team_exit_velo"),
        ("sprint_speed",       "team_sprint_speed"),
        ("xwoba",              "team_xwoba"),
        ("wobadiff",           "team_xwoba_diff"),
    ]:
        if col in merged.columns:
            agg_dict[out_col] = (col, "mean")
    if "n_outs_above_average" in merged.columns:
        agg_dict["team_oaa"] = ("n_outs_above_average", "sum")

    agg = merged.groupby(["season", "team"]).agg(**agg_dict).reset_index()
    team_map = {k: _code_to_name(k) for k in agg["team"].unique()}
    agg["team"] = agg["team"].map(team_map).fillna(agg["team"])
    return agg.round(3)


def _build_savant_sp_agg(min_year: int, max_year: int) -> pd.DataFrame:
    """Aggregate Savant pitcher stats to (season, id_retro) via Chadwick join.

    Returns DataFrame with columns:
        season, id (retrosheet pitcher ID), sp_xwoba, sp_wobadiff,
        sp_barrel_allowed, sp_whiff_pct, sp_edge_pct
    """
    sp = _load_savant_pitcher_csv(min_year, max_year)
    if sp.empty:
        return pd.DataFrame()

    needed_sp = ["player_id", "year", "xwoba", "wobadiff", "barrel_batted_rate",
                 "whiff_percent", "edge_percent"]
    sp = sp[[c for c in needed_sp if c in sp.columns]].copy()
    sp["player_id"] = pd.to_numeric(sp["player_id"], errors="coerce")

    try:
        from src.ingestion.chadwick import load_player_registry
        registry = load_player_registry()
        id_map = (
            registry[["key_retro", "key_mlbam"]]
            .dropna(subset=["key_retro", "key_mlbam"])
            .copy()
        )
        id_map["key_mlbam"] = pd.to_numeric(id_map["key_mlbam"], errors="coerce").astype(int)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    sp = sp.merge(
        id_map.rename(columns={"key_mlbam": "player_id", "key_retro": "id"}),
        on="player_id", how="inner",
    )
    sp = sp.rename(columns={
        "year":                "season",
        "xwoba":               "sp_xwoba",
        "wobadiff":            "sp_wobadiff",
        "barrel_batted_rate":  "sp_barrel_allowed",
        "whiff_percent":       "sp_whiff_pct",
        "edge_percent":        "sp_edge_pct",
    })
    keep = ["id", "season"] + [c for c in ["sp_xwoba", "sp_wobadiff",
            "sp_barrel_allowed", "sp_whiff_pct", "sp_edge_pct"] if c in sp.columns]
    return sp[keep].drop_duplicates(subset=["id", "season"])


# ---------------------------------------------------------------------------
# Priority 4 — Team Scoring Consistency (baseballr: team_consistency)
# ---------------------------------------------------------------------------

def team_consistency(min_year: int, max_year: int) -> pd.DataFrame:
    """Proportion of games where each team scores above / allows below median.

    ``con_r``  = fraction of games where team scored > season median scored
    ``con_ra`` = fraction of games where team allowed < season median allowed

    High con_r + high con_ra → reliable scoring environment (better for unders).
    Low con_r → boom-or-bust offense (wider run distribution).

    Returns: season, team, con_r, con_ra
    """
    gi = _load_gameinfo_csv(min_year, max_year)
    gi["vruns"] = pd.to_numeric(gi["vruns"], errors="coerce")
    gi["hruns"] = pd.to_numeric(gi["hruns"], errors="coerce")
    gi = gi.dropna(subset=["vruns", "hruns"])

    # Stack home and away into one long table: (season, team, scored, allowed)
    home = gi[["season", "hometeam", "hruns", "vruns"]].rename(
        columns={"hometeam": "team", "hruns": "scored", "vruns": "allowed"}
    )
    away = gi[["season", "visteam", "vruns", "hruns"]].rename(
        columns={"visteam": "team", "vruns": "scored", "hruns": "allowed"}
    )
    df = pd.concat([home, away], ignore_index=True)

    # Season-level medians
    season_med = df.groupby("season").agg(
        med_scored=("scored", "median"),
        med_allowed=("allowed", "median"),
    )
    df = df.merge(season_med, on="season", how="left")

    agg = (
        df.groupby(["season", "team"])
        .apply(lambda g: pd.Series({
            "con_r":  (g["scored"]  > g["med_scored"].iloc[0]).mean(),
            "con_ra": (g["allowed"] < g["med_allowed"].iloc[0]).mean(),
        }))
        .reset_index()
    )
    agg["team"] = agg["team"].map(_code_to_name).fillna(agg["team"])
    return agg[["season", "team", "con_r", "con_ra"]].round(3)


# ---------------------------------------------------------------------------
# Priority 1/13 — wOBA (year-specific FanGraphs weights) per team season
# ---------------------------------------------------------------------------

def woba_team_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season wOBA per team using FanGraphs Guts! linear weights.

    More accurate than OPS as a per-PA offensive value metric; weights for
    singles, doubles, etc. are era-calibrated rather than hardcoded.

    Returns: season, team, team_wOBA
    """
    try:
        from src.ingestion.fg_guts import load_fg_guts
        guts = load_fg_guts()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    bat = pd.read_parquet(_RETRO / "batting.parquet")
    bat["season"] = pd.to_numeric(bat["date"].astype(str).str[:4], errors="coerce")
    bat = bat[(bat["season"] >= min_year) & (bat["season"] <= max_year)].copy()

    for col in ("b_ab", "b_h", "b_d", "b_t", "b_hr", "b_sf",
                "b_hbp", "b_w", "b_k"):
        bat[col] = pd.to_numeric(bat[col], errors="coerce").fillna(0)

    bat["b_1b"] = bat["b_h"] - bat["b_d"] - bat["b_t"] - bat["b_hr"]

    # Aggregate to (season, team)
    grp = bat.groupby(["season", "team"]).agg(
        AB =("b_ab",  "sum"),
        BB =("b_w",   "sum"),
        HBP=("b_hbp", "sum"),
        H1B=("b_1b",  "sum"),
        H2B=("b_d",   "sum"),
        H3B=("b_t",   "sum"),
        HR =("b_hr",  "sum"),
        SF =("b_sf",  "sum"),
    ).reset_index()

    results = []
    for season, season_df in grp.groupby("season"):
        row = guts[guts["season"] == season]
        if row.empty:
            past = guts[guts["season"] <= season]
            row = past.sort_values("season").iloc[[-1]] if not past.empty else guts.sort_values("season").iloc[[0]]
        if row.empty:
            continue
        w = row.iloc[0]
        denom = (season_df["AB"] + season_df["BB"] + season_df["SF"] + season_df["HBP"]).clip(lower=1)
        season_df["team_wOBA"] = (
            w["wBB"]  * season_df["BB"]
            + w["wHBP"] * season_df["HBP"]
            + w["w1B"]  * season_df["H1B"]
            + w["w2B"]  * season_df["H2B"]
            + w["w3B"]  * season_df["H3B"]
            + w["wHR"]  * season_df["HR"]
        ) / denom
        results.append(season_df)

    out = pd.concat(results, ignore_index=True)
    out["team"] = out["team"].map(_code_to_name).fillna(out["team"])
    return out[["season", "team", "team_wOBA"]].round(4)


# ---------------------------------------------------------------------------
# Priority 5 — FIP for Starting Pitchers (year-specific cFIP constant)
# ---------------------------------------------------------------------------

def fip_sp_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Season FIP for each starting pitcher, using the year's cFIP constant.

    FIP outperforms ERA as a predictor of future performance because it strips
    out defensive luck and batted-ball variation.

    Returns: gid, home_sp_FIP, away_sp_FIP
    """
    try:
        from src.ingestion.fg_guts import load_fg_guts
        guts = load_fg_guts()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    p = pd.read_parquet(_RETRO / "pitching.parquet")
    p["season"] = pd.to_numeric(p["date"].astype(str).str[:4], errors="coerce")
    p = p[(p["season"] >= min_year) & (p["season"] <= max_year)
          & (p["p_gs"] == 1.0)].copy()
    for col in ("p_ipouts", "p_hr", "p_w", "p_iw", "p_k", "p_hbp"):
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
    p["ip"] = p["p_ipouts"] / 3
    p["uBB"] = (p["p_w"] - p["p_iw"]).clip(lower=0)

    # Season aggregates per pitcher
    sp_agg = p.groupby(["season", "id"]).agg(
        total_ip=("ip",    "sum"),
        total_hr=("p_hr",  "sum"),
        total_bb=("uBB",   "sum"),
        total_hbp=("p_hbp","sum"),
        total_k=("p_k",    "sum"),
    ).reset_index()

    results = []
    for season, season_df in sp_agg.groupby("season"):
        row = guts[guts["season"] == season]
        if row.empty:
            past = guts[guts["season"] <= season]
            row = past.sort_values("season").iloc[[-1]] if not past.empty else guts.sort_values("season").iloc[[0]]
        if row.empty:
            continue
        c_fip = float(row["cFIP"].iloc[0])
        ip_s = season_df["total_ip"].clip(lower=0.1)
        season_df["sp_FIP"] = (
            (13 * season_df["total_hr"]
             + 3 * (season_df["total_bb"] + season_df["total_hbp"])
             - 2 * season_df["total_k"]) / ip_s + c_fip
        ).round(2)
        results.append(season_df)

    sp_agg = pd.concat(results, ignore_index=True)

    # Merge back to game-starter rows
    p = p.merge(sp_agg[["season", "id", "sp_FIP"]], on=["season", "id"], how="left")

    home = (
        p[p["vishome"] == "h"][["gid", "sp_FIP"]]
        .rename(columns={"sp_FIP": "home_sp_FIP"})
        .drop_duplicates("gid")
    )
    away = (
        p[p["vishome"] == "v"][["gid", "sp_FIP"]]
        .rename(columns={"sp_FIP": "away_sp_FIP"})
        .drop_duplicates("gid")
    )
    return home.merge(away, on="gid", how="outer").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Priority 3, 7, 11, 12 — Savant team-level features via Chadwick
# ---------------------------------------------------------------------------

def savant_team_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Savant-derived team batting quality metrics (season level).

    Requires:
      - data_files/raw/batting/savant_batter_*.csv
      - data_files/processed/player_registry.parquet (Chadwick)
      - data_files/retrosheet/batting.parquet

    Returns DataFrame with columns (all optional — absent if Savant CSV missing):
        season, team,
        team_barrel_pct, team_exit_velo, team_sprint_speed, team_oaa,
        team_xwoba, team_xwoba_diff
    """
    return _build_savant_batter_team_agg(min_year, max_year)


def savant_sp_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Savant-derived SP quality metrics joined to each game's starter row.

    Returns: gid, home_sp_xwoba, home_sp_wobadiff, home_sp_barrel_allowed,
                  home_sp_whiff_pct, home_sp_edge_pct,
                  away_sp_xwoba, away_sp_wobadiff, away_sp_barrel_allowed,
                  away_sp_whiff_pct, away_sp_edge_pct
    """
    sp_stats = _build_savant_sp_agg(min_year, max_year)
    if sp_stats.empty:
        return pd.DataFrame()

    # Starter rows from retrosheet (game-level)
    p = pd.read_parquet(_RETRO / "pitching.parquet")
    p["season"] = pd.to_numeric(p["date"].astype(str).str[:4], errors="coerce")
    p = p[(p["season"] >= min_year) & (p["season"] <= max_year)
          & (p["p_gs"] == 1.0)].copy()
    p = p.merge(sp_stats, on=["id", "season"], how="left")

    sp_cols = [c for c in ["sp_xwoba", "sp_wobadiff", "sp_barrel_allowed",
                            "sp_whiff_pct", "sp_edge_pct"] if c in p.columns]
    if not sp_cols:
        return pd.DataFrame()

    home = (
        p[p["vishome"] == "h"][["gid"] + sp_cols]
        .rename(columns={c: f"home_{c}" for c in sp_cols})
        .drop_duplicates("gid")
    )
    away = (
        p[p["vishome"] == "v"][["gid"] + sp_cols]
        .rename(columns={c: f"away_{c}" for c in sp_cols})
        .drop_duplicates("gid")
    )
    return home.merge(away, on="gid", how="outer").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Priority 6 — Handedness-split park factors (FanGraphs)
# ---------------------------------------------------------------------------

def park_factor_features(min_year: int, max_year: int) -> pd.DataFrame:
    """Handedness-split park factors for each game (home team's stadium).

    Fetches FanGraphs park factors if not cached.  Falls back to pf_basic=100
    (neutral) when the API is unavailable.

    Returns: gid, pf_basic_R, pf_basic_L, pf_hr_R, pf_hr_L
    """
    try:
        from src.ingestion.fg_park import load_fg_park_factors
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    gi = _load_gameinfo_csv(min_year, max_year)

    rows = []
    for year in range(min_year, max_year + 1):
        year_gi = gi[gi["season"] == year][["gid", "hometeam"]].copy()
        if year_gi.empty:
            continue
        try:
            pf = load_fg_park_factors(year)
        except Exception:  # noqa: BLE001
            pf = pd.DataFrame()

        for hand in ("R", "L"):
            hand_pf = pf[pf.get("hand", pd.Series(dtype=str)) == hand] if not pf.empty else pd.DataFrame()
            for col_out, col_src in [
                (f"pf_basic_{hand}", "pf_basic"),
                (f"pf_hr_{hand}",    "pf_hr"),
            ]:
                if not hand_pf.empty and col_src in hand_pf.columns:
                    # Map team abbreviation to retrosheet code — best-effort
                    year_gi[col_out] = 100.0  # default neutral
                    if "team_abbrev" in hand_pf.columns:
                        abbrev_map = dict(zip(hand_pf["team_abbrev"], hand_pf[col_src]))
                        year_gi[col_out] = year_gi["hometeam"].map(abbrev_map).fillna(100.0)
                else:
                    year_gi[col_out] = 100.0

        rows.append(year_gi)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    keep = ["gid"] + [c for c in ["pf_basic_R", "pf_basic_L", "pf_hr_R", "pf_hr_L"]
                      if c in result.columns]
    return result[keep].drop_duplicates("gid").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for name, fn in [
        ("rest_days",        lambda: rest_days_features(2024, 2025)),
        ("fielding",         lambda: fielding_features(2024, 2025)),
        ("kb_rates",         lambda: kb_rate_features(2024, 2025)),
        ("lob",              lambda: lob_features(2024, 2025)),
        ("weather",          lambda: weather_interaction_features(2024, 2025)),
        ("umpire",           lambda: umpire_features(2024, 2025)),
        ("ump_positions",    lambda: umpire_position_features(2024, 2025)),
        ("pyth_diff",        lambda: pythagorean_diff_features(2024, 2025)),
        ("baserunning",      lambda: baserunning_features(2024, 2025)),
        ("bullpen_fatigue",  lambda: bullpen_fatigue_features(2024, 2025)),
        ("sp_vs_opp",        lambda: sp_vs_opp_features(2024, 2025)),
        ("daynight_splits",  lambda: daynight_split_features(2024, 2025)),
        ("platoon",          lambda: platoon_features(2024, 2025)),
        # New functions
        ("team_consistency", lambda: team_consistency(2024, 2025)),
        ("woba_team",        lambda: woba_team_features(2024, 2025)),
        ("fip_sp",           lambda: fip_sp_features(2024, 2025)),
        ("savant_team",      lambda: savant_team_features(2024, 2025)),
        ("savant_sp",        lambda: savant_sp_features(2024, 2025)),
        ("park_factors",     lambda: park_factor_features(2024, 2025)),
    ]:
        df = fn()
        print(f"{name:20s}  shape={df.shape}  cols={list(df.columns)}")
