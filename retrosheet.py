"""
Retrosheet data access layer.
Loads raw CSVs, filters to modern era, computes derived stats,
and exposes clean DataFrames for dashboard consumption.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

RAW_DIR = Path(__file__).parent / "data_files" / "retrosheet"
MODERN_START = 2020  # default cutoff year

# ---------------------------------------------------------------------------
# Team name lookup (Retrosheet 3-letter codes → full franchise names)
# ---------------------------------------------------------------------------
TEAM_NAMES: dict[str, str] = {
    "ANA": "Angels", "ARI": "Diamondbacks", "ATH": "Athletics",
    "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHA": "White Sox", "CHN": "Cubs", "CIN": "Reds",
    "CLE": "Guardians", "COL": "Rockies", "DET": "Tigers",
    "HOU": "Astros", "KCA": "Royals", "LAN": "Dodgers",
    "MIA": "Marlins", "MIL": "Brewers", "MIN": "Twins",
    "MON": "Expos", "NYA": "Yankees", "NYN": "Mets",
    "OAK": "Athletics", "PHI": "Phillies", "PIT": "Pirates",
    "SDN": "Padres", "SEA": "Mariners", "SFN": "Giants",
    "SLN": "Cardinals", "TBA": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WAS": "Nationals", "WSN": "Nationals",
    "FLO": "Marlins", "CAL": "Angels", "TBD": "Rays",
    "MIL": "Brewers",
}


def _team_name(code: str) -> str:
    """Return full team name, falling back to the code if not mapped."""
    return TEAM_NAMES.get(str(code).upper(), code)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_retrodate(series: pd.Series) -> pd.Series:
    """Convert YYYYMMDD integer dates to datetime."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def _extract_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str[:4], errors="coerce")


# ---------------------------------------------------------------------------
# Game Info
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_gameinfo(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """
    Loads gameinfo.csv and returns one row per game with key fields.
    Filters to regular-season games within [min_year, max_year].
    """
    min_year = max(min_year, MODERN_START)
    df = pd.read_parquet(RAW_DIR / "gameinfo.parquet")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]
    df["date"] = _parse_retrodate(df["date"])
    df["vruns"] = pd.to_numeric(df["vruns"], errors="coerce")
    df["hruns"] = pd.to_numeric(df["hruns"], errors="coerce")
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce")
    df["total_runs"] = df["vruns"] + df["hruns"]
    df["run_diff"] = df["hruns"] - df["vruns"]  # positive = home win margin
    for col in ("visteam", "hometeam", "wteam", "lteam"):
        if col in df.columns:
            df[col] = df[col].map(_team_name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team Stats (per game)
# ---------------------------------------------------------------------------

_TEAMSTATS_COLS = [
    "gid", "team", "stattype",
    "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
    "b_rbi", "b_sh", "b_sf", "b_hbp", "b_w", "b_iw", "b_k",
    "b_sb", "b_cs", "b_gdp",
    "p_ipouts", "p_bfp", "p_h", "p_hr", "p_r", "p_er",
    "p_w", "p_iw", "p_k", "p_hbp", "p_wp", "p_bk",
    "d_po", "d_a", "d_e", "d_dp",
    "date", "vishome", "opp", "win", "loss", "tie", "gametype",
]


@st.cache_data(show_spinner=False)
def load_teamstats(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """
    Loads teamstats.csv (value rows only), filters to regular season,
    and computes derived rate stats (BA, OBP, SLG, OPS, ERA, WHIP, etc.).
    Returns one row per team per game.
    """
    min_year = max(min_year, MODERN_START)
    df = pd.read_parquet(RAW_DIR / "teamstats.parquet")
    cols = [c for c in _TEAMSTATS_COLS if c in df.columns]
    df["date"] = _parse_retrodate(df["date"])
    df["season"] = _extract_year(df["date"].dt.strftime("%Y%m%d"))
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]

    # Numeric coerce
    num_cols = [c for c in cols if c not in ("gid", "team", "stattype", "vishome", "opp", "gametype", "date")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----- Derived batting stats -----
    df["ip"] = df["p_ipouts"] / 3
    df["singles"] = df["b_h"] - df["b_d"] - df["b_t"] - df["b_hr"]
    # BA
    df["ba"] = (df["b_h"] / df["b_ab"]).round(3)
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    df["obp"] = (
        (df["b_h"] + df["b_w"] + df["b_hbp"])
        / (df["b_ab"] + df["b_w"] + df["b_hbp"] + df["b_sf"].fillna(0))
    ).round(3)
    # SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
    df["slg"] = (
        (df["singles"] + 2 * df["b_d"] + 3 * df["b_t"] + 4 * df["b_hr"])
        / df["b_ab"]
    ).round(3)
    df["ops"] = (df["obp"] + df["slg"]).round(3)

    # ----- Derived pitching stats -----
    ip_safe = df["ip"].where(df["ip"] > 0)
    bb_safe = pd.to_numeric(df["p_w"], errors="coerce").where(pd.to_numeric(df["p_w"], errors="coerce") > 0)
    df["era"] = (9 * df["p_er"] / ip_safe).round(2)
    df["whip"] = ((df["p_h"] + df["p_w"]) / ip_safe).round(3)
    df["k9"] = (9 * df["p_k"] / ip_safe).round(2)
    df["bb9"] = (9 * df["p_w"] / ip_safe).round(2)
    df["k_bb"] = (pd.to_numeric(df["p_k"], errors="coerce") / bb_safe).round(2)
    df["hr9"] = (9 * df["p_hr"] / ip_safe).round(2)
    df["team"] = df["team"].map(_team_name)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batting (individual)
# ---------------------------------------------------------------------------

_BAT_COLS = [
    "gid", "id", "team", "stattype",
    "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
    "b_rbi", "b_w", "b_k", "b_sb", "b_hbp", "b_sf",
    "date", "vishome", "opp", "win", "loss", "gametype",
]


@st.cache_data(show_spinner=False)
def load_batting(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    min_year = max(min_year, MODERN_START)
    df = pd.read_parquet(RAW_DIR / "batting.parquet")
    cols = [c for c in _BAT_COLS if c in df.columns]
    df["date"] = _parse_retrodate(df["date"])
    df["season"] = _extract_year(df["date"].dt.strftime("%Y%m%d"))
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]
    num_cols = [c for c in cols if c not in ("gid", "id", "team", "stattype", "vishome", "opp", "gametype", "date")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["team"] = df["team"].map(_team_name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pitching (individual)
# ---------------------------------------------------------------------------

_PITCH_COLS = [
    "gid", "id", "team", "stattype",
    "p_ipouts", "p_bfp", "p_h", "p_hr", "p_r", "p_er",
    "p_w", "p_iw", "p_k", "p_hbp", "p_wp", "p_bk",
    "p_gs", "p_gf", "p_cg",
    "wp", "lp", "save",
    "date", "vishome", "opp", "win", "loss", "gametype",
]


@st.cache_data(show_spinner=False)
def load_pitching(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    min_year = max(min_year, MODERN_START)
    df = pd.read_parquet(RAW_DIR / "pitching.parquet")
    cols = [c for c in _PITCH_COLS if c in df.columns]
    df["date"] = _parse_retrodate(df["date"])
    df["season"] = _extract_year(df["date"].dt.strftime("%Y%m%d"))
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]
    num_cols = [c for c in cols if c not in ("gid", "id", "team", "stattype", "vishome", "opp", "gametype", "date", "wp", "lp", "save")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ip"] = df["p_ipouts"] / 3
    ip_safe = df["ip"].where(df["ip"] > 0)
    df["era"] = (9 * df["p_er"] / ip_safe).round(2)
    df["whip"] = ((df["p_h"] + df["p_w"]) / ip_safe).round(3)
    df["k9"] = (9 * df["p_k"] / ip_safe).round(2)
    df["team"] = df["team"].map(_team_name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player registry
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_players(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    min_year = max(min_year, MODERN_START)
    df = pd.read_parquet(RAW_DIR / "allplayers.parquet")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[(df["season"] >= min_year) & (df["season"] <= max_year)]
    df["full_name"] = df["first"] + " " + df["last"]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Season aggregations (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def season_team_batting(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """Aggregate team batting stats by season."""
    df = load_teamstats(min_year, max_year)
    agg = df.groupby(["season", "team"]).agg(
        G=("b_ab", "count"),
        PA=("b_pa", "sum"),
        AB=("b_ab", "sum"),
        R=("b_r", "sum"),
        H=("b_h", "sum"),
        doubles=("b_d", "sum"),
        triples=("b_t", "sum"),
        HR=("b_hr", "sum"),
        RBI=("b_rbi", "sum"),
        BB=("b_w", "sum"),
        K=("b_k", "sum"),
        SB=("b_sb", "sum"),
    ).reset_index()
    agg["singles"] = agg["H"] - agg["doubles"] - agg["triples"] - agg["HR"]
    agg["BA"] = (agg["H"] / agg["AB"]).round(3)
    agg["SLG"] = ((agg["singles"] + 2 * agg["doubles"] + 3 * agg["triples"] + 4 * agg["HR"]) / agg["AB"]).round(3)
    agg["OPS"] = agg["SLG"]  # simplified without OBP denominator at aggregate level
    return agg


@st.cache_data(show_spinner=False)
def season_team_pitching(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """Aggregate team pitching stats by season."""
    df = load_teamstats(min_year, max_year)
    agg = df.groupby(["season", "team"]).agg(
        G=("p_ipouts", "count"),
        IPouts=("p_ipouts", "sum"),
        HA=("p_h", "sum"),
        HRA=("p_hr", "sum"),
        RA=("p_r", "sum"),
        ER=("p_er", "sum"),
        BB=("p_w", "sum"),
        SO=("p_k", "sum"),
        WP=("p_wp", "sum"),
    ).reset_index()
    agg["IP"] = (agg["IPouts"] / 3).round(1)
    ip_s = agg["IP"].where(agg["IP"] > 0)
    agg["ERA"] = (9 * agg["ER"] / ip_s).round(2)
    agg["WHIP"] = ((agg["HA"] + agg["BB"]) / ip_s).round(3)
    agg["K9"] = (9 * agg["SO"] / ip_s).round(2)
    agg["BB9"] = (9 * agg["BB"] / ip_s).round(2)
    agg["HR9"] = (9 * agg["HRA"] / ip_s).round(2)
    return agg


@st.cache_data(show_spinner=False)
def season_standings(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """W/L standings per team per season with run differential (vectorized)."""
    gi = load_gameinfo(min_year, max_year)

    # Visitor perspective
    vis = gi[["season", "visteam", "vruns", "hruns", "wteam", "lteam"]].copy()
    vis.columns = ["season", "team", "RS", "RA", "wteam", "lteam"]
    vis["W"] = (vis["team"] == vis["wteam"]).astype(int)
    vis["L"] = (vis["team"] == vis["lteam"]).astype(int)
    vis["home"] = 0

    # Home perspective
    hom = gi[["season", "hometeam", "hruns", "vruns", "wteam", "lteam"]].copy()
    hom.columns = ["season", "team", "RS", "RA", "wteam", "lteam"]
    hom["W"] = (hom["team"] == hom["wteam"]).astype(int)
    hom["L"] = (hom["team"] == hom["lteam"]).astype(int)
    hom["home"] = 1

    df = pd.concat([vis, hom], ignore_index=True)

    overall = df.groupby(["season", "team"]).agg(
        W=("W", "sum"), L=("L", "sum"),
        RS=("RS", "sum"), RA=("RA", "sum"),
    ).reset_index()

    home_df = df[df["home"] == 1].groupby(["season", "team"]).agg(
        Home_W=("W", "sum"), Home_L=("L", "sum"),
    ).reset_index()
    away_df = df[df["home"] == 0].groupby(["season", "team"]).agg(
        Away_W=("W", "sum"), Away_L=("L", "sum"),
    ).reset_index()

    agg = overall.merge(home_df, on=["season", "team"]).merge(away_df, on=["season", "team"])
    agg["G"] = agg["W"] + agg["L"]
    agg["WPct"] = (agg["W"] / agg["G"]).round(3)
    agg["RD"] = agg["RS"] - agg["RA"]
    agg["RD_per_G"] = (agg["RD"] / agg["G"]).round(2)
    agg["RS_per_G"] = (agg["RS"] / agg["G"]).round(2)
    agg["RA_per_G"] = (agg["RA"] / agg["G"]).round(2)
    agg["PythWPct"] = (agg["RS"] ** 2 / (agg["RS"] ** 2 + agg["RA"] ** 2)).round(3)
    return agg.sort_values(["season", "WPct"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def head_to_head(team_a: str, team_b: str, min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """All games between two franchises in the date window."""
    gi = load_gameinfo(min_year, max_year)
    mask = (
        ((gi["visteam"] == team_a) & (gi["hometeam"] == team_b))
        | ((gi["visteam"] == team_b) & (gi["hometeam"] == team_a))
    )
    df = gi[mask].copy()
    df["a_runs"] = df.apply(lambda r: r["vruns"] if r["visteam"] == team_a else r["hruns"], axis=1)
    df["b_runs"] = df.apply(lambda r: r["vruns"] if r["visteam"] == team_b else r["hruns"], axis=1)
    df["a_win"] = (df["wteam"] == team_a).astype(int)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def rolling_team_form(team: str, window: int = 10, min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> pd.DataFrame:
    """Rolling N-game averages for a single team (runs scored/allowed, win rate)."""
    gi = load_gameinfo(min_year, max_year)
    vis = gi[gi["visteam"] == team][["date", "season", "vruns", "hruns", "wteam"]].copy()
    vis.rename(columns={"vruns": "RS", "hruns": "RA"}, inplace=True)
    hom = gi[gi["hometeam"] == team][["date", "season", "hruns", "vruns", "wteam"]].copy()
    hom.rename(columns={"hruns": "RS", "vruns": "RA"}, inplace=True)
    combined = pd.concat([vis, hom]).sort_values("date").reset_index(drop=True)
    combined["W"] = (combined["wteam"] == team).astype(int)
    for col in ("RS", "RA", "W"):
        combined[f"roll_{col}_{window}"] = combined[col].rolling(window, min_periods=1).mean().round(3)
    combined["roll_RD"] = combined[f"roll_RS_{window}"] - combined[f"roll_RA_{window}"]
    return combined


@st.cache_data(show_spinner=False)
def season_batting_leaders(min_year: int = MODERN_START, max_year: int = datetime.date.today().year, min_pa: int = 300) -> pd.DataFrame:
    """Individual batter season totals, filtered to qualified hitters."""
    bat = load_batting(min_year, max_year)
    agg = bat.groupby(["season", "id", "team"]).agg(
        PA=("b_pa", "sum"), AB=("b_ab", "sum"),
        R=("b_r", "sum"), H=("b_h", "sum"),
        doubles=("b_d", "sum"), triples=("b_t", "sum"),
        HR=("b_hr", "sum"), RBI=("b_rbi", "sum"),
        BB=("b_w", "sum"), K=("b_k", "sum"), SB=("b_sb", "sum"),
    ).reset_index()
    agg = agg[agg["PA"] >= min_pa]
    agg["singles"] = agg["H"] - agg["doubles"] - agg["triples"] - agg["HR"]
    agg["BA"] = (agg["H"] / agg["AB"]).round(3)
    agg["SLG"] = ((agg["singles"] + 2*agg["doubles"] + 3*agg["triples"] + 4*agg["HR"]) / agg["AB"]).round(3)
    players = load_players(min_year, max_year)[["id", "full_name", "season"]].drop_duplicates()
    agg = agg.merge(players, on=["id", "season"], how="left")
    agg["full_name"] = agg["full_name"].fillna(agg["id"])
    return agg.sort_values(["season", "BA"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def season_pitching_leaders(min_year: int = MODERN_START, max_year: int = datetime.date.today().year, min_ip: float = 100.0) -> pd.DataFrame:
    """Individual pitcher season totals, filtered to qualified starters."""
    pit = load_pitching(min_year, max_year)
    agg = pit.groupby(["season", "id", "team"]).agg(
        IPouts=("p_ipouts", "sum"),
        H=("p_h", "sum"), HR=("p_hr", "sum"),
        R=("p_r", "sum"), ER=("p_er", "sum"),
        BB=("p_w", "sum"), SO=("p_k", "sum"),
        HBP=("p_hbp", "sum"), WP=("p_wp", "sum"),
        GS=("p_gs", "sum"),
    ).reset_index()
    agg["IP"] = (agg["IPouts"] / 3).round(1)
    agg = agg[agg["IP"] >= min_ip]
    ip_s = agg["IP"].where(agg["IP"] > 0)
    bb_s = agg["BB"].where(agg["BB"] > 0)
    agg["ERA"] = (9 * agg["ER"] / ip_s).round(2)
    agg["WHIP"] = ((agg["H"] + agg["BB"]) / ip_s).round(3)
    agg["K9"] = (9 * agg["SO"] / ip_s).round(2)
    agg["BB9"] = (9 * agg["BB"] / ip_s).round(2)
    agg["K_BB"] = (agg["SO"] / bb_s).round(2)
    players = load_players(min_year, max_year)[["id", "full_name", "season"]].drop_duplicates()
    agg = agg.merge(players, on=["id", "season"], how="left")
    agg["full_name"] = agg["full_name"].fillna(agg["id"])
    return agg.sort_values(["season", "ERA"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def team_list(min_year: int = MODERN_START, max_year: int = datetime.date.today().year) -> list[str]:
    gi = load_gameinfo(min_year, max_year)
    teams = sorted(set(gi["visteam"].dropna()) | set(gi["hometeam"].dropna()))
    return teams
