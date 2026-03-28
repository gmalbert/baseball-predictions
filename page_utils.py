"""
Shared utilities, constants, and cached data loaders for all Streamlit pages.
Import from this module instead of duplicating code across pages.
"""

import sys
import math
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.resolve()
# ROOT first on sys.path so local src/ package is found before any PyPI 'src'
sys.path.insert(0, str(ROOT))

from retrosheet import head_to_head, load_gameinfo, rolling_team_form, TEAM_NAMES
from footer import add_betting_oracle_footer
import statsapi

PROCESSED = ROOT / "data_files" / "processed"

# ─── Brand Colors ─────────────────────────────────────────────────────────────
MLB_BLUE   = "#002D72"
MLB_RED    = "#D50032"
DARK_GREEN = "#1a4731"

CONF_COLORS: dict[str, str] = {
    "HIGH":   "#16a34a",
    "MEDIUM": "#d97706",
    "LOW":    "#6b7280",
    "High":   "#16a34a",
    "Medium": "#d97706",
    "Low":    "#6b7280",
}

# ─── MLB full-name → Retrosheet short-name ────────────────────────────────────
_MLB_TO_RETRO: dict[str, str] = {
    "Arizona Diamondbacks": "Diamondbacks",
    "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros",
    "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins",
    "New York Mets": "Mets",
    "New York Yankees": "Yankees",
    "Oakland Athletics": "Athletics",
    "Sacramento Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres",
    "Seattle Mariners": "Mariners",
    "San Francisco Giants": "Giants",
    "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Washington Nationals": "Nationals",
}

# ─── Human-readable column headers ────────────────────────────────────────────
READABLE_COLS: dict[str, str] = {
    "team": "Team",
    "G": "Games",
    "W": "W",
    "L": "L",
    "WPct": "Win %",
    "PythWPct": "Pythagorean W%",
    "Home_W": "Home W",
    "Home_L": "Home L",
    "Away_W": "Away W",
    "Away_L": "Away L",
    "RS": "Runs Scored",
    "RA": "Runs Allowed",
    "RD": "Run Diff",
    "RD_per_G": "Run Diff / G",
    "RS_per_G": "RS / G",
    "RA_per_G": "RA / G",
    "PA": "Plate App",
    "AB": "At Bats",
    "R": "Runs",
    "H": "Hits",
    "HR": "HR",
    "RBI": "RBI",
    "BB": "Walks",
    "K": "K",
    "SB": "Stolen Bases",
    "BA": "AVG",
    "SLG": "SLG",
    "OPS": "OPS",
    "IP": "Inn. Pitched",
    "HA": "Hits Allowed",
    "HRA": "HR Allowed",
    "ER": "Earned Runs",
    "SO": "Strikeouts",
    "ERA": "ERA",
    "WHIP": "WHIP",
    "K9": "K / 9",
    "BB9": "BB / 9",
    "HR9": "HR / 9",
    "K_BB": "K / BB",
    "GS": "Starts",
    "full_name": "Player",
    "date": "Date",
    "visteam": "Visitor",
    "hometeam": "Home",
    "vruns": "Visitor Runs",
    "hruns": "Home Runs",
}


# ─── Cached API / Data Loaders ────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_todays_schedule() -> list[dict]:
    """Fetch today's MLB schedule via the MLB Stats API. Cached 1 hour."""
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        games = statsapi.schedule(date=today)
        return [g for g in games if g.get("game_type", "R") in ("R", "F", "D", "L", "W", "S")]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def _load_latest_odds() -> pd.DataFrame:
    """Load the most-recent Odds API CSV, or empty DataFrame if none exist."""
    odds_dir = ROOT / "data_files" / "raw" / "odds"
    if not odds_dir.exists():
        return pd.DataFrame()
    files = sorted(odds_dir.glob("odds_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_team_standings() -> dict[str, dict]:
    """Current-season W/L records via the free MLB Stats API."""
    try:
        standings = statsapi.standings_data()
        result: dict[str, dict] = {}
        for _div_id, div_data in standings.items():
            for team in div_data.get("teams", []):
                result[team["name"]] = {
                    "W":      team.get("w", "—"),
                    "L":      team.get("l", "—"),
                    "pct":    team.get("pct", "—"),
                    "streak": team.get("streak", "—"),
                    "L10":    team.get("lastTen", "—"),
                }
        return result
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_pitcher_stats(pitcher_name: str) -> dict:
    """Season pitching stats for a named pitcher via the free MLB Stats API."""
    if not pitcher_name or pitcher_name.strip().upper() == "TBD":
        return {}
    try:
        results = statsapi.lookup_player(pitcher_name)
        if not results:
            return {}
        player_id = results[0]["id"]
        data = statsapi.player_stat_data(player_id, group="pitching", type="season", sportId=1)
        if not data or not data.get("stats"):
            return {}
        s = data["stats"][0]["stats"]
        return {
            "W-L":  f"{s.get('wins', '?')}-{s.get('losses', '?')}",
            "ERA":  s.get("era", "—"),
            "IP":   s.get("inningsPitched", "—"),
            "GS":   s.get("gamesStarted", "—"),
            "K":    s.get("strikeOuts", "—"),
            "BB":   s.get("baseOnBalls", "—"),
            "HR":   s.get("homeRuns", "—"),
            "WHIP": s.get("whip", "—"),
            "K/9":  s.get("strikeoutsPer9Inn", "—"),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_espn_odds() -> list[dict]:
    """
    Fetch today's MLB odds from ESPN's public APIs.
    1. Scoreboard (with date + limit=30) -> all events
    2. Per-event odds from sports.core.api.espn.com (parallel)
    Free, no key needed. Cached 30 min.
    """
    import requests as _requests
    import datetime as _dt
    from concurrent.futures import ThreadPoolExecutor, as_completed

    today = _dt.date.today().strftime("%Y%m%d")
    try:
        resp = _requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
            f"?dates={today}&limit=30",
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception:
        return []

    def _fetch_one(event: dict) -> dict | None:
        try:
            comp = event["competitions"][0]
            eid  = event["id"]
            cid  = comp["id"]
            home_name = next(
                (c["team"]["displayName"] for c in comp["competitors"] if c["homeAway"] == "home"), ""
            )
            away_name = next(
                (c["team"]["displayName"] for c in comp["competitors"] if c["homeAway"] == "away"), ""
            )
            odds_resp = _requests.get(
                f"https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
                f"/events/{eid}/competitions/{cid}/odds",
                timeout=10,
            )
            items = odds_resp.json().get("items", [])
            if not items:
                return None
            o = items[0]
            home_odds = o.get("homeTeamOdds", {})
            away_odds = o.get("awayTeamOdds", {})
            spread_h = home_odds.get("current", {}).get("spread", {}).get("american", "—")
            spread_a = away_odds.get("current", {}).get("spread", {}).get("american", "—")
            return {
                "home_team":   home_name,
                "away_team":   away_name,
                "provider":    o.get("provider", {}).get("name", "ESPN"),
                "ml_home":     home_odds.get("moneyLine"),
                "ml_away":     away_odds.get("moneyLine"),
                "spread_home": spread_h,
                "spread_away": spread_a,
                "details":     o.get("details", "—"),
                "over_under":  o.get("overUnder"),
                "over_odds":   o.get("overOdds", "—"),
                "under_odds":  o.get("underOdds", "—"),
            }
        except Exception:
            return None

    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, e): e for e in events}
        for fut in as_completed(futures):
            item = fut.result()
            if item:
                result.append(item)
    return result


@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict:
    """Load all pre-computed aggregated datasets once at startup."""
    gi = pd.read_parquet(ROOT / "data_files" / "retrosheet" / "gameinfo.parquet")

    mc_ranking: pd.DataFrame | None = None
    mc_trials: pd.DataFrame | None = None
    if (PROCESSED / "mc_feature_ranking.csv").exists():
        mc_ranking = pd.read_csv(PROCESSED / "mc_feature_ranking.csv")
    if (PROCESSED / "mc_feature_trials.parquet").exists():
        mc_trials = pd.read_parquet(PROCESSED / "mc_feature_trials.parquet")

    savant_metrics: pd.DataFrame | None = None
    savant_imps: pd.DataFrame | None = None
    if (PROCESSED / "savant_model_metrics.parquet").exists():
        savant_metrics = pd.read_parquet(PROCESSED / "savant_model_metrics.parquet")
    if (PROCESSED / "savant_model_importances.parquet").exists():
        savant_imps = pd.read_parquet(PROCESSED / "savant_model_importances.parquet")

    return {
        "gameinfo":         gi,
        "standings":        pd.read_parquet(PROCESSED / "standings.parquet"),
        "team_batting":     pd.read_parquet(PROCESSED / "team_batting.parquet"),
        "team_pitching":    pd.read_parquet(PROCESSED / "team_pitching.parquet"),
        "batting_leaders":  pd.read_parquet(PROCESSED / "batting_leaders.parquet"),
        "pitching_leaders": pd.read_parquet(PROCESSED / "pitching_leaders.parquet"),
        "model_features":   pd.read_parquet(PROCESSED / "model_features.parquet"),
        "mc_ranking":       mc_ranking,
        "mc_trials":        mc_trials,
        "savant_metrics":   savant_metrics,
        "savant_imps":      savant_imps,
    }


@st.cache_data(show_spinner=False)
def _load_model_results() -> dict | None:
    """Load pre-computed model training results from parquet files."""
    try:
        metrics_df = pd.read_parquet(PROCESSED / "model_metrics.parquet")
        imps_df    = pd.read_parquet(PROCESSED / "model_importances.parquet")
        results = {}
        for model_name in ["moneyline", "spread", "totals"]:
            row     = metrics_df[metrics_df["model"] == model_name].iloc[0]
            test_df = pd.read_parquet(PROCESSED / f"{model_name}_test_df.parquet")
            results[model_name] = {
                "model":       None,
                "metrics": {
                    "roc_auc":     float(row["roc_auc"]),
                    "accuracy":    float(row["accuracy"]),
                    "brier_score": float(row["brier_score"]),
                    "log_loss":    float(row["log_loss"]),
                },
                "importances": (
                    imps_df[imps_df["model"] == model_name][["feature", "importance"]]
                    .reset_index(drop=True)
                ),
                "feature_cols": [],
                "test_df":     test_df,
                "train_size":  int(row["train_size"]),
                "test_size":   int(row["test_size"]),
            }
        return results
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _load_eval_backtests() -> dict | None:
    """Load BacktestResult objects from pre-computed parquet files."""
    try:
        from src.evaluation.backtester import BacktestResult, BetResult
        bets_df    = pd.read_parquet(PROCESSED / "backtest_bets.parquet")
        summary_df = pd.read_parquet(PROCESSED / "backtest_summary.parquet")
        backtests  = {}
        for model_name in ["moneyline", "totals"]:
            subset = bets_df[bets_df["model_name"] == model_name]
            s_row  = summary_df[summary_df["model"] == model_name].iloc[0]
            bets   = [
                BetResult(
                    game_id=row.game_id,
                    date=row.date,
                    pick_type=row.pick_type,
                    pick_value="",
                    predicted_prob=float(row.predicted_prob),
                    confidence_score=float(row.confidence_score),
                    confidence=row.confidence,
                    edge=float(row.edge),
                    american_odds=int(row.american_odds),
                    result=row.result,
                    profit_units=float(row.profit_units),
                )
                for row in subset.itertuples(index=False)
            ]
            backtests[model_name] = BacktestResult(
                model_name=model_name,
                pick_type=str(s_row["pick_type"]),
                period=str(s_row["period"]),
                bets=bets,
            )
        return backtests
    except FileNotFoundError:
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _kelly_fraction(prob: float, american_odds: int) -> float:
    """Return the full-Kelly bet fraction."""
    if american_odds >= 0:
        decimal = american_odds / 100.0 + 1.0
    else:
        decimal = 100.0 / abs(american_odds) + 1.0
    b = decimal - 1.0
    q = 1.0 - prob
    return max((b * prob - q) / b, 0.0)


def get_dataframe_height(df: pd.DataFrame, row_height: int = 35,
                         header_height: int = 38, padding: int = 2,
                         max_height: int = 600) -> int:
    """Calculate optimal Streamlit dataframe height in pixels."""
    calculated = len(df) * row_height + header_height + padding
    return min(calculated, max_height) if max_height else calculated


def _american_to_implied_prob(american_odds: int) -> float:
    """Convert American moneyline to implied probability (no vig removed)."""
    if american_odds >= 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def _estimate_win_prob(home_full: str, away_full: str,
                       live_standings: dict[str, dict]) -> float:
    """
    Quick logistic win-probability estimate from current-season W%.
    Includes ~4% home-field advantage.
    """
    def _pct(name: str) -> float:
        data = live_standings.get(name, {})
        try:
            return float(str(data.get("pct", "0.500")).replace("—", "0.500") or "0.500")
        except (ValueError, TypeError):
            return 0.500

    diff = _pct(home_full) - _pct(away_full) + 0.04
    return max(0.10, min(0.90, 1.0 / (1.0 + math.exp(-diff * 8))))


# ─── HTML Components ──────────────────────────────────────────────────────────

def _prob_bar_html(home_prob: float, home: str, away: str) -> str:
    """Inline HTML win-probability bar (no Plotly overhead)."""
    hp = round(home_prob * 100)
    ap = 100 - hp
    return (
        f'<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;'
        f'font-size:0.75rem;font-weight:600">'
        f'<div style="width:{hp}%;background:{MLB_BLUE};color:white;display:flex;'
        f'align-items:center;justify-content:center">{hp}%</div>'
        f'<div style="width:{ap}%;background:{MLB_RED};color:white;display:flex;'
        f'align-items:center;justify-content:center">{ap}%</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;'
        f'color:#888;margin-top:2px"><span>{home} (home)</span><span>{away} (away)</span></div>'
    )


def _conf_badge(tier: str) -> str:
    c = CONF_COLORS.get(tier, "#6b7280")
    return (
        f'<span style="background:{c};color:white;padding:2px 10px;border-radius:10px;'
        f'font-size:0.72rem;font-weight:700">{tier.upper()}</span>'
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(show_year_filter: bool = True) -> tuple[int, int]:
    """
    Render sidebar branding + optional year-range filter.
    Returns (min_year, max_year).
    """
    with st.sidebar:
        _logo = ROOT / "data_files" / "logo.png"
        if _logo.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(_logo), width=150)

        if show_year_filter:
            st.markdown("---")
            st.header("Season Filters")
            min_year, max_year = st.slider(
                "Season range",
                min_value=2020,
                max_value=2025,
                value=(2020, 2025),
                step=1,
            )
            st.caption(f"Using {min_year}–{max_year} regular-season games.")
            return min_year, max_year

    return 2020, 2025


# ─── Session State ────────────────────────────────────────────────────────────

def init_session_state(features_df: pd.DataFrame | None = None) -> None:
    """Pre-populate session state from pre-computed results on first load."""
    if "ml_results" not in st.session_state:
        st.session_state["ml_results"] = _load_model_results()
    if features_df is not None and "ml_feat_df" not in st.session_state:
        st.session_state["ml_feat_df"] = features_df
    if "eval_backtests" not in st.session_state:
        st.session_state["eval_backtests"] = _load_eval_backtests()
    if "schedule_selected_game" not in st.session_state:
        st.session_state["schedule_selected_game"] = None
