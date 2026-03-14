import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# NOTE: `st.plotly_chart` and `st.dataframe` now accept a `width` argument
#       (use `'stretch'` in place of the old `width="stretch"`, or
#       `'content'` for the opposite).  This prevents legend overlap and
#       is the preferred API going forward.

ROOT = Path(__file__).parent.resolve()
# ROOT first on sys.path so ROOT/src/ (local package) shadows the PyPI 'src' package
sys.path.insert(0, str(ROOT))

from retrosheet import (
    head_to_head,
    load_gameinfo,
    rolling_team_form,
    TEAM_NAMES,
)
from src.models.features import build_model_features
from src.models.underdog_model import train_moneyline_model
from src.models.spread_model import train_spread_model
from src.models.totals_model import train_totals_model

# evaluation utilities
from src.evaluation.backtester import walk_forward_backtest, calculate_profit, BacktestResult
from src.evaluation.calibration import calibration_plot_data
from src.evaluation.profitability import profitability_report, edge_filter_analysis
from src.evaluation.dashboard import generate_dashboard_data

from footer import add_betting_oracle_footer

import statsapi  # MLB Stats API — used by Daily Schedule tab

PROCESSED = ROOT / "data_files" / "processed"

# ---------------------------------------------------------------------------
# MLB Stats API full-name → Retrosheet short-name (used by head_to_head)
# ---------------------------------------------------------------------------
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


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_todays_schedule() -> list[dict]:
    """Fetch today's MLB schedule via the MLB Stats API. Cached for 1 hour."""
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        games = statsapi.schedule(date=today)
        return [g for g in games if g.get("game_type", "R") in ("R", "F", "D", "L", "W", "S")]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def _load_latest_odds() -> pd.DataFrame:
    """Load the most-recently fetched odds CSV, or empty DataFrame if none exist."""
    odds_dir = ROOT / "data_files" / "raw" / "odds"
    if not odds_dir.exists():
        return pd.DataFrame()
    files = sorted(odds_dir.glob("odds_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_team_standings() -> dict[str, dict]:
    """Current-season W/L records via the free MLB Stats API (no key required)."""
    try:
        standings = statsapi.standings_data()
        result: dict[str, dict] = {}
        for _div_id, div_data in standings.items():
            for team in div_data.get("teams", []):
                result[team["name"]] = {
                    "W": team.get("w", "—"),
                    "L": team.get("l", "—"),
                    "pct": team.get("pct", "—"),
                    "streak": team.get("streak", "—"),
                    "L10": team.get("lastTen", "—"),
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
    Fetch today's MLB scoreboard from ESPN's public API.
    Completely free — no key, no monthly quota.
    Cached 30 min so refreshes don't hammer ESPN.
    """
    import requests as _requests
    try:
        resp = _requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            timeout=10,
        )
        resp.raise_for_status()
        result = []
        for event in resp.json().get("events", []):
            for comp in event.get("competitions", []):
                odds_list = comp.get("odds", [])
                if not odds_list:
                    continue
                o = odds_list[0]
                competitors = comp.get("competitors", [])
                home_name = next(
                    (c["team"]["displayName"] for c in competitors if c["homeAway"] == "home"), ""
                )
                away_name = next(
                    (c["team"]["displayName"] for c in competitors if c["homeAway"] == "away"), ""
                )
                result.append({
                    "home_team":   home_name,
                    "away_team":   away_name,
                    "provider":    o.get("provider", {}).get("name", "ESPN"),
                    "ml_home":     o.get("homeTeamOdds", {}).get("moneyLine", "—"),
                    "ml_away":     o.get("awayTeamOdds", {}).get("moneyLine", "—"),
                    "spread_home": o.get("homeTeamOdds", {}).get("spreadOdds", "—"),
                    "details":     o.get("details", "—"),
                    "over_under":  o.get("overUnder", "—"),
                    "over_odds":   o.get("overOdds", "—"),
                    "under_odds":  o.get("underOdds", "—"),
                })
        return result
    except Exception:
        return []

def _kelly_fraction(prob: float, american_odds: int) -> float:
    """Return the full-Kelly bet fraction for a given win probability + American odds pair."""
    if american_odds >= 0:
        decimal = american_odds / 100.0 + 1.0
    else:
        decimal = 100.0 / abs(american_odds) + 1.0
    b = decimal - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(f, 0.0)


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height


@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict:
    """Load all pre-computed aggregated datasets once at startup.

    Returns a dict of DataFrames covering the full historical range.
    Year filtering happens in-memory after the sidebar slider fires — instant.
    """
    gi = pd.read_parquet(ROOT / "data_files" / "retrosheet" / "gameinfo.parquet")

    mc_ranking: pd.DataFrame | None = None
    mc_trials: pd.DataFrame | None = None
    ranking_path = PROCESSED / "mc_feature_ranking.csv"
    trials_path  = PROCESSED / "mc_feature_trials.parquet"
    if ranking_path.exists():
        mc_ranking = pd.read_csv(ranking_path)
    if trials_path.exists():
        mc_trials = pd.read_parquet(trials_path)

    savant_metrics: pd.DataFrame | None = None
    savant_imps: pd.DataFrame | None = None
    if (PROCESSED / "savant_model_metrics.parquet").exists():
        savant_metrics = pd.read_parquet(PROCESSED / "savant_model_metrics.parquet")
    if (PROCESSED / "savant_model_importances.parquet").exists():
        savant_imps = pd.read_parquet(PROCESSED / "savant_model_importances.parquet")

    return {
        "gameinfo":        gi,
        "standings":       pd.read_parquet(PROCESSED / "standings.parquet"),
        "team_batting":    pd.read_parquet(PROCESSED / "team_batting.parquet"),
        "team_pitching":   pd.read_parquet(PROCESSED / "team_pitching.parquet"),
        "batting_leaders": pd.read_parquet(PROCESSED / "batting_leaders.parquet"),
        "pitching_leaders":pd.read_parquet(PROCESSED / "pitching_leaders.parquet"),
        "model_features":  pd.read_parquet(PROCESSED / "model_features.parquet"),
        "mc_ranking":      mc_ranking,
        "mc_trials":       mc_trials,
        "savant_metrics":  savant_metrics,
        "savant_imps":     savant_imps,
    }


@st.cache_data(show_spinner=False)
def _load_model_results() -> dict | None:
    """Reconstruct model training results from pre-computed parquet files.

    Returns None if the files haven't been generated yet (pre-first run).
    """
    try:
        metrics_df = pd.read_parquet(PROCESSED / "model_metrics.parquet")
        imps_df    = pd.read_parquet(PROCESSED / "model_importances.parquet")
        results = {}
        for model_name in ["moneyline", "spread", "totals"]:
            row     = metrics_df[metrics_df["model"] == model_name].iloc[0]
            test_df = pd.read_parquet(PROCESSED / f"{model_name}_test_df.parquet")
            results[model_name] = {
                "model":       None,   # .joblib not needed for display
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
    """Reconstruct BacktestResult objects from pre-computed parquet files.

    Returns None if the files haven't been generated yet (pre-first run).
    """
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

# Human-readable column header mapping for all st.dataframe() displays
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

# use light theme unconditionally; dark mode made text hard to read
mode = "light"

st.set_page_config(
    page_title="Betting Cleanup - Baseball Predictions",
    page_icon="⚾",
    layout="wide",
)

# light theme only
css = """
<style>
.stApp { background-color: white; color: black; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
st.title("Betting Cleanup")

# ─────────────────────────────────────────────
# Pre-load all data once (runs at startup, cached permanently)
# ─────────────────────────────────────────────
_pre = _load_precomputed()

# ─────────────────────────────────────────────
# Sidebar – global filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Retrosheet Filters")
    min_year, max_year = st.slider(
        "Season range",
        min_value=2020,
        max_value=2025,
        value=(2020, 2025),
        step=1,
    )
    st.caption(f"Using {min_year}–{max_year} regular-season games.")

    # Branding (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(ROOT / "data_files" / "logo.png"), width=150)

# Filter in-memory — instant regardless of year selection
def _yr(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["season"].between(min_year, max_year)].copy()

_gi      = _yr(_pre["gameinfo"])
standings = _yr(_pre["standings"])
tbat      = _yr(_pre["team_batting"])
tpitch    = _yr(_pre["team_pitching"])
bleaders  = _yr(_pre["batting_leaders"])
pleaders  = _yr(_pre["pitching_leaders"])
features_df = _yr(_pre["model_features"])
def _code_to_name(c: str) -> str:
    return TEAM_NAMES.get(str(c).upper(), c)

teams     = sorted(set(
    _gi["visteam"].dropna().map(_code_to_name)
) | set(
    _gi["hometeam"].dropna().map(_code_to_name)
))

# ─────────────────────────────────────────────
# Session state — pre-populate from precomputed results on first load
# ─────────────────────────────────────────────
if "ml_results" not in st.session_state:
    st.session_state["ml_results"] = _load_model_results()
    st.session_state["ml_feat_df"] = features_df
if "eval_backtests" not in st.session_state:
    st.session_state["eval_backtests"] = _load_eval_backtests()
if "schedule_selected_game" not in st.session_state:
    st.session_state["schedule_selected_game"] = None

(
    tab_schedule,
    tab_standings,
    tab_tbat,
    tab_tpitch,
    tab_bleaders,
    tab_pleaders,
    tab_h2h,
    tab_form,
    tab_features,
    tab_models,
    tab_evaluation,
    tab_savant,
    tab_history,
    tab_model_perf,
    tab_bankroll,
    tab_about,
) = st.tabs([
    "📅 Daily Schedule",
    "📊 Standings",
    "🏏 Team Batting",
    "⚾ Team Pitching",
    "🔝 Batting Leaders",
    "🔝 Pitching Leaders",
    "🆚 Head-to-Head",
    "📈 Rolling Form",
    "🧮 Betting Features",
    "🤖 Models",
    "📊 Evaluation",
    "🔬 Savant Research",
    "📋 Pick History",
    "🏆 Model Performance",
    "💰 Bankroll",
    "ℹ️ About",
])


# ══════════════════════════════════════════════
# TAB 0 — Daily Schedule
# ══════════════════════════════════════════════
with tab_schedule:
    _games_today = _fetch_todays_schedule()

    # ── Game Detail View ──────────────────────────────────────────────────────
    if st.session_state["schedule_selected_game"] is not None:
        g = st.session_state["schedule_selected_game"]
        away_full = g.get("away_name", "Away")
        home_full = g.get("home_name", "Home")
        away_retro = _MLB_TO_RETRO.get(away_full, away_full)
        home_retro = _MLB_TO_RETRO.get(home_full, home_full)

        if st.button("← Back to Schedule", key="back_to_schedule"):
            st.session_state["schedule_selected_game"] = None
            st.rerun()

        st.markdown(f"## {away_full} @ {home_full}")

        # ── Top row: key details ───────────────────────────────────────────
        status = g.get("status", "Scheduled")
        venue  = g.get("venue_name", "—")
        series = g.get("series_description", "")
        gtime_raw = g.get("game_datetime", "")
        if gtime_raw:
            try:
                dt_utc = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
                dt_et  = dt_utc - datetime.timedelta(hours=4)   # UTC → ET (rough)
                gtime_str = dt_et.strftime("%I:%M %p ET")
            except Exception:
                gtime_str = gtime_raw
        else:
            gtime_str = "TBD"

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Status", status)
        dc2.metric("Game Time", gtime_str)
        dc3.metric("Venue", venue)
        dc4.metric("Series", series or "Regular Season")

        st.divider()

        # ── Team Records ─────────────────────────────────────────────────
        _standings = _fetch_team_standings()
        away_rec = _standings.get(away_full, {})
        home_rec = _standings.get(home_full, {})
        if away_rec or home_rec:
            st.markdown("### 📋 Team Records (Current Season)")
            tr1, tr2 = st.columns(2)
            with tr1:
                st.markdown(f"**{away_full} (Away)**")
                if away_rec:
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("W-L", f"{away_rec['W']}-{away_rec['L']}")
                    rc2.metric("Win %", away_rec.get("pct", "—"))
                    rc3.metric("Streak", away_rec.get("streak", "—"))
                    rc4.metric("Last 10", away_rec.get("L10", "—"))
                else:
                    st.caption("Record unavailable.")
            with tr2:
                st.markdown(f"**{home_full} (Home)**")
                if home_rec:
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("W-L", f"{home_rec['W']}-{home_rec['L']}")
                    rc2.metric("Win %", home_rec.get("pct", "—"))
                    rc3.metric("Streak", home_rec.get("streak", "—"))
                    rc4.metric("Last 10", home_rec.get("L10", "—"))
                else:
                    st.caption("Record unavailable.")
            st.divider()

        # ── Probable Pitchers ─────────────────────────────────────────────
        away_sp = g.get("away_probable_pitcher", "TBD") or "TBD"
        home_sp = g.get("home_probable_pitcher", "TBD") or "TBD"
        st.markdown("### ⚾ Probable Pitchers")
        with st.spinner("Fetching pitcher stats…"):
            away_sp_stats = _fetch_pitcher_stats(away_sp)
            home_sp_stats = _fetch_pitcher_stats(home_sp)
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown(f"**{away_full} (Away)**")
            st.markdown(f"##### {away_sp}")
            if away_sp_stats:
                st.dataframe(
                    pd.DataFrame(away_sp_stats.items(), columns=["Stat", "Value"]),
                    hide_index=True, width="stretch",
                )
            elif away_sp != "TBD":
                st.caption("Stats not yet available for this season.")
        with pc2:
            st.markdown(f"**{home_full} (Home)**")
            st.markdown(f"##### {home_sp}")
            if home_sp_stats:
                st.dataframe(
                    pd.DataFrame(home_sp_stats.items(), columns=["Stat", "Value"]),
                    hide_index=True, width="stretch",
                )
            elif home_sp != "TBD":
                st.caption("Stats not yet available for this season.")

        st.divider()

        # ── Historical Head-to-Head ───────────────────────────────────────
        st.markdown("### 🆚 Head-to-Head History (2020–present)")
        with st.spinner("Loading H2H data…"):
            h2h_detail = head_to_head(away_retro, home_retro, 2020, 2025)

        if h2h_detail.empty:
            st.info(
                f"No historical matchups found between **{away_retro}** and **{home_retro}** "
                "in the 2020–2025 dataset."
            )
        else:
            a_w = int(h2h_detail["a_win"].sum())
            b_w = len(h2h_detail) - a_w
            tot = len(h2h_detail)
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric(f"{away_retro} wins", f"{a_w}  ({a_w/tot:.0%})")
            hc2.metric(f"{home_retro} wins", f"{b_w}  ({b_w/tot:.0%})")
            hc3.metric("Games played", tot)

            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Scatter(
                x=h2h_detail["date"], y=h2h_detail["a_runs"],
                mode="markers+lines", name=f"{away_retro} runs",
                line=dict(color="#1f77b4"),
            ))
            fig_h2h.add_trace(go.Scatter(
                x=h2h_detail["date"], y=h2h_detail["b_runs"],
                mode="markers+lines", name=f"{home_retro} runs",
                line=dict(color="#d62728"),
            ))
            fig_h2h.update_layout(
                title=f"{away_retro} vs {home_retro} — Runs per game",
                xaxis_title="Date", yaxis_title="Runs",
            )
            st.plotly_chart(fig_h2h, width="stretch")

            h2h_detail["season"] = h2h_detail["date"].dt.year
            by_szn = h2h_detail.groupby("season").agg(
                away_wins=("a_win", "sum"),
                games=("a_win", "count"),
            ).reset_index()
            by_szn["away_wpct"] = by_szn["away_wins"] / by_szn["games"]
            fig_szn = px.bar(
                by_szn, x="season", y="away_wpct",
                title=f"{away_retro} win % vs {home_retro} by season",
                labels={"away_wpct": f"{away_retro} W%", "season": "Season"},
                color="away_wpct", color_continuous_scale="RdYlGn",
            )
            fig_szn.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig_szn.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_szn, width="stretch")

            with st.expander("Full game log"):
                st.dataframe(
                    h2h_detail[["date", "visteam", "hometeam", "vruns", "hruns", "a_win"]]
                    .assign(date=lambda d: d["date"].dt.date)
                    .rename(columns={**READABLE_COLS, "a_win": f"{away_retro} Win"}),
                    hide_index=True, width="stretch",
                )

        st.divider()

        # ── Recent Form (rolling 10g) ─────────────────────────────────────
        st.markdown("### 📈 Recent Form — Last 20 Games")
        fc1, fc2 = st.columns(2)
        for col_ctx, team_retro, team_full in [
            (fc1, away_retro, away_full),
            (fc2, home_retro, home_full),
        ]:
            with col_ctx:
                st.markdown(f"**{team_full}**")
                with st.spinner(f"Loading {team_retro} form…"):
                    form = rolling_team_form(team_retro, 10, 2020, 2025)
                if form.empty:
                    st.caption("No form data available.")
                else:
                    recent = form.tail(20)
                    fig_form = px.line(
                        recent, x="date", y="roll_W_10",
                        title=f"{team_retro} — 10-game win rate",
                        labels={"roll_W_10": "Win rate (10g)", "date": ""},
                        color_discrete_sequence=["#1f77b4"] if team_retro == away_retro else ["#d62728"],
                    )
                    fig_form.add_hline(y=0.5, line_dash="dot", line_color="gray")
                    fig_form.update_layout(height=250, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_form, width="stretch")
                    # Last 5 games quick summary
                    last5 = form.tail(5)[["date", "RS", "RA", "W"]].copy()
                    last5["date"] = last5["date"].dt.strftime("%b %d")
                    last5["Result"] = last5["W"].map({1: "✔ W", 0: "✘ L"})
                    last5 = last5.rename(columns={"date": "Date", "RS": "R", "RA": "RA"})
                    st.dataframe(last5[["Date", "R", "RA", "Result"]], hide_index=True)

        st.divider()

        # ── Odds ─────────────────────────────────────────────────────────
        # ESPN scoreboard API: free, no key, no monthly quota, cached 30 min.
        # The Odds API CSV (optional): only written when you manually run
        #   fetch_current_odds() — the dashboard itself never calls that API.
        #   Monthly quota: 500 req. We used 1 today (341 remaining).
        st.markdown("### 💰 Odds")

        _espn_odds_list = _fetch_espn_odds()
        _game_espn: dict | None = None
        for _eo in _espn_odds_list:
            if (
                away_full.split()[-1].lower() in _eo["away_team"].lower()
                or home_full.split()[-1].lower() in _eo["home_team"].lower()
            ):
                _game_espn = _eo
                break

        if _game_espn:
            st.caption(f"Source: **{_game_espn['provider']}** (ESPN public API — free, no quota) · refreshes every 30 min")
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown("**💵 Moneyline**")
                st.metric(away_full, str(_game_espn["ml_away"]))
                st.metric(home_full, str(_game_espn["ml_home"]))
            with oc2:
                st.markdown("**📏 Run Line**")
                st.metric("Spread", str(_game_espn["details"]))
                st.metric("Home spread odds", str(_game_espn["spread_home"]))
            with oc3:
                st.markdown("**📊 Over/Under**")
                st.metric("Total", str(_game_espn["over_under"]))
                st.markdown(
                    f"Over: **{_game_espn['over_odds']}** &nbsp;/&nbsp; Under: **{_game_espn['under_odds']}**",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No ESPN odds found for this game yet. Odds typically open 1–2 days before game time.")

        # Optional: multi-book comparison from saved Odds API CSV
        _odds_csv = _load_latest_odds()
        if not _odds_csv.empty:
            with st.expander("📚 Multi-book comparison (saved Odds API data)"):
                st.caption(
                    "This data was saved from a manual run of `fetch_current_odds()`. "
                    "The dashboard never calls The Odds API automatically — your 500 req/month quota "
                    "is only consumed when you explicitly run that command."
                )
                mask_odds = (
                    _odds_csv["home_team"].str.contains(home_full.split()[-1], case=False, na=False)
                    | _odds_csv["away_team"].str.contains(away_full.split()[-1], case=False, na=False)
                )
                game_odds = _odds_csv[mask_odds].copy()
                if game_odds.empty:
                    st.caption("No multi-book data for this matchup in the saved file.")
                else:
                    for market_key, market_label in [
                        ("h2h",     "💵 Moneyline"),
                        ("spreads", "📏 Run Line"),
                        ("totals",  "📊 Over/Under"),
                    ]:
                        mdf = game_odds[game_odds["market"] == market_key]
                        if mdf.empty:
                            continue
                        st.markdown(f"**{market_label}**")
                        pivot = (
                            mdf[["bookmaker", "outcome_name", "outcome_price", "outcome_point"]]
                            .sort_values("bookmaker")
                        )
                        st.dataframe(
                            pivot.rename(columns={
                                "bookmaker": "Book", "outcome_name": "Side",
                                "outcome_price": "Odds", "outcome_point": "Line",
                            }),
                            hide_index=True, width="stretch",
                        )

    # ── Schedule List View ────────────────────────────────────────────────────
    else:
        today_str = datetime.date.today().strftime("%A, %B %d, %Y")
        st.subheader(f"Today's Schedule — {today_str}")

        if not _games_today:
            st.info(
                "No MLB games scheduled today, or the MLB Stats API is unreachable. "
                "Check back on a game day."
            )
        else:
            st.caption(f"{len(_games_today)} game{'s' if len(_games_today) != 1 else ''} today")

            # Status badge colours
            _status_badge = {
                "Final": "🏁",
                "Game Over": "🏁",
                "In Progress": "🔴 LIVE",
                "Scheduled": "🕐",
                "Pre-Game": "⏳",
                "Warmup": "⏳",
                "Delayed": "⚠️",
                "Suspended": "⚠️",
                "Postponed": "🚫",
                "Cancelled": "🚫",
            }

            for idx, g in enumerate(_games_today):
                away_name  = g.get("away_name", "Away")
                home_name  = g.get("home_name", "Home")
                away_sp    = g.get("away_probable_pitcher", "TBD") or "TBD"
                home_sp    = g.get("home_probable_pitcher", "TBD") or "TBD"
                venue      = g.get("venue_name", "—")
                status     = g.get("status", "Scheduled")
                status_icon = _status_badge.get(status, "")
                gtime_raw  = g.get("game_datetime", "")

                if gtime_raw:
                    try:
                        dt_utc = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
                        dt_et  = dt_utc - datetime.timedelta(hours=4)
                        gtime_str = dt_et.strftime("%I:%M %p ET")
                    except Exception:
                        gtime_str = "TBD"
                else:
                    gtime_str = "TBD"

                # Show score only if the game is in progress or complete
                score_str = ""
                score_status = str(status).lower()
                if score_status in ("final", "game over", "in progress", "live", "completed"):
                    if g.get("away_score") is not None and g.get("home_score") is not None:
                        score_str = f"  **{g['away_score']} – {g['home_score']}**"

                with st.container(border=True):
                    sc1, sc2, sc3 = st.columns([4, 3, 2])

                    with sc1:
                        st.markdown(
                            f"**{away_name}** @ **{home_name}**{score_str}  \n"
                            f"<small>🏟 {venue}</small>",
                            unsafe_allow_html=True,
                        )

                    with sc2:
                        st.markdown(
                            f"<small>🕐 {gtime_str} &nbsp;|&nbsp; {status_icon} {status}</small>  \n"
                            f"<small>Away SP: {away_sp}</small>  \n"
                            f"<small>Home SP: {home_sp}</small>",
                            unsafe_allow_html=True,
                        )

                    with sc3:
                        if st.button(
                            "View Details →",
                            key=f"sched_detail_{idx}",
                            width="stretch",
                        ):
                            st.session_state["schedule_selected_game"] = g
                            st.rerun()


# ══════════════════════════════════════════════
# TAB 1 — Season Standings
# ══════════════════════════════════════════════
with tab_standings:
    st.subheader("Season Standings")

    year_sel = st.selectbox(
        "Select season", sorted(standings["season"].unique(), reverse=True), key="standings_year"
    )
    df_yr = standings[standings["season"] == year_sel].copy()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Teams", len(df_yr))
    if not df_yr.empty:
        best = df_yr.loc[df_yr["WPct"].idxmax()]
        worst = df_yr.loc[df_yr["WPct"].idxmin()]
        col_b.metric("Best record", f"{best['team']} ({best['W']}-{best['L']})")
        col_c.metric("Worst record", f"{worst['team']} ({worst['W']}-{worst['L']})")

    st.dataframe(
        df_yr[["team", "G", "W", "L", "WPct", "PythWPct", "RS", "RA", "RD", "RD_per_G", "RS_per_G", "RA_per_G"]]
        .sort_values("WPct", ascending=False)
        .reset_index(drop=True)
        .rename(columns=READABLE_COLS),
        width='stretch',
        hide_index=True,
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_yr.sort_values("WPct", ascending=True),
            x="WPct", y="team",
            orientation="h",
            title=f"Win % — {year_sel}",
            color="WPct",
            color_continuous_scale="RdYlGn",
            labels={"WPct": "Win %", "team": ""},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig2 = px.scatter(
            df_yr,
            x="RS_per_G", y="RA_per_G",
            text="team",
            title=f"Runs Scored vs Allowed per game — {year_sel}",
            color="WPct",
            color_continuous_scale="RdYlGn",
            labels={"RS_per_G": "RS / G", "RA_per_G": "RA / G"},
        )
        fig2.add_hline(y=df_yr["RA_per_G"].mean(), line_dash="dot", line_color="gray")
        fig2.add_vline(x=df_yr["RS_per_G"].mean(), line_dash="dot", line_color="gray")
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Win % Trend — Select Teams")
    top_teams = standings.groupby("team")["WPct"].mean().nlargest(10).index.tolist()
    standings_teams = sorted(standings["team"].unique())
    team_filter = st.multiselect("Select teams", standings_teams, default=top_teams[:6], key="trend_teams")
    if team_filter:
        trend_df = standings[standings["team"].isin(team_filter)]
        fig3 = px.line(
            trend_df, x="season", y="WPct", color="team",
            title="Season Win % over time",
            labels={"WPct": "Win %", "season": "Season"},
        )
        st.plotly_chart(fig3, width='stretch')


# ══════════════════════════════════════════════
# TAB 2 — Team Batting
# ══════════════════════════════════════════════
with tab_tbat:
    st.subheader("Team Batting")

    bat_year = st.selectbox("Season", sorted(tbat["season"].unique(), reverse=True), key="tbat_year")
    bat_metric = st.selectbox("Sort by", ["BA", "SLG", "HR", "R", "SB", "K"], key="tbat_metric")
    df_bat = tbat[tbat["season"] == bat_year].sort_values(bat_metric, ascending=False).reset_index(drop=True)

    st.dataframe(
        df_bat[["team", "G", "PA", "AB", "R", "H", "doubles", "triples", "HR", "RBI", "BB", "K", "SB", "BA", "SLG"]]
        .rename(columns={"doubles": "2B", "triples": "3B"})
        .rename(columns=READABLE_COLS),
        width='stretch',
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_bat.nlargest(15, bat_metric).sort_values(bat_metric),
            x=bat_metric, y="team", orientation="h",
            title=f"{bat_metric} leaders — {bat_year}",
            color=bat_metric, color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        avg_metric = tbat.groupby("season")[bat_metric].mean().reset_index()
        fig2 = px.line(avg_metric, x="season", y=bat_metric, title=f"League-Avg {bat_metric} over time")
        st.plotly_chart(fig2, width='stretch')

    fig3 = px.scatter(
        df_bat, x="BA", y="HR", text="team", color="R",
        color_continuous_scale="Viridis",
        title=f"BA vs HR — {bat_year}",
        labels={"BA": "Batting Average", "HR": "Home Runs", "R": "Runs"},
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, width='stretch')


# ══════════════════════════════════════════════
# TAB 3 — Team Pitching
# ══════════════════════════════════════════════
with tab_tpitch:
    st.subheader("Team Pitching")

    pitch_year = st.selectbox("Season", sorted(tpitch["season"].unique(), reverse=True), key="tpitch_year")
    pitch_metric = st.selectbox(
        "Sort by (lower = better for ERA/WHIP)",
        ["ERA", "WHIP", "K9", "BB9", "HR9"],
        key="pitch_metric",
    )
    asc = pitch_metric in ("ERA", "WHIP", "BB9", "HR9")
    df_pt = tpitch[tpitch["season"] == pitch_year].sort_values(pitch_metric, ascending=asc).reset_index(drop=True)

    st.dataframe(
        df_pt[["team", "G", "IP", "HA", "HRA", "RA", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "HR9"]]
        .rename(columns=READABLE_COLS),
        width='stretch',
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        bar_df = df_pt.nsmallest(15, "ERA") if asc else df_pt.nlargest(15, pitch_metric)
        fig = px.bar(
            bar_df.sort_values(pitch_metric, ascending=not asc),
            x=pitch_metric, y="team", orientation="h",
            title=f"{pitch_metric} — {pitch_year}",
            color=pitch_metric,
            color_continuous_scale="RdYlGn_r" if asc else "RdYlGn",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig2 = px.scatter(
            df_pt, x="ERA", y="WHIP", text="team", color="K9",
            color_continuous_scale="Viridis",
            title=f"ERA vs WHIP — {pitch_year}",
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, width='stretch')

    league_avg = tpitch.groupby("season")[["ERA", "WHIP", "K9"]].mean().reset_index()
    fig3 = px.line(
        league_avg.melt(id_vars="season", value_vars=["ERA", "WHIP", "K9"]),
        x="season", y="value", color="variable", facet_col="variable",
        facet_col_wrap=3,
        title="League-Average ERA / WHIP / K9 over time",
    )
    fig3.update_yaxes(matches=None)
    fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig3, width='stretch')


# ══════════════════════════════════════════════
# TAB 4 — Batting Leaders
# ══════════════════════════════════════════════
with tab_bleaders:
    st.subheader("Individual Batting Leaders")

    bl_year = st.selectbox("Season", sorted(bleaders["season"].unique(), reverse=True), key="bl_year")
    bl_metric = st.selectbox("Sort by", ["BA", "SLG", "HR", "RBI", "SB", "BB", "K"], key="bl_metric")
    bl_top = st.slider("Show top N", 10, 50, 25, key="bl_top")

    df_bl = (
        bleaders[bleaders["season"] == bl_year]
        .sort_values(bl_metric, ascending=(bl_metric == "K"))
        .head(bl_top)
        .reset_index(drop=True)
    )

    st.dataframe(
        df_bl[["full_name", "team", "PA", "AB", "H", "doubles", "triples", "HR", "RBI", "BB", "K", "SB", "BA", "SLG"]]
        .rename(columns={"doubles": "2B", "triples": "3B"})
        .rename(columns=READABLE_COLS),
        width='stretch',
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_bl.sort_values(bl_metric, ascending=True),
            x=bl_metric, y="full_name", orientation="h",
            title=f"Top {bl_top} by {bl_metric} — {bl_year}",
            color=bl_metric, color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig2 = px.scatter(
            bleaders[bleaders["season"] == bl_year],
            x="BA", y="SLG", size="PA",
            hover_name="full_name",
            color="HR", color_continuous_scale="Reds",
            title=f"BA vs SLG (size=PA) — {bl_year}",
        )
        st.plotly_chart(fig2, width='stretch')


# ══════════════════════════════════════════════
# TAB 5 — Pitching Leaders
# ══════════════════════════════════════════════
with tab_pleaders:
    st.subheader("Individual Pitching Leaders")

    pl_year = st.selectbox("Season", sorted(pleaders["season"].unique(), reverse=True), key="pl_year")
    pl_options = ["ERA", "WHIP", "K9", "BB9", "K_BB", "SO", "IP"]
    pl_metric = st.selectbox(
        "Sort by", pl_options,
        format_func=lambda x: READABLE_COLS.get(x, x),
        key="pl_metric",
    )
    pl_top = st.slider("Show top N", 10, 50, 25, key="pl_top")
    asc_p = pl_metric in ("ERA", "WHIP", "BB9")

    with st.expander("Stats data dictionary"):
        dict_cols = ["GS", "IP", "H", "HR", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "K_BB"]
        dict_desc = [
            "Games started",
            "Innings pitched",
            "Hits allowed",
            "Home runs allowed",
            "Earned runs allowed",
            "Walks allowed",
            "Strikeouts",
            "Earned run average",
            "Walks+Hits per inning pitched",
            "Strikeouts per 9 innings",
            "Walks per 9 innings",
            "Strikeout-to-walk ratio",
        ]
        dict_df = pd.DataFrame({"column": dict_cols, "description": dict_desc})
        st.dataframe(dict_df.rename(columns=READABLE_COLS), width='stretch', hide_index=True)

    df_pl = (
        pleaders[pleaders["season"] == pl_year]
        .sort_values(pl_metric, ascending=asc_p)
        .head(pl_top)
        .reset_index(drop=True)
    )

    st.dataframe(
        df_pl[["full_name", "team", "GS", "IP", "H", "HR", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "K_BB"]]
        .rename(columns={"full_name": "Pitcher"})
        .rename(columns=READABLE_COLS),
        width='stretch',
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_pl.sort_values(pl_metric, ascending=not asc_p),
            x=pl_metric, y="full_name", orientation="h",
            title=f"Top {pl_top} by {pl_metric} — {pl_year}",
            color=pl_metric,
            color_continuous_scale="OrRd" if asc_p else "Greens",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig2 = px.scatter(
            pleaders[pleaders["season"] == pl_year],
            x="ERA", y="WHIP", size="IP",
            hover_name="full_name",
            color="K9", color_continuous_scale="Viridis",
            title=f"ERA vs WHIP (size=IP) — {pl_year}",
        )
        st.plotly_chart(fig2, width='stretch')


# ══════════════════════════════════════════════
# TAB 6 — Head-to-Head
# ══════════════════════════════════════════════
with tab_h2h:
    st.subheader("Head-to-Head History")

    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=0, key="h2h_a")
    team_b = col2.selectbox("Team B", teams, index=min(1, len(teams) - 1), key="h2h_b")

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        with st.spinner("Loading matchup data…"):
            h2h = head_to_head(team_a, team_b, min_year, max_year)

        if h2h.empty:
            st.info("No matchups found for the selected teams and year range.")
        else:
            a_wins = int(h2h["a_win"].sum())
            b_wins = len(h2h) - a_wins
            total = len(h2h)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{team_a} wins", f"{a_wins} ({a_wins/total:.1%})")
            mc2.metric(f"{team_b} wins", f"{b_wins} ({b_wins/total:.1%})")
            mc3.metric("Total games", total)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=h2h["date"], y=h2h["a_runs"],
                mode="markers+lines", name=f"{team_a} runs",
                line=dict(color="#1f77b4"),
            ))
            fig.add_trace(go.Scatter(
                x=h2h["date"], y=h2h["b_runs"],
                mode="markers+lines", name=f"{team_b} runs",
                line=dict(color="#d62728"),
            ))
            fig.update_layout(
                title=f"{team_a} vs {team_b} — Runs by game",
                xaxis_title="Date", yaxis_title="Runs",
            )
            st.plotly_chart(fig, width='stretch')

            h2h["season"] = h2h["date"].dt.year
            by_season = h2h.groupby("season").agg(
                a_wins=("a_win", "sum"),
                games=("a_win", "count"),
            ).reset_index()
            by_season["a_wpct"] = by_season["a_wins"] / by_season["games"]

            fig2 = px.bar(
                by_season, x="season", y="a_wpct",
                title=f"{team_a} win % vs {team_b} by season",
                labels={"a_wpct": f"{team_a} W%", "season": "Season"},
                color="a_wpct", color_continuous_scale="RdYlGn",
            )
            fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig2, width='stretch')

            with st.expander("Raw game log"):
                st.dataframe(
                    h2h[["date", "visteam", "hometeam", "vruns", "hruns", "a_win"]]
                    .assign(date=lambda d: d["date"].dt.date)
                    .rename(columns={**READABLE_COLS, "a_win": f"{team_a} Win"}),
                    width='stretch',
                    hide_index=True,
                )


# ══════════════════════════════════════════════
# TAB 7 — Rolling Form
# ══════════════════════════════════════════════
with tab_form:
    st.subheader("Team Rolling Form")

    form_team = st.selectbox("Team", teams, key="form_team")
    form_window = st.slider("Rolling window (games)", 5, 30, 10, key="form_window")

    with st.spinner(f"Computing {form_window}-game rolling form for {form_team}…"):
        form_df = rolling_team_form(form_team, form_window, min_year, max_year)

    if form_df.empty:
        st.info("No data found.")
    else:
        rs_col = f"roll_RS_{form_window}"
        ra_col = f"roll_RA_{form_window}"
        rw_col = f"roll_W_{form_window}"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=form_df["date"], y=form_df[rs_col], name="Avg RS", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=form_df["date"], y=form_df[ra_col], name="Avg RA", line=dict(color="red")))
        fig.update_layout(
            title=f"{form_team} — {form_window}-game rolling runs scored/allowed",
            xaxis_title="Date", yaxis_title="Runs (rolling avg)",
        )
        st.plotly_chart(fig, width='stretch')

        fig2 = px.line(
            form_df, x="date", y=rw_col,
            title=f"{form_team} — {form_window}-game rolling win rate",
            labels={rw_col: "Win rate", "date": "Date"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")
        st.plotly_chart(fig2, width='stretch')

        st.markdown("#### Recent 20 games")
        recent_20 = (
            form_df.tail(20)[["date", "RS", "RA", "W", rs_col, ra_col, rw_col, "roll_RD"]]
            .sort_values("date", ascending=False)
            .reset_index(drop=True)
            .assign(date=lambda d: d["date"].dt.date)
            .rename(columns={
                **READABLE_COLS,
                rs_col: f"Avg RS ({form_window}g)",
                ra_col: f"Avg RA ({form_window}g)",
                rw_col: f"Win Rate ({form_window}g)",
                "roll_RD": "Rolling Run Diff",
            })
        )
        st.dataframe(recent_20, width='stretch', hide_index=True)


# ══════════════════════════════════════════════
# TAB 8 — Betting Features
# ══════════════════════════════════════════════
with tab_features:
    st.subheader("Engineered Betting Features")
    st.markdown(
        "Feature matrix built from season-level stats — designed as inputs for ML models."
    )

    all_standings = _pre["standings"]

    feat_season = st.selectbox(
        "Season",
        sorted(all_standings["season"].unique().tolist(), reverse=True),
        key="feat_season",
    )

    with st.spinner("Building feature matrix…"):
        gi = load_gameinfo(min_year=feat_season, max_year=feat_season)

    if gi.empty:
        st.info("No games in selected season.")
    else:
        ts_yr = all_standings[all_standings["season"] == feat_season].set_index("team")

        rows = []
        for _, g in gi.iterrows():
            vt, ht = g["visteam"], g["hometeam"]
            if vt not in ts_yr.index or ht not in ts_yr.index:
                continue
            rows.append({
                "date": g["date"],
                "visitor": vt,
                "home_team": ht,
                "home_WPct": ts_yr.loc[ht, "WPct"],
                "vis_WPct": ts_yr.loc[vt, "WPct"],
                "home_RS_G": ts_yr.loc[ht, "RS_per_G"],
                "home_RA_G": ts_yr.loc[ht, "RA_per_G"],
                "vis_RS_G": ts_yr.loc[vt, "RS_per_G"],
                "vis_RA_G": ts_yr.loc[vt, "RA_per_G"],
                "home_RD_G": ts_yr.loc[ht, "RD_per_G"],
                "vis_RD_G": ts_yr.loc[vt, "RD_per_G"],
                "home_PythWPct": ts_yr.loc[ht, "PythWPct"],
                "vis_PythWPct": ts_yr.loc[vt, "PythWPct"],
                "WPct_diff": ts_yr.loc[ht, "WPct"] - ts_yr.loc[vt, "WPct"],
                "PythWPct_diff": ts_yr.loc[ht, "PythWPct"] - ts_yr.loc[vt, "PythWPct"],
                "RS_advantage": ts_yr.loc[ht, "RS_per_G"] - ts_yr.loc[vt, "RS_per_G"],
                "RA_advantage": ts_yr.loc[vt, "RA_per_G"] - ts_yr.loc[ht, "RA_per_G"],
                "daynight": g.get("daynight", ""),
                "attendance": g.get("attendance", None),
                "temp": g.get("temp", None),
                "windspeed": g.get("windspeed", None),
                "home_win": int(g["wteam"] == ht),
                "total_runs": g["total_runs"],
            })

        feat_df = pd.DataFrame(rows)
        st.markdown(f"**{len(feat_df)} games** in {feat_season} with full feature coverage.")
        display_feat = feat_df.head(50).copy()
        display_feat["date"] = display_feat["date"].dt.date
        # human-readable headers for feature matrix
        feat_rename = {
            **READABLE_COLS,
            "visitor": "Visitor",
            "home_team": "Home",
            "home_WPct": "Home Win %",
            "vis_WPct": "Visitor Win %",
            "home_RS_G": "Home RS/G",
            "home_RA_G": "Home RA/G",
            "vis_RS_G": "Visitor RS/G",
            "vis_RA_G": "Visitor RA/G",
            "home_PythWPct": "Home Pyth W%",
            "vis_PythWPct": "Visitor Pyth W%",
            "WPct_diff": "Win % Diff",
            "PythWPct_diff": "Pyth W% Diff",
            "RS_advantage": "RS Advantage",
            "RA_advantage": "RA Advantage",
            "daynight": "Day/Night",
            "attendance": "Attendance",
            "temp": "Temperature",
            "windspeed": "Wind Speed",
            "home_win": "Home Win?",
            "total_runs": "Total Runs",
        }
        display_feat = display_feat.rename(columns=feat_rename)
        st.dataframe(display_feat, width='stretch', hide_index=True)

        num_feats = [
            "home_WPct", "vis_WPct", "WPct_diff", "PythWPct_diff",
            "home_RS_G", "home_RA_G", "vis_RS_G", "vis_RA_G",
            "RS_advantage", "RA_advantage", "home_win", "total_runs",
        ]
        corr = feat_df[num_feats].corr()
        # rename axis labels for readability in the heatmap
        readable_feats = [
            "Home Win %", "Visitor Win %", "Win % Diff", "Pyth W% Diff",
            "Home RS/G", "Home RA/G", "Visitor RS/G", "Visitor RA/G",
            "RS Advantage", "RA Advantage", "Home Win?", "Total Runs",
        ]
        corr.index = readable_feats
        corr.columns = readable_feats
        fig = px.imshow(
            corr,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        st.plotly_chart(fig, width='stretch')
        # data dictionary for the features
        dict_df = pd.DataFrame({
            "column": num_feats,
            "description": [
                "Home Win %, season standings",
                "Visitor Win %, season standings",
                "Home WPct minus visitor WPct",
                "Home Pythagorean WPct minus visitor",
                "Home runs scored per game",
                "Home runs allowed per game",
                "Visitor runs scored per game",
                "Visitor runs allowed per game",
                "Home RS/G minus visitor RS/G",
                "Visitor RA/G minus home RA/G",
                "Indicator (1=home team won)",
                "Total runs scored in game",
            ],
        })
        with st.expander("Feature data dictionary"):
            st.dataframe(dict_df.rename(columns=READABLE_COLS), width='stretch', hide_index=True)

        st.markdown("#### Home Win % over time (all seasons in range)")
        gi_all = load_gameinfo(min_year, max_year)
        gi_all["home_win"] = (gi_all["wteam"] == gi_all["hometeam"]).astype(int)
        hfa = gi_all.groupby("season")["home_win"].mean().reset_index()
        hfa.columns = ["season", "home_win_pct"]
        fig2 = px.line(
            hfa, x="season", y="home_win_pct",
            title="Home Field Advantage — Win % by season",
            labels={"home_win_pct": "Home Win %", "season": "Season"},
        )
        fig2.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="50%")
        st.plotly_chart(fig2, width='stretch')

        if gi_all["temp"].notna().sum() > 100:
            temp_df = gi_all.dropna(subset=["temp", "total_runs"])
            temp_df = temp_df[temp_df["temp"] > 0]
            fig3 = px.scatter(
                temp_df.sample(min(5000, len(temp_df)), random_state=42),
                x="temp", y="total_runs",
                trendline="lowess",
                title="Temperature vs Total Runs (sample)",
                labels={"temp": "Temp (°F)", "total_runs": "Total Runs"},
                opacity=0.4,
            )
            st.plotly_chart(fig3, width='stretch')

        with st.expander("Download feature CSV"):
            st.download_button(
                label="Download features.csv",
                data=feat_df.to_csv(index=False),
                file_name=f"features_{feat_season}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════
# TAB 9 — ML Betting Models
# ══════════════════════════════════════════════
with tab_models:
    st.subheader("ML Betting Models")
    st.markdown(
        """
        Three XGBoost classifiers trained on **2020+ Retrosheet** game data.
        All models use a **chronological train/test split** (no lookahead).

        | Model | Target | Features |
        |-------|--------|----------|
        | **Moneyline** | P(home team wins) | Team stats, SP stats, weather |
        | **Spread** | P(home covers −1.5) | Same as moneyline |
        | **Over/Under** | P(total > expected) | Same + expected-total offset |

        > **Note on expected totals:** Without live odds data the Over/Under model
        > uses each team's season RS/G sum as a surrogate "posted total."
        > Real edge calculations become available once live moneyline feeds are wired in.
        """
    )

    use_lgbm = False  # reserved for future admin re-train

    if False:  # re-train disabled; models are pre-loaded at startup
        with st.spinner("Building feature matrix…"):
            ml_df = features_df
            st.session_state["ml_feat_df"] = ml_df

        prog = st.progress(0, text="Training moneyline model…")
        r_ml = train_moneyline_model(ml_df)
        prog.progress(33, text="Training spread model…")
        r_sp = train_spread_model(ml_df)
        prog.progress(66, text="Training over/under model…")
        r_ou = train_totals_model(ml_df, use_lightgbm=use_lgbm)
        prog.progress(100, text="Done ✓")
        prog.empty()

        st.session_state["ml_results"] = {
            "moneyline": r_ml,
            "spread":    r_sp,
            "totals":    r_ou,
        }
        st.success(
            f"Models trained on {r_ml['train_size']:,} games • "
            f"tested on {r_ml['test_size']:,} games"
        )

    results = st.session_state["ml_results"]
    if results is None:
        st.info("Pre-trained results not found. Click **Re-train Models** above to train all three models.")
    else:
        # ── Model metrics ─────────────────────────────────────────────────────────
        st.markdown("### Model Performance (test set)")
    
        _model_labels = {
            "moneyline": "🏆 Moneyline (P home win)",
            "spread":    "📏 Spread (P home covers −1.5)",
            "totals":    "📈 Over/Under (P went over)",
        }
        c1, c2, c3 = st.columns(3)
        for col, key in zip([c1, c2, c3], ["moneyline", "spread", "totals"]):
            m = results[key]["metrics"]
            col.markdown(f"**{_model_labels[key]}**")
            col.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")
            col.metric("Accuracy",  f"{m['accuracy']:.4f}")
            col.metric("Brier",     f"{m['brier_score']:.4f}")
            col.metric("Log Loss",  f"{m['log_loss']:.4f}")
    
        with st.expander("ℹ️ Metric guide"):
            st.markdown(
                """
                - **ROC-AUC** – probability that the model ranks a 'positive' game higher than a
                  'negative' game. 0.5 = random; 1.0 = perfect. MLB games are inherently noisy,
                  so 0.60–0.65 is a solid baseline with season-level features.
                - **Accuracy** – fraction of correct binary picks on the test set.
                - **Brier Score** – mean squared error of predicted probabilities. Lower is better.
                  Perfect calibration ≈ 0.25 for a 50/50 event.
                - **Log Loss** – cross-entropy between predicted probabilities and outcomes.
                  Lower is better.
                """
            )
    
        # ── Feature importances ───────────────────────────────────────────────────
        st.markdown("### Feature Importances")
        imp_tab_ml, imp_tab_sp, imp_tab_ou = st.tabs([
            "Moneyline", "Spread", "Over/Under"
        ])
    
        _FEAT_LABELS = {
            "WPct_diff":     "Win % Diff",
            "PythWPct_diff": "Pythagorean W% Diff",
            "sp_ERA_gap":    "SP ERA Gap (away−home)",
            "home_WPct":     "Home Win %",
            "away_WPct":     "Away Win %",
            "home_PythWPct": "Home Pyth W%",
            "away_PythWPct": "Away Pyth W%",
            "home_RS_G":     "Home RS / G",
            "home_RA_G":     "Home RA / G",
            "away_RS_G":     "Away RS / G",
            "away_RA_G":     "Away RA / G",
            "home_RD_G":     "Home Run Diff / G",
            "away_RD_G":     "Away Run Diff / G",
            "home_ERA":      "Home ERA",
            "away_ERA":      "Away ERA",
            "ERA_diff":      "ERA Diff (away−home)",
            "home_WHIP":     "Home WHIP",
            "away_WHIP":     "Away WHIP",
            "WHIP_diff":     "WHIP Diff (away−home)",
            "home_K9":       "Home K/9",
            "away_K9":       "Away K/9",
            "home_BA":       "Home BA",
            "away_BA":       "Away BA",
            "home_SLG":      "Home SLG",
            "away_SLG":      "Away SLG",
            "home_sp_ERA":   "Home SP ERA",
            "away_sp_ERA":   "Away SP ERA",
            "home_sp_WHIP":  "Home SP WHIP",
            "away_sp_WHIP":  "Away SP WHIP",
            "home_sp_K9":    "Home SP K/9",
            "away_sp_K9":    "Away SP K/9",
            "temp":          "Temperature (°F)",
            "windspeed":     "Wind Speed",
            "is_day":        "Day game?",
            "exp_total":     "Expected Total Runs",
        }
    
        def _importance_chart(model_key: str, top_n: int = 20) -> None:
            imp = (
                results[model_key]["importances"]
                .head(top_n)
                .copy()
            )
            imp["label"] = imp["feature"].map(lambda x: _FEAT_LABELS.get(x, x))
            fig = px.bar(
                imp.sort_values("importance"),
                x="importance", y="label",
                orientation="h",
                title=f"Top {top_n} features — {_model_labels[model_key]}",
                labels={"importance": "Importance", "label": "Feature"},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")
    
        with imp_tab_ml:
            _importance_chart("moneyline")
        with imp_tab_sp:
            _importance_chart("spread")
        with imp_tab_ou:
            _importance_chart("totals")
    
        # ── Backtest — predicted probability distributions ────────────────────────
        st.markdown("### Backtest: Predicted Probability Distributions")
        dist_tab_ml, dist_tab_sp, dist_tab_ou = st.tabs([
            "Moneyline", "Spread", "Over/Under"
        ])
    
        def _prob_dist_chart(model_key: str, prob_col: str, label: str) -> None:
            tdf = results[model_key]["test_df"].copy()
            fig = px.histogram(
                tdf, x=prob_col,
                color="correct",
                barmode="overlay",
                nbins=40,
                title=f"{_model_labels[model_key]} — predicted prob by correctness",
                labels={prob_col: label, "correct": "Correct?"},
                color_discrete_map={0: "#d62728", 1: "#2ca02c"},
                opacity=0.7,
            )
            fig.update_layout(legend_title="Correct prediction")
            st.plotly_chart(fig, width="stretch")
    
        with dist_tab_ml:
            _prob_dist_chart("moneyline", "pred_prob", "Model predicted probability (home win)")
        with dist_tab_sp:
            _prob_dist_chart("spread", "pred_prob", "Model predicted probability (home covers −1.5)")
        with dist_tab_ou:
            _prob_dist_chart("totals", "pred_prob_over", "Model predicted probability (went over)")
    
        # ── Calibration curves ────────────────────────────────────────────────────
        st.markdown("### Calibration: Predicted vs Actual Win Rate")
        cal_tab_ml, cal_tab_sp, cal_tab_ou = st.tabs([
            "Moneyline", "Spread", "Over/Under"
        ])
    
        def _calibration_chart(model_key: str, prob_col: str, actual_col: str, label: str) -> None:
            tdf = results[model_key]["test_df"][[prob_col, actual_col]].copy()
            tdf["bin"] = pd.cut(tdf[prob_col], bins=10)
            cal = tdf.groupby("bin", observed=False).agg(
                mean_pred=(prob_col, "mean"),
                actual_rate=(actual_col, "mean"),
                count=(prob_col, "count"),
            ).reset_index().dropna()
            fig = px.scatter(
                cal,
                x="mean_pred", y="actual_rate",
                size="count",
                title=f"{_model_labels[model_key]} — calibration curve",
                labels={"mean_pred": f"Mean predicted {label}", "actual_rate": "Actual rate"},
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dot", color="gray"),
            )
            st.plotly_chart(fig, width="stretch")
    
        with cal_tab_ml:
            _calibration_chart("moneyline", "pred_prob", "home_win", "Model predicted probability (home win)")
        with cal_tab_sp:
            _calibration_chart("spread", "pred_prob", "home_cover", "P(cover −1.5)")
        with cal_tab_ou:
            _calibration_chart("totals", "pred_prob_over", "went_over", "P(over)")
    
        # ── Backtest sample table ─────────────────────────────────────────────────
        st.markdown("### Backtest Sample — Recent Test-Set Games")
    
        bt_model = st.selectbox(
            "Model", ["moneyline", "spread", "totals"],
            format_func=lambda x: _model_labels[x],
            key="bt_model_sel",
        )
        n_show = st.slider("Games to display", 25, 200, 50, key="bt_n_show")
    
        bt_df = results[bt_model]["test_df"].tail(n_show).copy()
        bt_df["date"] = pd.to_datetime(bt_df["date"]).dt.date
    
        # convert model pick codes to human-readable strings and
        # render correctness as a checkmark column
        if bt_model == "moneyline" and "pred_win" in bt_df.columns:
            bt_df["pred_win"] = bt_df["pred_win"].map({1: "Home", 0: "Away"})
        if bt_model == "spread" and "pred_cover" in bt_df.columns:
            bt_df["pred_cover"] = bt_df["pred_cover"].map({
                1: "Home −1.5",
                0: "Away +1.5",
            })
    
        if "correct" in bt_df.columns:
            bt_df["correct"] = bt_df["correct"].astype(bool).map({True: "✔", False: ""})

        # show actual outcomes as checkmarks rather than raw 0/1
        for col in ("home_win", "home_cover", "went_over"):
            if col in bt_df.columns:
                bt_df[col] = bt_df[col].astype(bool).map({True: "✔", False: ""})

        # choose which columns to display depending on model type
        if bt_model == "moneyline":
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "hruns": "H Runs", "vruns": "V Runs",
                "home_win": "Actually Won?",
                "pred_prob": "Predicted home-win probability", "pred_win": "Model Pick", "correct": "Correct?",
            }
        elif bt_model == "spread":
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "home_margin": "Margin", "home_cover": "Actually Covered?",
                "pred_prob": "P(cover −1.5)", "pred_cover": "Model Pick", "correct": "Correct?",
            }
        else:
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "total_runs": "Total Runs", "exp_total": "Exp Total",
                "went_over": "Went Over?",
                "pred_prob_over": "P(over)", "pick_side": "Pick", "correct": "Correct?",
            }
    
        existing = [c for c in display_cols if c in bt_df.columns]
        st.dataframe(
            bt_df[existing]
            .reset_index(drop=True)
            .rename(columns=display_cols),
            width="stretch",
            hide_index=True,
        )
    
        # ── Download backtest ─────────────────────────────────────────────────────
        with st.expander("Download backtest CSVs"):
            c_dl1, c_dl2, c_dl3 = st.columns(3)
            for col, key, label in zip(
                [c_dl1, c_dl2, c_dl3],
                ["moneyline", "spread", "totals"],
                ["moneyline", "spread", "totals"],
            ):
                col.download_button(
                    label=f"Download {label}.csv",
                    data=results[key]["test_df"].to_csv(index=False),
                    file_name=f"backtest_{label}.csv",
                    mime="text/csv",
                )
    
    # ────────────────────────────────────────────────────────────────────────────
# TAB 10 — Model Evaluation (backtesting & analysis)
# ────────────────────────────────────────────────────────────────────────────
with tab_evaluation:
    st.subheader("Model Evaluation")
    st.markdown(
        """Use the trained models (from the **Models** tab) and the full
        feature matrix to run walk‑forward backtests, calibration analyses,
        and profitability reports. The evaluation can be expensive, so results
        are cached in session state.
        """
    )

    if False:  # re-run disabled; evaluation is pre-loaded at startup
        feat_df = st.session_state["ml_feat_df"].copy()
        # use same backtest logic as script but simpler for UI
        from xgboost import XGBClassifier

        def _train_xgb(X, y):
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

        def _predict_xgb(model, X):
            return model.predict_proba(X.fillna(0))[:, 1]

        from src.models.features import MONEYLINE_FEATURES, TOTALS_FEATURES

        # prepare dataframes (gid → game_id for backtester)
        ml_bt_cols = ["date", "gid", "home_win"] + [
            c for c in MONEYLINE_FEATURES if c in feat_df.columns
        ]
        ml_bt = feat_df[ml_bt_cols].rename(columns={"gid": "game_id"}).dropna()
        tot_bt_cols = ["date", "gid", "went_over"] + [
            c for c in TOTALS_FEATURES if c in feat_df.columns
        ]
        tot_bt = feat_df[tot_bt_cols].rename(columns={"gid": "game_id"}).dropna()

        ml_backtest = walk_forward_backtest(
            features_df=ml_bt,
            train_fn=_train_xgb,
            predict_fn=_predict_xgb,
            target_col="home_win",
            odds_col="home_ml",
            pick_type="underdog",
            model_name="moneyline",
            min_edge=0.02,
            train_window_games=1200,
            test_window_games=200,
            step_size=100,
        )
        tot_backtest = walk_forward_backtest(
            features_df=tot_bt,
            train_fn=_train_xgb,
            predict_fn=_predict_xgb,
            target_col="went_over",
            odds_col="total_line",
            pick_type="over_under",
            model_name="totals",
            min_edge=0.02,
            train_window_games=1200,
            test_window_games=200,
            step_size=100,
        )

        st.session_state["eval_backtests"] = {
            "moneyline": ml_backtest,
            "totals": tot_backtest,
        }
        st.success("Evaluation complete. Scroll down to view results.")

    if st.session_state["eval_backtests"]:
        # show leaderboard
        leaderboard = []
        for name, bt in st.session_state["eval_backtests"].items():
            leaderboard.append(bt.summary())
        lb_df = pd.DataFrame(leaderboard).sort_values("roi", ascending=False)
        if "period" in lb_df.columns:
            # Remove time-of-day / ranges from period strings (e.g. "2020-07-23 00:00:00 to 2025-09-28 00:00:00").
            # Take the first 10 characters (YYYY-MM-DD) when possible.
            lb_df["period"] = (
                lb_df["period"].astype(str)
                .str.strip()
                .str.slice(0, 10)
            )

        st.markdown("### Backtest Leaderboard")
        st.dataframe(
            lb_df.rename(columns={
                "model":         "Model",
                "pick_type":     "Pick Type",
                "period":        "Period",
                "total_bets":    "Bets",
                "wins":          "Wins",
                "losses":        "Losses",
                "pushes":        "Pushes",
                "win_rate":      "Win Rate",
                "total_units":   "Units",
                "max_drawdown":  "Max Drawdown",
                "roi":           "ROI",
            }),
            hide_index=True,
            width="stretch",
        )

        # calibration plot for each
        st.markdown("### Calibration Charts")
        for name, bt in st.session_state["eval_backtests"].items():
            arr_true = [1 if b.result == "win" else 0 for b in bt.bets]
            arr_prob = [b.predicted_prob for b in bt.bets]
            cal_data = calibration_plot_data(np.array(arr_true), np.array(arr_prob))
            fig = px.scatter(
                x=cal_data["mean_predicted"],
                y=cal_data["fraction_positive"],
                size=[1]*len(cal_data["mean_predicted"]),
                title=f"{name.capitalize()} Calibration",
                labels={"x": "Mean pred", "y": "Actual rate"},
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# TAB 11 — Savant Research (Monte Carlo Feature Selection)
# ══════════════════════════════════════════════
with tab_savant:
    st.subheader("Savant Feature Research — Monte Carlo Selection")
    st.markdown(
        "Results of a Monte Carlo search over Baseball Savant advanced metrics. "
        "1,000 random trials were run, each sampling a random subset of batter and pitcher "
        "Savant columns, training XGBoost with TimeSeriesSplit cross-validation across three "
        "bet targets (moneyline, run line, total). Features are ranked by how often they "
        "appeared in the **top 10% of trials by ROC-AUC**."
    )

    mc_ranking = _pre.get("mc_ranking")
    mc_trials  = _pre.get("mc_trials")
    savant_metrics = _pre.get("savant_metrics")
    savant_imps    = _pre.get("savant_imps")

    if mc_ranking is None or mc_ranking.empty:
        st.info(
            "Monte Carlo results not found. "
            "Run `python scripts/monte_carlo_features.py --trials 1000` to generate them."
        )
    else:
        # ── Summary metrics ───────────────────────────────────────────────
        n_valid = len(mc_trials) if mc_trials is not None else "—"
        top_cutoff = mc_trials["mean_auc"].quantile(0.90) if mc_trials is not None else None

        BASELINE_AUC = {"moneyline": 0.6253, "spread": 0.6304, "totals": 0.6157}

        col_m, col_s, col_t, col_v = st.columns(4)
        col_m.metric("Baseline Moneyline AUC", f"{BASELINE_AUC['moneyline']:.4f}",
                     help="Retrosheet-only features, no Savant")
        col_s.metric("Baseline Spread AUC",    f"{BASELINE_AUC['spread']:.4f}",
                     help="Retrosheet-only features, no Savant")
        col_t.metric("Baseline Totals AUC",    f"{BASELINE_AUC['totals']:.4f}",
                     help="Retrosheet-only features, no Savant")
        col_v.metric("Valid Trials", f"{n_valid:,}" if isinstance(n_valid, int) else n_valid,
                     help="Trials with ≥500 rows after joining Savant to game data")

        # ── Savant-enriched model performance ─────────────────────────────
        if savant_metrics is not None and not savant_metrics.empty:
            st.markdown("---")
            st.markdown("#### Savant-Enriched Model Performance")
            # st.caption(
            #     "Trained with top MC-ranked Savant features merged onto the Retrosheet baseline. "
            #     "Rebuilds every Monday via GitHub Actions → `build_savant_model.py`."
            # )
            perf_cols = st.columns(3)
            for i, model_name in enumerate(["moneyline", "spread", "totals"]):
                row = savant_metrics[savant_metrics["model"] == model_name]
                if row.empty:
                    continue
                row = row.iloc[0]
                auc      = float(row["roc_auc"])
                baseline = BASELINE_AUC[model_name]
                delta    = auc - baseline
                perf_cols[i].metric(
                    label=f"{model_name.capitalize()} AUC (Savant)",
                    value=f"{auc:.4f}",
                    delta=f"{delta:+.4f} vs baseline",
                    delta_color="normal",
                )

            row0 = savant_metrics.iloc[0]
            with st.expander("Features used in Savant model"):
                bat_used = str(row0.get("savant_bat_features", "")).split(",")
                pit_used = str(row0.get("savant_pit_features", "")).split(",")
                c1, c2 = st.columns(2)
                c1.markdown("**Batter features**")
                for f in bat_used:
                    c1.markdown(f"- `{f.strip()}`")
                c2.markdown("**Pitcher features**")
                for f in pit_used:
                    c2.markdown(f"- `{f.strip()}`")

            if savant_imps is not None and not savant_imps.empty:
                st.markdown("##### Feature Importances — Top 20 per model")
                imp_tabs = st.tabs(["Moneyline", "Spread", "Totals"])
                for tab_i, model_name in zip(imp_tabs, ["moneyline", "spread", "totals"]):
                    with tab_i:
                        df_imp = savant_imps[savant_imps["model"] == model_name].nlargest(20, "importance")
                        fig_imp = px.bar(
                            df_imp.sort_values("importance"),
                            x="importance", y="feature",
                            orientation="h",
                            title=f"{model_name.capitalize()} — Savant model feature importance",
                            labels={"importance": "XGBoost Importance", "feature": "Feature"},
                            color="importance",
                            color_continuous_scale="Viridis",
                        )
                        fig_imp.update_layout(coloraxis_showscale=False, height=480)
                        st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info(
                "Savant model not yet built. "
                "Run `python scripts/build_savant_model.py` after the MC run to train it."
            )

        # ── AUC distribution across MC trials ─────────────────────────────
        if mc_trials is not None:
            st.markdown("---")
            st.markdown("#### AUC Distribution Across All MC Trials")
            auc_plot_cols = [c for c in mc_trials.columns if c.endswith("_auc") and c != "mean_auc"]
            auc_long = mc_trials[auc_plot_cols + ["mean_auc"]].melt(var_name="target", value_name="auc")
            auc_long["target"] = auc_long["target"].str.replace("_auc", "").str.capitalize()
            fig_dist = px.box(
                auc_long[auc_long["target"] != "Mean"],
                x="target", y="auc",
                color="target",
                points=False,
                title="Trial AUC by Bet Target (1,000 trials)",
                labels={"target": "Bet Target", "auc": "ROC-AUC"},
                color_discrete_map={
                    "Moneyline": "#636EFA",
                    "Spread":    "#EF553B",
                    "Totals":    "#00CC96",
                },
            )
            for target_name, baseline in [
                ("Moneyline", BASELINE_AUC["moneyline"]),
                ("Spread",    BASELINE_AUC["spread"]),
                ("Totals",    BASELINE_AUC["totals"]),
            ]:
                fig_dist.add_hline(
                    y=baseline, line_dash="dot", line_color="gray",
                    annotation_text=f"{target_name} baseline",
                    annotation_position="right",
                )
            # move legend below plot to avoid overlap with data
            fig_dist.update_layout(
                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_dist, width='stretch')

        # ── Feature ranking charts ────────────────────────────────────────
        st.markdown("---")
        bat_ranks = mc_ranking[mc_ranking["type"] == "batter"].head(20).copy()
        pit_ranks = mc_ranking[mc_ranking["type"] == "pitcher"].head(20).copy()

        bat_ranks["appearance_pct"] = bat_ranks["appearance_rate"] * 100
        pit_ranks["appearance_pct"] = pit_ranks["appearance_rate"] * 100

        st.markdown("#### Top Batter Features (by appearance in top-10% trials)")
        fig_bat = px.bar(
            bat_ranks.sort_values("appearance_pct"),
            x="appearance_pct", y="feature",
            orientation="h",
            title="Batter Feature Selection Frequency",
            labels={"appearance_pct": "Appearance Rate in Top Trials (%)", "feature": "Savant Column"},
            color="appearance_pct",
            color_continuous_scale="Blues",
        )
        fig_bat.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_bat, width='stretch')

        st.markdown("#### Top Pitcher Features (by appearance in top-10% trials)")
        fig_pit = px.bar(
            pit_ranks.sort_values("appearance_pct"),
            x="appearance_pct", y="feature",
            orientation="h",
            title="Pitcher Feature Selection Frequency",
            labels={"appearance_pct": "Appearance Rate in Top Trials (%)", "feature": "Savant Column"},
            color="appearance_pct",
            color_continuous_scale="Reds",
        )
        fig_pit.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_pit, width='stretch')

        # ── Full ranking table ────────────────────────────────────────────
        with st.expander("Full feature ranking table"):
            display_rank = mc_ranking.copy()
            display_rank["appearance_rate"] = (display_rank["appearance_rate"] * 100).round(1)
            display_rank = display_rank.rename(columns={
                "feature":               "Savant Column",
                "type":                  "Type",
                "top_trial_appearances": "Appearances in Top Trials",
                "appearance_rate":       "Rate (%)",
            })
            st.dataframe(display_rank, hide_index=True, width='stretch')

        # ── Data dictionary ───────────────────────────────────────────────
        with st.expander("📖  Data dictionary — all Savant columns"):
            st.markdown("All columns below can appear as batter stats (stat earned) "
                        "or pitcher stats (stat allowed), except where noted.")
            dict_data = [
                # Expected stats
                ("xba",                    "both",    "Expected Batting Average based on exit velocity + launch angle; removes park and defense bias."),
                ("xslg",                   "both",    "Expected Slugging Percentage based on quality of contact."),
                ("xwoba",                  "both",    "Expected Weighted On-Base Average — comprehensive offensive value, contact-quality adjusted."),
                ("xobp",                   "both",    "Expected On-Base Percentage."),
                ("xiso",                   "both",    "Expected Isolated Power (xSLG − xBA)."),
                ("xera",                   "pitcher", "Expected ERA based on contact quality allowed; strips out sequencing and BABIP luck."),
                ("wobacon",                "both",    "wOBA on contact only — excludes walks and strikeouts."),
                ("xwobacon",               "both",    "Expected wOBA on contact."),
                ("bacon",                  "both",    "Batting Average on Contact."),
                ("xbacon",                 "both",    "Expected Batting Average on Contact."),
                ("xbadiff",                "both",    "xBA − actual BA. Positive = hitter has been unlucky (expect regression up)."),
                ("xslgdiff",               "both",    "xSLG − actual SLG. Positive = power regression candidate."),
                ("wobadiff",               "both",    "xwOBA − actual wOBA. Overall luck/regression signal."),
                # Bat tracking
                ("avg_swing_speed",        "batter",  "Average bat speed (mph) at contact across all swings."),
                ("fast_swing_rate",        "batter",  "Fraction of swings above 75 mph bat speed."),
                ("squared_up_contact",     "batter",  "Fraction of contacts classified as 'squared up' per bat-tracking sensors."),
                ("squared_up_swing",       "batter",  "Fraction of all swings resulting in a squared-up contact."),
                ("avg_swing_length",       "batter",  "Average arc length of the swing path (feet)."),
                ("swords",                 "batter",  "Rate of swing-and-miss on pitches well outside the zone ('showing the sword')."),
                ("attack_angle",           "batter",  "Average bat attack angle at contact (degrees); 10–18° is considered optimal for loft."),
                ("ideal_angle_rate",       "batter",  "Fraction of swings within the ideal 5–30° attack angle window."),
                # Batted-ball quality
                ("exit_velocity_avg",      "both",    "Average exit velocity (mph) on all batted balls."),
                ("launch_angle_avg",       "both",    "Average launch angle (degrees) on all batted balls."),
                ("sweet_spot_percent",     "both",    "Fraction of batted balls with 8–32° launch angle ('sweet spot')."),
                ("barrel_batted_rate",     "both",    "Fraction of PAs producing a 'barrel': exit velo ≥98 mph at optimal launch angle."),
                ("solidcontact_percent",   "both",    "Fraction of batted balls rated as solid contact."),
                ("flareburner_percent",    "both",    "Fraction classified as flares (bloops) or burners (weak grounders that find holes)."),
                ("poorlyunder_percent",    "batter",  "Fraction of poorly-hit balls — popped up badly."),
                ("poorlytopped_percent",   "batter",  "Fraction of poorly-hit balls — topped or slow roller."),
                ("poorlyweak_percent",     "batter",  "Fraction of poorly-hit balls — weak contact."),
                ("hard_hit_percent",       "both",    "Fraction of batted balls at ≥95 mph exit velocity."),
                ("avg_best_speed",         "both",    "Average of a hitter's top-50% exit velocities — ceiling speed indicator."),
                ("avg_hyper_speed",        "both",    "Average of top-10% exit velocities; 'maximum effort' swing speed."),
                # Plate discipline
                ("k_percent",              "both",    "Strikeout rate (K%)."),
                ("bb_percent",             "both",    "Walk rate (BB%)."),
                ("z_swing_percent",        "both",    "Swing rate on pitches inside the strike zone."),
                ("z_swing_miss_percent",   "both",    "Whiff rate on pitches inside the strike zone."),
                ("oz_swing_percent",       "both",    "Chase rate: swing% on pitches outside the zone."),
                ("oz_swing_miss_percent",  "both",    "Whiff rate on pitches outside the zone."),
                ("oz_contact_percent",     "both",    "Contact rate when swinging at pitches outside the zone."),
                ("iz_contact_percent",     "both",    "Contact rate on pitches inside the zone."),
                ("meatball_swing_percent", "both",    "Swing rate on 'meatball' pitches (dead center of zone, easiest to hit)."),
                ("meatball_percent",       "pitcher", "Fraction of all pitches classified as meatballs — command/location metric."),
                ("edge_percent",           "both",    "Fraction of pitches on the edge of the strike zone."),
                ("whiff_percent",          "both",    "Overall swing-and-miss rate across all swings."),
                ("swing_percent",          "both",    "Overall swing rate."),
                ("f_strike_percent",       "both",    "Fraction of PAs where the first pitch is a called or swinging strike."),
                # Batted-ball direction/type
                ("pull_percent",           "both",    "Fraction of batted balls pulled (toward batter's dominant side)."),
                ("straightaway_percent",   "both",    "Fraction of batted balls hit up the middle."),
                ("opposite_percent",       "both",    "Fraction of batted balls hit to opposite field. High rate signals contact-first profile."),
                ("groundballs_percent",    "both",    "Groundball rate."),
                ("flyballs_percent",       "both",    "Flyball rate."),
                ("linedrives_percent",     "both",    "Line drive rate."),
                ("popups_percent",         "both",    "Infield popup rate — high rate is a red flag for batters, good sign for pitchers."),
                # Speed (batters only)
                ("sprint_speed",           "batter",  "Average sprint speed (ft/sec) on all-out running plays — raw athleticism."),
                ("hp_to_1b",               "batter",  "Home-to-first time (sec) on groundball plays — proxy for foot speed."),
                # Arsenal (pitchers only)
                ("velocity",               "pitcher", "Average fastball velocity (mph) across all pitch types."),
                ("ff_avg_speed",           "pitcher", "Four-seam fastball average velocity."),
                ("ff_avg_spin",            "pitcher", "Four-seam fastball average spin rate (rpm)."),
                ("ff_avg_break_x",         "pitcher", "Four-seam horizontal break (inches); positive = arm-side run."),
                ("ff_avg_break_z",         "pitcher", "Four-seam induced vertical break (inches); higher = more 'rise' effect."),
                ("sl_avg_speed",           "pitcher", "Slider average velocity."),
                ("sl_avg_spin",            "pitcher", "Slider average spin rate."),
                ("ch_avg_speed",           "pitcher", "Changeup average velocity."),
                ("ch_avg_spin",            "pitcher", "Changeup average spin rate."),
                ("cu_avg_speed",           "pitcher", "Curveball average velocity."),
                ("cu_avg_spin",            "pitcher", "Curveball average spin rate."),
                ("release_extension",      "pitcher", "How far in front of the pitching rubber the pitcher releases (feet); higher = shorter reaction time for batters."),
                ("arm_angle",              "pitcher", "Arm slot angle at release (degrees from horizontal); affects pitch movement and deception profile."),
            ]
            dict_df = pd.DataFrame(dict_data, columns=["Column", "Applies to", "Description"])
            cols_in_ranking = set(mc_ranking["feature"].tolist())
            shown = dict_df[dict_df["Column"].isin(cols_in_ranking)].copy()
            rest  = dict_df[~dict_df["Column"].isin(cols_in_ranking)].copy()
            shown.insert(0, "In MC Ranking", "✅")
            rest.insert(0, "In MC Ranking", "")
            st.dataframe(
                pd.concat([shown, rest], ignore_index=True),
                hide_index=True,
                width='stretch',
            )

        # ── Download ─────────────────────────────────────────────────────
        st.download_button(
            label="Download mc_feature_ranking.csv",
            data=mc_ranking.to_csv(index=False),
            file_name="mc_feature_ranking.csv",
            mime="text/csv",
        )

        # ── Methodology note ──────────────────────────────────────────────
        with st.expander("Methodology: why 6 batter + 4 pitcher per trial?"):
            st.markdown("""
**Why sample subsets instead of using all features at once?**

The Monte Carlo approach uses *random subsets* per trial for two reasons:

1. **Overfitting prevention** — With ~110 total Savant columns on top of 32 Retrosheet
   features, XGBoost would have more features than useful training rows per fold (~4,000).
   Testing all features at once inflates in-sample accuracy while making the model fragile.

2. **Stability-based ranking** — A single full-feature model's importance scores are noisy
   (collinearity, random splits). The MC frequency approach asks: *across many random subsets,
   which features consistently appear in the best-performing combinations?* That is a more
   reliable selection signal than one model's SHAP values.

**Why 6 batter + 4 pitcher?**
Tunable via `--n-bat` / `--n-pit`. The defaults create enough trial-to-trial variation
(~44 batter and ~35 pitcher candidates available) while keeping each trial's model lean.

**Automation pipeline:**
MC runs every Monday 05:00 UTC → `mc_feature_ranking.csv` updates →
`build_savant_model.py` trains with the new top features → metrics update here.
            """)



# ══════════════════════════════════════════════
# TAB — Pick History
# ══════════════════════════════════════════════
with tab_history:
    st.subheader("Pick History")
    st.markdown(
        "Backtest history for all modeled picks. "
        "Once the daily pipeline runs, live picks will appear here automatically."
    )

    _bt = st.session_state["eval_backtests"]
    if _bt is None:
        st.info(
            "No pick history yet. Run the daily pipeline or backtest scripts to populate data."
        )
    else:
        # Flatten all bets across models into a single DataFrame
        _hist_rows: list[dict] = []
        for _model_name, _bt_result in _bt.items():
            for _b in _bt_result.bets:
                _hist_rows.append({
                    "model":          _model_name,
                    "date":           _b.date,
                    "game_id":        _b.game_id,
                    "pick_type":      _b.pick_type,
                    "confidence":     _b.confidence,
                    "predicted_prob": _b.predicted_prob,
                    "edge":           _b.edge,
                    "american_odds":  _b.american_odds,
                    "result":         _b.result,
                    "profit_units":   _b.profit_units,
                })
        _hist_df = pd.DataFrame(_hist_rows)
        if _hist_df.empty:
            st.info("No picks data available.")
        else:
            _hist_df["model"] = _hist_df["model"].str.title()
            _hist_df["date"] = pd.to_datetime(_hist_df["date"])
            _PICK_TYPE_LABELS = {"totals": "Totals", "over_under": "Over/Under"}
            _hist_df["pick_type"] = _hist_df["pick_type"].map(
                lambda x: _PICK_TYPE_LABELS.get(x, x.title())
            )
            _hist_df["confidence"] = _hist_df["confidence"].str.title()

            # ── Filters ────────────────────────────────────────────────────
            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            with _fc1:
                _models_avail = ["All"] + sorted(_hist_df["model"].unique().tolist())
                _sel_model = st.selectbox("Model", _models_avail, key="hist_model_filter")
            with _fc2:
                _sel_result = st.selectbox(
                    "Result", ["All", "win", "loss"], key="hist_result_filter"
                )
            with _fc3:
                _conf_opts = ["All"] + sorted(_hist_df["confidence"].dropna().unique().tolist())
                _sel_conf = st.selectbox("Confidence", _conf_opts, key="hist_conf_filter")
            with _fc4:
                _pt_opts = ["All"] + sorted(_hist_df["pick_type"].unique().tolist())
                _sel_pt = st.selectbox("Pick Type", _pt_opts, key="hist_pt_filter")

            _filtered = _hist_df.copy()
            if _sel_model != "All":
                _filtered = _filtered[_filtered["model"] == _sel_model]
            if _sel_result != "All":
                _filtered = _filtered[_filtered["result"] == _sel_result]
            if _sel_conf != "All":
                _filtered = _filtered[_filtered["confidence"] == _sel_conf]
            if _sel_pt != "All":
                _filtered = _filtered[_filtered["pick_type"] == _sel_pt]

            # ── Summary metrics ─────────────────────────────────────────────
            _total_bets  = len(_filtered)
            _wins        = int((_filtered["result"] == "win").sum())
            _losses      = int((_filtered["result"] == "loss").sum())
            _win_rate    = _wins / _total_bets if _total_bets > 0 else 0.0
            _total_units = float(_filtered["profit_units"].sum())
            _avg_edge    = float(_filtered["edge"].mean()) if _total_bets > 0 else 0.0

            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _m1.metric("Total Picks", _total_bets)
            _m2.metric("Record", f"{_wins}–{_losses}")
            _m3.metric("Win Rate", f"{_win_rate:.1%}")
            _m4.metric("Total Units", f"{_total_units:+.2f}")
            _m5.metric("Avg Edge", f"{_avg_edge:.1%}")

            st.divider()

            # ── Cumulative P&L chart ─────────────────────────────────────────
            _cum_df = (
                _filtered
                .sort_values("date")
                .copy()
            )
            _cum_df["cumulative_units"] = _cum_df.groupby("model")["profit_units"].cumsum()
            if not _cum_df.empty:
                _pnl_fig = px.line(
                    _cum_df,
                    x="date",
                    y="cumulative_units",
                    color="model",
                    title="Cumulative P&L (units)",
                    labels={
                        "date": "Date",
                        "cumulative_units": "Cumulative Units",
                        "model": "Model",
                    },
                )
                _pnl_fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(_pnl_fig, width="stretch")

            # ── Detailed ledger ──────────────────────────────────────────────
            st.markdown("#### Detailed Ledger")
            _display_df = _filtered[[
                "date", "model", "pick_type",
                "confidence", "predicted_prob", "edge",
                "american_odds", "result", "profit_units",
            ]].copy()
            _display_df["date"] = _display_df["date"].dt.strftime("%Y-%m-%d")
            _display_df["predicted_prob"] = (
                (_display_df["predicted_prob"] * 100).round(1).astype(str) + "%"
            )
            _display_df["edge"] = (
                (_display_df["edge"] * 100).round(1).astype(str) + "%"
            )
            _display_df["american_odds"] = _display_df["american_odds"].apply(
                lambda x: f"{int(x):,}"
            )
            _display_df = _display_df.rename(columns={
                "date":           "Date",
                "model":          "Model",
                "pick_type":      "Pick Type",
                "confidence":     "Confidence",
                "predicted_prob": "Pred. Prob",
                "edge":           "Edge",
                "american_odds":  "Odds",
                "result":         "Result",
                "profit_units":   "P&L (Units)",
            })
            _display_df = _display_df.sort_values("Date", ascending=False).reset_index(drop=True)
            st.dataframe(_display_df, hide_index=True, width="stretch")


# ══════════════════════════════════════════════
# TAB — Model Performance
# ══════════════════════════════════════════════
with tab_model_perf:
    st.subheader("Model Performance")
    st.markdown(
        "Backtest-derived profitability metrics — focused on betting outcomes "
        "rather than raw ML accuracy."
    )

    _bt = st.session_state["eval_backtests"]
    if _bt is None:
        st.info("No model performance data yet. Run the backtest scripts to populate data.")
    else:
        # ── Leaderboard ─────────────────────────────────────────────────────
        st.markdown("### Leaderboard")
        _lb_rows = [_btr.summary() for _btr in _bt.values()]
        _lb_df = pd.DataFrame(_lb_rows).sort_values("roi", ascending=False)
        _lb_df["model"] = _lb_df["model"].str.title()
        _PICK_TYPE_LABELS_MP = {"totals": "Totals", "over_under": "Over/Under"}
        if "pick_type" in _lb_df.columns:
            _lb_df["pick_type"] = _lb_df["pick_type"].map(
                lambda x: _PICK_TYPE_LABELS_MP.get(x, x.title())
            )
        if "period" in _lb_df.columns:
            _lb_df["period"] = (
                _lb_df["period"].astype(str)
                .str.replace(r" \d{2}:\d{2}:\d{2}", "", regex=True)
                .str.strip()
            )
        st.dataframe(
            _lb_df.rename(columns={
                "model":        "Model",
                "pick_type":    "Pick Type",
                "total_bets":   "Bets",
                "wins":         "Wins",
                "losses":       "Losses",
                "pushes":       "Pushes",
                "win_rate":     "Win Rate",
                "total_units":  "Units",
                "max_drawdown": "Max Drawdown",
                "roi":          "ROI",
                "period":       "Period",
            }),
            hide_index=True,
            width="stretch",
        )

        st.divider()

        # ── Build a combined bets DataFrame ─────────────────────────────────
        _mp_rows: list[dict] = []
        for _mn, _btr in _bt.items():
            for _b in _btr.bets:
                _mp_rows.append({
                    "model":        _mn,
                    "date":         pd.to_datetime(_b.date),
                    "profit_units": _b.profit_units,
                    "result":       _b.result,
                    "confidence":   _b.confidence,
                })
        _mp_df = pd.DataFrame(_mp_rows)
        _mp_df["model"] = _mp_df["model"].str.title()
        _mp_df["confidence"] = _mp_df["confidence"].str.title()

        if not _mp_df.empty:
            # ── Cumulative P&L comparison ────────────────────────────────────
            st.markdown("### Cumulative P&L by Model")
            _mp_df_sorted = _mp_df.sort_values(["model", "date"])
            _mp_df_sorted = _mp_df_sorted.copy()
            _mp_df_sorted["cum_units"] = _mp_df_sorted.groupby("model")["profit_units"].cumsum()
            _mp_fig = px.line(
                _mp_df_sorted,
                x="date",
                y="cum_units",
                color="model",
                title="Cumulative Units by Model",
                labels={"date": "Date", "cum_units": "Cumulative Units", "model": "Model"},
            )
            _mp_fig.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(_mp_fig, width="stretch")

            st.divider()

            # ── Confidence tier breakdown ────────────────────────────────────
            st.markdown("### Performance by Confidence Tier")
            _tier_grp = (
                _mp_df
                .groupby(["model", "confidence"])
                .agg(
                    bets=("profit_units", "count"),
                    wins=("result", lambda x: (x == "win").sum()),
                    total_units=("profit_units", "sum"),
                )
                .reset_index()
            )
            _tier_grp["win_rate"] = (_tier_grp["wins"] / _tier_grp["bets"]).round(3)
            _tier_grp["roi"]      = (_tier_grp["total_units"] / _tier_grp["bets"]).round(3)
            st.dataframe(
                _tier_grp.rename(columns={
                    "model":       "Model",
                    "confidence":  "Tier",
                    "bets":        "Bets",
                    "wins":        "Wins",
                    "win_rate":    "Win Rate",
                    "total_units": "Units",
                    "roi":         "ROI/Bet",
                }),
                hide_index=True,
                width="stretch",
            )
            _tier_bar = px.bar(
                _tier_grp,
                x="confidence",
                y="roi",
                color="model",
                barmode="group",
                title="ROI per Bet by Confidence Tier",
                labels={"confidence": "Confidence", "roi": "ROI per Bet", "model": "Model"},
            )
            st.plotly_chart(_tier_bar, width="stretch")


# ══════════════════════════════════════════════
# TAB — Bankroll
# ══════════════════════════════════════════════
with tab_bankroll:
    st.subheader("Bankroll Management")

    _bk_left, _bk_right = st.columns(2)

    # ── Kelly Calculator ─────────────────────────────────────────────────────
    with _bk_left:
        st.markdown("### Kelly Criterion Calculator")
        _kc1, _kc2 = st.columns(2)
        with _kc1:
            _bankroll_size = st.number_input(
                "Starting Bankroll ($)", min_value=100, max_value=1_000_000,
                value=1000, step=100, key="bk_bankroll",
            )
            _unit_size = st.number_input(
                "Unit Size ($)", min_value=1, max_value=100_000,
                value=50, step=5, key="bk_unit",
            )
        with _kc2:
            _kelly_conf = st.selectbox(
                "Confidence Tier", ["High", "Medium", "Low"], key="bk_conf"
            )
            _kelly_odds = st.number_input(
                "American Odds", min_value=-2000, max_value=2000,
                value=-110, step=5, key="bk_odds",
            )

        _kelly_prob = st.slider(
            "Win Probability",
            min_value=0.30, max_value=0.80, value=0.55, step=0.01,
            format="%.2f", key="bk_prob",
        )

        _full_kelly = _kelly_fraction(_kelly_prob, _kelly_odds)
        _half_kelly = _full_kelly / 2.0
        _qtr_kelly  = _full_kelly / 4.0
        _tier_mults = {"High": 0.5, "Medium": 0.25, "Low": 0.10}
        _tier_frac  = _full_kelly * _tier_mults[_kelly_conf]

        _ck1, _ck2 = st.columns(2)
        with _ck1:
            st.metric("Full Kelly",    f"{_full_kelly:.2%}")
            st.metric("Half Kelly",    f"{_half_kelly:.2%}")
            st.metric("Quarter Kelly", f"{_qtr_kelly:.2%}")
        with _ck2:
            st.metric(f"{_kelly_conf.capitalize()} Tier Fraction", f"{_tier_frac:.2%}")
            st.metric("Bet Size ($)",  f"${_bankroll_size * _tier_frac:,.2f}")
            _bet_units = round(_bankroll_size * _tier_frac / _unit_size, 2) if _unit_size else 0
            st.metric("Units to Bet",  f"{_bet_units:.2f}")

        with st.expander("Kelly Formula Reference"):
            st.latex(r"f^* = \frac{b \cdot p - q}{b}")
            st.markdown(
                "**b** = decimal odds − 1 &nbsp;·&nbsp; "
                "**p** = win probability &nbsp;·&nbsp; "
                "**q** = 1 − p  \n"
                "Negative values mean no edge — don't bet. "
                "Half-Kelly is recommended to reduce variance."
            )

    # ── Historical Bankroll Simulation ───────────────────────────────────────
    with _bk_right:
        st.markdown("### Historical Simulation")
        _bt = st.session_state["eval_backtests"]
        if _bt is None:
            st.info("Run backtests to view the historical bankroll simulation.")
        else:
            _sim_start = st.number_input(
                "Starting Bankroll ($)", min_value=100, max_value=1_000_000,
                value=1000, step=100, key="bk_sim_bankroll",
            )
            _sim_unit = st.number_input(
                "Unit Size ($)", min_value=1, max_value=100_000,
                value=50, step=5, key="bk_sim_unit",
            )
            _sim_rows: list[dict] = []
            for _mn, _btr in _bt.items():
                _running = float(_sim_start)
                for _b in sorted(_btr.bets, key=lambda x: str(x.date)):
                    _running += _b.profit_units * _sim_unit
                    _sim_rows.append({
                        "model":    _mn.title(),
                        "date":     pd.to_datetime(_b.date),
                        "bankroll": _running,
                    })
            _sim_df = pd.DataFrame(_sim_rows)
            if not _sim_df.empty:
                _sim_fig = px.line(
                    _sim_df, x="date", y="bankroll", color="model",
                    title="Simulated Bankroll Growth",
                    labels={"date": "Date", "bankroll": "Bankroll ($)", "model": "Model"},
                )
                _sim_fig.add_hline(
                    y=_sim_start, line_dash="dot", line_color="gray",
                    annotation_text="Starting bankroll",
                )
                st.plotly_chart(_sim_fig, width="stretch")

                _final = (
                    _sim_df.groupby("model")["bankroll"].last().reset_index()
                )
                _final["return_pct"] = (
                    (_final["bankroll"] - _sim_start) / _sim_start * 100
                ).round(1)
                st.dataframe(
                    _final.rename(columns={
                        "model":      "Model",
                        "bankroll":   "Final Bankroll ($)",
                        "return_pct": "Return %",
                    }),
                    hide_index=True,
                    width="stretch",
                )

with tab_about:
    st.subheader("About Betting Cleanup")
    st.markdown(
        """
**Betting Cleanup** is an MLB betting analytics platform that generates daily wagering
recommendations, backtests predictive models, and presents results through this dashboard.

---

### How Picks Are Generated

1. **Data Ingestion** — MLB game logs, player stats, odds, and weather data are fetched
   daily from multiple free sources (MLB Stats API, Retrosheet, Baseball Savant).
2. **Feature Engineering** — Batting and pitching performance metrics are aggregated
   into a feature matrix spanning recent seasons.
3. **Model Prediction** — XGBoost models generate win probabilities for each bet type.
4. **Edge Calculation** — Predicted probabilities are compared against market-implied odds
   to identify positive-expected-value opportunities.
5. **Pick Filtering** — Only picks meeting minimum edge thresholds are surfaced.

---

### Models

| Model | Target | Min Edge |
|-------|--------|----------|
| Underdog Moneyline | Upset probability (+120 or longer) | 2% |
| Run Line (Spread) | Covers −1.5 / +1.5 | 2% |
| Totals (Over/Under) | Goes over or under posted total | 2% |

Each model is trained using walk-forward cross-validation to prevent data leakage.

---

### Confidence Tiers

| Tier | Edge | Kelly Sizing |
|------|------|--------------|
| **HIGH** | > 6% | Half-Kelly |
| **MEDIUM** | 3–6% | Quarter-Kelly |
| **LOW** | 1–3% | Tracked only |

---

### Data Sources

- **MLB Stats API** (`statsapi`) — Schedules, standings, pitcher stats (free, no key)
- **ESPN API** — Live odds and scoreboard data (free public endpoint)
- **Retrosheet** — Historical game logs and play-by-play data
- **Baseball Savant** — Statcast metrics via `pybaseball`
- **The Odds API** — Multi-book odds (API key required; 500 req/month free tier)

---

### Technology Stack

Python 3.11 · Streamlit · XGBoost · LightGBM · pandas · Plotly · pyarrow

---
        """
    )
    # st.warning(
    #     "**Responsible Gambling Notice:** This tool is for research and entertainment purposes "
    #     "only. Past model performance does not guarantee future results. Never bet more than "
    #     "you can afford to lose. If gambling is affecting your life, please seek help at "
    #     "**1-800-522-4700** (National Problem Gambling Helpline).",
    #     icon="⚠️",
    # )


# ── Footer ───────────────────────────────────────────────────────────────────
add_betting_oracle_footer()

