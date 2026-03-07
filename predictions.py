import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


PROCESSED = ROOT / "data_files" / "processed"


@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict:
    """Load all pre-computed aggregated datasets once at startup.

    Returns a dict of DataFrames covering the full historical range.
    Year filtering happens in-memory after the sidebar slider fires — instant.
    """
    gi = pd.read_parquet(ROOT / "data_files" / "retrosheet" / "gameinfo.parquet")
    return {
        "gameinfo":        gi,
        "standings":       pd.read_parquet(PROCESSED / "standings.parquet"),
        "team_batting":    pd.read_parquet(PROCESSED / "team_batting.parquet"),
        "team_pitching":   pd.read_parquet(PROCESSED / "team_pitching.parquet"),
        "batting_leaders": pd.read_parquet(PROCESSED / "batting_leaders.parquet"),
        "pitching_leaders":pd.read_parquet(PROCESSED / "pitching_leaders.parquet"),
        "model_features":  pd.read_parquet(PROCESSED / "model_features.parquet"),
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
    page_title="Baseball Predictions",
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

st.title("⚾ Baseball Predictions")

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

(
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
) = st.tabs([
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
])


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
            _prob_dist_chart("moneyline", "pred_prob", "P(home win)")
        with dist_tab_sp:
            _prob_dist_chart("spread", "pred_prob", "P(home covers −1.5)")
        with dist_tab_ou:
            _prob_dist_chart("totals", "pred_prob_over", "P(went over)")
    
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
            _calibration_chart("moneyline", "pred_prob", "home_win", "P(home win)")
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
    
        if bt_model == "moneyline":
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "hruns": "H Runs", "vruns": "V Runs",
                "home_win": "Actually Won?",
                "pred_prob": "P(home win)", "pred_win": "Model Pick", "correct": "Correct?",
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
        st.markdown("### Backtest Leaderboard")
        st.dataframe(lb_df.rename(columns={
            "model": "Model",
            "total_bets": "Bets",
            "win_rate": "Win Rate",
            "roi": "ROI",
            "total_units": "Units",
        }), hide_index=True)

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

# ── Footer ───────────────────────────────────────────────────────────────────
add_betting_oracle_footer()

