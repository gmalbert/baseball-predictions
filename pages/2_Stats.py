"""Page: Stats — Standings · Team Batting · Team Pitching · Batting Leaders · Pitching Leaders"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from page_utils import (
    READABLE_COLS,
    _load_precomputed,
    render_sidebar,
    add_betting_oracle_footer,
)

min_year, max_year = render_sidebar()

_pre     = _load_precomputed()
standings = _pre["standings"][_pre["standings"]["season"].between(min_year, max_year)].copy()
tbat      = _pre["team_batting"][_pre["team_batting"]["season"].between(min_year, max_year)].copy()
tpitch    = _pre["team_pitching"][_pre["team_pitching"]["season"].between(min_year, max_year)].copy()
bleaders  = _pre["batting_leaders"][_pre["batting_leaders"]["season"].between(min_year, max_year)].copy()
pleaders  = _pre["pitching_leaders"][_pre["pitching_leaders"]["season"].between(min_year, max_year)].copy()

tab_stnd, tab_tbat, tab_tpitch, tab_bleaders, tab_pleaders = st.tabs([
    "Standings", "Team Batting", "Team Pitching", "Batting Leaders", "Pitching Leaders",
])

# ── Standings ─────────────────────────────────────────────────────────────────
with tab_stnd:
    st.subheader("Season Standings")
    year_sel = st.selectbox(
        "Select season", sorted(standings["season"].unique(), reverse=True), key="standings_year"
    )
    df_yr = standings[standings["season"] == year_sel].copy()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Teams", len(df_yr))
    if not df_yr.empty:
        best  = df_yr.loc[df_yr["WPct"].idxmax()]
        worst = df_yr.loc[df_yr["WPct"].idxmin()]
        col_b.metric("Best record",  f"{best['team']} ({best['W']}-{best['L']})")
        col_c.metric("Worst record", f"{worst['team']} ({worst['W']}-{worst['L']})")

    st.dataframe(
        df_yr[["team", "G", "W", "L", "WPct", "PythWPct", "RS", "RA", "RD", "RD_per_G", "RS_per_G", "RA_per_G"]]
        .sort_values("WPct", ascending=False).reset_index(drop=True).rename(columns=READABLE_COLS),
        width="stretch", hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_yr.sort_values("WPct", ascending=True),
            x="WPct", y="team", orientation="h",
            title=f"Win % — {year_sel}",
            color="WPct", color_continuous_scale="RdYlGn",
            labels={"WPct": "Win %", "team": ""},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.scatter(
            df_yr, x="RS_per_G", y="RA_per_G", text="team",
            title=f"Runs Scored vs Allowed / G — {year_sel}",
            color="WPct", color_continuous_scale="RdYlGn",
            labels={"RS_per_G": "RS / G", "RA_per_G": "RA / G"},
        )
        fig2.add_hline(y=df_yr["RA_per_G"].mean(), line_dash="dot", line_color="gray")
        fig2.add_vline(x=df_yr["RS_per_G"].mean(), line_dash="dot", line_color="gray")
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Win % Trend — Select Teams")
    top_teams    = standings.groupby("team")["WPct"].mean().nlargest(10).index.tolist()
    standings_teams = sorted(standings["team"].unique())
    team_filter  = st.multiselect(
        "Select teams", standings_teams, default=top_teams[:6], key="trend_teams"
    )
    if team_filter:
        trend_df = standings[standings["team"].isin(team_filter)]
        fig3 = px.line(
            trend_df, x="season", y="WPct", color="team",
            title="Season Win % over time",
            labels={"WPct": "Win %", "season": "Season"},
        )
        st.plotly_chart(fig3, width="stretch")

# ── Team Batting ──────────────────────────────────────────────────────────────
with tab_tbat:
    st.subheader("Team Batting")
    bat_year   = st.selectbox("Season", sorted(tbat["season"].unique(), reverse=True), key="tbat_year")
    bat_metric = st.selectbox("Sort by", ["BA", "SLG", "HR", "R", "SB", "K"], key="tbat_metric")
    df_bat = tbat[tbat["season"] == bat_year].sort_values(bat_metric, ascending=False).reset_index(drop=True)

    st.dataframe(
        df_bat[["team", "G", "PA", "AB", "R", "H", "doubles", "triples", "HR", "RBI", "BB", "K", "SB", "BA", "SLG"]]
        .rename(columns={"doubles": "2B", "triples": "3B"}).rename(columns=READABLE_COLS),
        width="stretch", hide_index=True,
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
        st.plotly_chart(fig, width="stretch")
    with c2:
        avg_metric = tbat.groupby("season")[bat_metric].mean().reset_index()
        fig2 = px.line(avg_metric, x="season", y=bat_metric, title=f"League-Avg {bat_metric} over time")
        st.plotly_chart(fig2, width="stretch")

    fig3 = px.scatter(
        df_bat, x="BA", y="HR", text="team", color="R",
        color_continuous_scale="Viridis",
        title=f"BA vs HR — {bat_year}",
        labels={"BA": "Batting Average", "HR": "Home Runs", "R": "Runs"},
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, width="stretch")

# ── Team Pitching ─────────────────────────────────────────────────────────────
with tab_tpitch:
    st.subheader("Team Pitching")
    pitch_year   = st.selectbox("Season", sorted(tpitch["season"].unique(), reverse=True), key="tpitch_year")
    pitch_metric = st.selectbox(
        "Sort by (lower = better for ERA/WHIP)",
        ["ERA", "WHIP", "K9", "BB9", "HR9"],
        key="pitch_metric",
    )
    asc    = pitch_metric in ("ERA", "WHIP", "BB9", "HR9")
    df_pt  = tpitch[tpitch["season"] == pitch_year].sort_values(pitch_metric, ascending=asc).reset_index(drop=True)

    st.dataframe(
        df_pt[["team", "G", "IP", "HA", "HRA", "RA", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "HR9"]]
        .rename(columns=READABLE_COLS),
        width="stretch", hide_index=True,
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
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.scatter(
            df_pt, x="ERA", y="WHIP", text="team", color="K9",
            color_continuous_scale="Viridis",
            title=f"ERA vs WHIP — {pitch_year}",
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, width="stretch")

    league_avg = tpitch.groupby("season")[["ERA", "WHIP", "K9"]].mean().reset_index()
    fig3 = px.line(
        league_avg.melt(id_vars="season", value_vars=["ERA", "WHIP", "K9"]),
        x="season", y="value", color="variable", facet_col="variable", facet_col_wrap=3,
        title="League-Average ERA / WHIP / K9 over time",
    )
    fig3.update_yaxes(matches=None)
    fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig3, width="stretch")

# ── Batting Leaders ───────────────────────────────────────────────────────────
with tab_bleaders:
    st.subheader("Individual Batting Leaders")
    bl_year   = st.selectbox("Season", sorted(bleaders["season"].unique(), reverse=True), key="bl_year")
    bl_metric = st.selectbox("Sort by", ["BA", "SLG", "HR", "RBI", "SB", "BB", "K"], key="bl_metric")
    bl_top    = st.slider("Show top N", 10, 50, 25, key="bl_top")

    df_bl = (
        bleaders[bleaders["season"] == bl_year]
        .sort_values(bl_metric, ascending=(bl_metric == "K"))
        .head(bl_top).reset_index(drop=True)
    )

    st.dataframe(
        df_bl[["full_name", "team", "PA", "AB", "H", "doubles", "triples", "HR", "RBI", "BB", "K", "SB", "BA", "SLG"]]
        .rename(columns={"doubles": "2B", "triples": "3B"}).rename(columns=READABLE_COLS),
        width="stretch", hide_index=True,
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
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.scatter(
            bleaders[bleaders["season"] == bl_year],
            x="BA", y="SLG", size="PA",
            hover_name="full_name",
            color="HR", color_continuous_scale="Reds",
            title=f"BA vs SLG (size=PA) — {bl_year}",
        )
        st.plotly_chart(fig2, width="stretch")

# ── Pitching Leaders ──────────────────────────────────────────────────────────
with tab_pleaders:
    st.subheader("Individual Pitching Leaders")
    pl_year    = st.selectbox("Season", sorted(pleaders["season"].unique(), reverse=True), key="pl_year")
    pl_options = ["ERA", "WHIP", "K9", "BB9", "K_BB", "SO", "IP"]
    pl_metric  = st.selectbox(
        "Sort by", pl_options,
        format_func=lambda x: READABLE_COLS.get(x, x),
        key="pl_metric",
    )
    pl_top = st.slider("Show top N", 10, 50, 25, key="pl_top")
    asc_p  = pl_metric in ("ERA", "WHIP", "BB9")

    with st.expander("Stats data dictionary"):
        dict_df = pd.DataFrame({
            "column": ["GS", "IP", "H", "HR", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "K_BB"],
            "description": [
                "Games started", "Innings pitched", "Hits allowed",
                "Home runs allowed", "Earned runs allowed", "Walks allowed",
                "Strikeouts", "Earned run average", "Walks+Hits per inning pitched",
                "Strikeouts per 9 innings", "Walks per 9 innings", "Strikeout-to-walk ratio",
            ],
        })
        st.dataframe(dict_df.rename(columns=READABLE_COLS), width="stretch", hide_index=True)

    df_pl = (
        pleaders[pleaders["season"] == pl_year]
        .sort_values(pl_metric, ascending=asc_p)
        .head(pl_top).reset_index(drop=True)
    )

    st.dataframe(
        df_pl[["full_name", "team", "GS", "IP", "H", "HR", "ER", "BB", "SO", "ERA", "WHIP", "K9", "BB9", "K_BB"]]
        .rename(columns={"full_name": "Pitcher"}).rename(columns=READABLE_COLS),
        width="stretch", hide_index=True,
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
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.scatter(
            pleaders[pleaders["season"] == pl_year],
            x="ERA", y="WHIP", size="IP",
            hover_name="full_name",
            color="K9", color_continuous_scale="Viridis",
            title=f"ERA vs WHIP (size=IP) — {pl_year}",
        )
        st.plotly_chart(fig2, width="stretch")

add_betting_oracle_footer()
