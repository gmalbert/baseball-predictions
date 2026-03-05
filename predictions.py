import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from retrosheet import (
    head_to_head,
    load_gameinfo,
    rolling_team_form,
    season_batting_leaders,
    season_pitching_leaders,
    season_standings,
    season_team_batting,
    season_team_pitching,
    team_list,
)

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

st.set_page_config(
    page_title="Baseball Predictions",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ Baseball Predictions")

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

with st.spinner("Loading team list…"):
    teams = team_list(min_year, max_year)

(
    tab_standings,
    tab_tbat,
    tab_tpitch,
    tab_bleaders,
    tab_pleaders,
    tab_h2h,
    tab_form,
    tab_features,
) = st.tabs([
    "📊 Standings",
    "🏏 Team Batting",
    "⚾ Team Pitching",
    "🔝 Batting Leaders",
    "🔝 Pitching Leaders",
    "🆚 Head-to-Head",
    "📈 Rolling Form",
    "🧮 Betting Features",
])


# ══════════════════════════════════════════════
# TAB 1 — Season Standings
# ══════════════════════════════════════════════
with tab_standings:
    st.subheader("Season Standings")
    with st.spinner("Loading standings…"):
        standings = season_standings(min_year, max_year)

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
    team_filter = st.multiselect("Select teams", teams, default=top_teams[:6], key="trend_teams")
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
    with st.spinner("Loading team batting…"):
        tbat = season_team_batting(min_year, max_year)

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
    with st.spinner("Loading team pitching…"):
        tpitch = season_team_pitching(min_year, max_year)

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
    with st.spinner("Loading batting leaders…"):
        bleaders = season_batting_leaders(min_year, max_year)

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
    with st.spinner("Loading pitching leaders…"):
        pleaders = season_pitching_leaders(min_year, max_year)

    pl_year = st.selectbox("Season", sorted(pleaders["season"].unique(), reverse=True), key="pl_year")
    pl_metric = st.selectbox("Sort by", ["ERA", "WHIP", "K9", "BB9", "K_BB", "SO", "IP"], key="pl_metric")
    pl_top = st.slider("Show top N", 10, 50, 25, key="pl_top")
    asc_p = pl_metric in ("ERA", "WHIP", "BB9")

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

    with st.spinner("Loading standings for feature builder…"):
        all_standings = season_standings(min_year, max_year)

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
        st.dataframe(display_feat, width='stretch', hide_index=True)

        num_feats = [
            "home_WPct", "vis_WPct", "WPct_diff", "PythWPct_diff",
            "home_RS_G", "home_RA_G", "vis_RS_G", "vis_RA_G",
            "RS_advantage", "RA_advantage", "home_win", "total_runs",
        ]
        corr = feat_df[num_feats].corr()
        fig = px.imshow(
            corr,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        st.plotly_chart(fig, width='stretch')

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



