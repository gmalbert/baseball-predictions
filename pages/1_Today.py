"""Page: Today — today's schedule, pitcher matchups & odds detail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from page_utils import (
    ROOT,
    READABLE_COLS,
    _MLB_TO_RETRO,
    _fetch_todays_schedule,
    _fetch_team_standings,
    _fetch_pitcher_stats,
    _fetch_espn_odds,
    _load_latest_odds,
    _estimate_win_prob,
    _prob_bar_html,
    render_sidebar,
    init_session_state,
    add_betting_oracle_footer,
)
from retrosheet import head_to_head, rolling_team_form

init_session_state()
render_sidebar(show_year_filter=False)

_games_today = _fetch_todays_schedule()

# ── Game Detail View ──────────────────────────────────────────────────────────
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

    status = g.get("status", "Scheduled")
    venue  = g.get("venue_name", "—")
    series = g.get("series_description", "")
    gtime_raw = g.get("game_datetime", "")
    if gtime_raw:
        try:
            dt_utc = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
            dt_et  = dt_utc - datetime.timedelta(hours=4)
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

    away_sp = g.get("away_probable_pitcher", "TBD") or "TBD"
    home_sp = g.get("home_probable_pitcher", "TBD") or "TBD"
    st.markdown("### ⚾ Probable Pitchers")
    with st.spinner("Fetching pitcher stats…"):
        away_sp_stats = _fetch_pitcher_stats(away_sp)
        home_sp_stats = _fetch_pitcher_stats(home_sp)
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**{away_full} (Away)** · {away_sp}")
        if away_sp_stats:
            st.dataframe(
                pd.DataFrame(away_sp_stats.items(), columns=["Stat", "Value"]),
                hide_index=True, width="stretch",
            )
        elif away_sp != "TBD":
            st.caption("Stats not yet available for this season.")
    with pc2:
        st.markdown(f"**{home_full} (Home)** · {home_sp}")
        if home_sp_stats:
            st.dataframe(
                pd.DataFrame(home_sp_stats.items(), columns=["Stat", "Value"]),
                hide_index=True, width="stretch",
            )
        elif home_sp != "TBD":
            st.caption("Stats not yet available for this season.")

    st.divider()

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
                last5 = form.tail(5)[["date", "RS", "RA", "W"]].copy()
                last5["date"] = last5["date"].dt.strftime("%b %d")
                last5["Result"] = last5["W"].map({1: "✔ W", 0: "✘ L"})
                st.dataframe(
                    last5[["date", "RS", "RA", "Result"]].rename(
                        columns={"date": "Date", "RS": "R", "RA": "RA"}
                    ),
                    hide_index=True,
                )

    st.divider()

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
        st.caption(
            f"Source: **{_game_espn['provider']}** (ESPN public API — free, no quota) "
            "· refreshes every 30 min"
        )
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

    _odds_csv = _load_latest_odds()
    if not _odds_csv.empty:
        with st.expander("📚 Multi-book comparison (saved Odds API data)"):
            st.caption(
                "This data was saved from a manual run of `fetch_current_odds()`. "
                "The dashboard never calls The Odds API automatically."
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

# ── Schedule List View ────────────────────────────────────────────────────────
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

        _status_badge = {
            "Final": "🏁", "Game Over": "🏁",
            "In Progress": "🔴 LIVE", "Scheduled": "🕐",
            "Pre-Game": "⏳", "Warmup": "⏳",
            "Delayed": "⚠️", "Suspended": "⚠️",
            "Postponed": "🚫", "Cancelled": "🚫",
        }

        for idx, g in enumerate(_games_today):
            away_name   = g.get("away_name", "Away")
            home_name   = g.get("home_name", "Home")
            away_sp     = g.get("away_probable_pitcher", "TBD") or "TBD"
            home_sp     = g.get("home_probable_pitcher", "TBD") or "TBD"
            venue       = g.get("venue_name", "—")
            status      = g.get("status", "Scheduled")
            status_icon = _status_badge.get(status, "")
            gtime_raw   = g.get("game_datetime", "")

            if gtime_raw:
                try:
                    dt_utc    = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
                    dt_et     = dt_utc - datetime.timedelta(hours=4)
                    gtime_str = dt_et.strftime("%I:%M %p ET")
                except Exception:
                    gtime_str = "TBD"
            else:
                gtime_str = "TBD"

            score_str = ""
            if str(status).lower() in ("final", "game over", "in progress", "live", "completed"):
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
                        use_container_width=True,
                    ):
                        st.session_state["schedule_selected_game"] = g
                        st.rerun()

add_betting_oracle_footer()
