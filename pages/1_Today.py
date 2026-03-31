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
    _load_game_context_cache,
    _fetch_pitcher_throw_hand,
    _fetch_team_il_players,
    _fetch_team_rest_days,
    render_sidebar,
    init_session_state,
    add_betting_oracle_footer,
)
from retrosheet import head_to_head, rolling_team_form, load_gameinfo
from src.ingestion.weather import fetch_forecast

init_session_state()
render_sidebar(show_year_filter=False)

_games_today = _fetch_todays_schedule()

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

    # ── 🔍 Game Context Factors ───────────────────────────────────────────────
    st.markdown("### 🔍 Game Context Factors")
    _ctx = _load_game_context_cache()

    # Venue / park row (full width)
    _pf      = _ctx["park_factors"].get(home_retro)
    _ump_avg = _ctx["ump_park_avg"].get(home_retro)
    _dn_data = _ctx["daynight"]

    _pf_label  = f"{_pf:.3f}" if _pf else "N/A"
    _pf_delta  = f"{'↑ hitter-friendly' if _pf and _pf > 1.05 else ('↓ pitcher-friendly' if _pf and _pf < 0.95 else '≈ neutral')}" if _pf else ""
    _ump_label = f"{_ump_avg:.1f} R/G" if _ump_avg else "N/A"

    _ctx_v1, _ctx_v2, _ctx_v3 = st.columns(3)
    _ctx_v1.metric(
        "🏟️ Park Factor",
        _pf_label,
        delta=_pf_delta if _pf_delta else None,
        delta_color="normal",
        help="Avg total runs/game at this park vs league avg (>1.0 = hitter-friendly). Last 3 seasons.",
    )
    _ctx_v2.metric(
        "⚾ Ump Park Avg R/G",
        _ump_label,
        help="Historical avg total runs/game in games played at this park. Reflects umpire/park run environment.",
    )

    # Retrieve today's SP handedness
    _home_sp_hand = _fetch_pitcher_throw_hand(home_sp)
    _away_sp_hand = _fetch_pitcher_throw_hand(away_sp)
    _home_plat    = _ctx["platoon"].get(home_retro, {})
    _away_plat    = _ctx["platoon"].get(away_retro, {})

    # Home batters face away SP; away batters face home SP
    def _plat_adv(bat_pct_left: float, sp_throws: str) -> str:
        if sp_throws == "L":
            adv = 1 - bat_pct_left   # right-handed batters advantage vs LHP
            return f"{adv:.0%} RHB vs LHP"
        elif sp_throws == "R":
            return f"{bat_pct_left:.0%} LHB vs RHP"
        return "?"

    _home_bat_adv = _plat_adv(_home_plat.get("pct_left", 0.5), _away_sp_hand)
    _away_bat_adv = _plat_adv(_away_plat.get("pct_left", 0.5), _home_sp_hand)

    with _ctx_v3:
        st.markdown("**⚔️ Platoon Matchup**")
        sp_lbl_a = f"{away_sp.split()[-1] if away_sp != 'TBD' else 'TBD'} ({_away_sp_hand}HP)"
        sp_lbl_h = f"{home_sp.split()[-1] if home_sp != 'TBD' else 'TBD'} ({_home_sp_hand}HP)"
        st.caption(f"Away SP: **{sp_lbl_a}** → Home batters: {_home_bat_adv}")
        st.caption(f"Home SP: **{sp_lbl_h}** → Away batters: {_away_bat_adv}")

    st.divider()

    # Team-specific rows: rest days, bullpen, day/night, injuries
    _gc_away, _gc_home = st.columns(2)
    for _col, _team_full, _team_retro in [
        (_gc_away, away_full, away_retro),
        (_gc_home, home_full, home_retro),
    ]:
        with _col:
            _side = "(Away)" if _team_full == away_full else "(Home)"
            st.markdown(f"**{_team_full} {_side}**")

            # Rest days
            _rest = _fetch_team_rest_days(_team_full)
            if _rest is None:
                _rest_label = "N/A"
                _rest_help  = "Could not determine — check schedule."
            elif _rest == 0:
                _rest_label = "Back-to-back"
                _rest_help  = "Played yesterday."
            else:
                _rest_label = f"{_rest}d rest"
                _rest_help  = f"{_rest} day(s) since last game."

            # Bullpen load
            _bp_ipg = _ctx["bullpen_ip_pg"].get(_team_retro)
            _bp_label = f"{_bp_ipg:.1f} IP/G" if _bp_ipg else "N/A"

            # Day/night tendency
            _dn_team = _dn_data.get(_team_retro, {})
            _day_w   = _dn_team.get("day")
            _night_w = _dn_team.get("night")
            _dn_label = (
                f"Day {_day_w:.0%} / Night {_night_w:.0%}"
                if _day_w is not None and _night_w is not None
                else "N/A"
            )

            r1c1, r1c2 = _col.columns(2)
            r1c1.metric("📅 Rest", _rest_label, help=_rest_help)
            r1c2.metric("💪 Bullpen IP/G", _bp_label,
                        help="Avg relief innings/game (last 2 seasons). Higher = bullpen-heavy team.")
            _col.metric("🌙 Day/Night W%", _dn_label,
                        help="Historical win % in day vs night games (last 3 seasons).")

            # IL players
            _il = _fetch_team_il_players(_team_full)
            if _il:
                with _col.expander(f"🏥 IL ({len(_il)} player{'s' if len(_il) != 1 else ''})"):
                    for _name in sorted(_il):
                        st.markdown(f"- {_name}")
            else:
                _col.caption("🏥 IL: None reported")

    st.divider()

    _cur_year = datetime.date.today().year
    st.markdown("### 🆚 Head-to-Head History (2020–present)")
    with st.spinner("Loading H2H data…"):
        h2h_detail = head_to_head(away_retro, home_retro, 2020, _cur_year)

    if h2h_detail.empty:
        st.info(
            f"No historical matchups found between **{away_retro}** and **{home_retro}** "
            f"in the 2020–{_cur_year} dataset."
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
                form = rolling_team_form(team_retro, 10, 2020, _cur_year)
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

    # ── Weather / situational context ─────────────────────────────────────────
    st.markdown("### 🌤️ Weather & Situational Context")
    _venue_name = g.get("venue_name", "")
    _game_date_str = ""
    if g.get("game_datetime"):
        try:
            _game_date_str = (
                datetime.datetime.fromisoformat(g["game_datetime"].replace("Z", "+00:00"))
                .date()
                .isoformat()
            )
        except Exception:
            pass

    _wx = fetch_forecast(_venue_name, _game_date_str) if (_venue_name and _game_date_str) else None

    def _format_flag_value(val):
        if val is True:
            return "Yes ✅"
        if val is False:
            return "No ❌"
        return "N/A"

    if _wx is None:
        st.info("⚠️ Weather data unavailable — venue not recognised or network error.")
    elif _wx.get("is_dome"):
        st.info(
            f"🏟️ **{_venue_name}** has a retractable / fixed roof — "
            "weather has no significant impact on this game."
        )

        st.markdown("**Model Weather Flags**")
        _dome_flag_defs = [
            ("Wind Out", None, "Not applicable for domed parks"),
            ("Wind In",  None, "Not applicable for domed parks"),
            ("Dome Park", True, "Enclosed/retractable roof park — weather effects are muted"),
            ("Cold Temp", None, "Not applicable for domed parks"),
            ("Hot Temp", None, "Not applicable for domed parks"),
            ("Overcast", None, "Not applicable for domed parks"),
        ]
        _dfc = st.columns(3)
        for _di, (_dn, _dv, _dh) in enumerate(_dome_flag_defs):
            with _dfc[_di % 3]:
                st.metric(_dn, _format_flag_value(_dv), help=_dh)
    else:
        _temp_f   = _wx.get("temp_f", 0.0)
        _wind_mph = _wx.get("wind_mph", 0.0)
        _precip   = _wx.get("precip_mm", 0.0)
        _humid    = _wx.get("humidity_pct", 0.0)
        _cloud    = _wx.get("cloud_cover_pct", 0.0)
        _is_past  = _game_date_str < datetime.date.today().isoformat()
        _wx_api   = "archive" if _is_past else "forecast"

        _wx_cols = st.columns(4)
        with _wx_cols[0]:
            st.metric("Temperature", f"{_temp_f:.0f} °F")
        with _wx_cols[1]:
            st.metric("Wind", f"{_wind_mph:.0f} mph")
        with _wx_cols[2]:
            st.metric("Precip", f"{_precip:.1f} mm")
        with _wx_cols[3]:
            st.metric("Cloud Cover", f"{_cloud:.0f}%")
        st.caption(
            f"Humidity: {_humid:.0f}%  ·  "
            f"Source: Open-Meteo {_wx_api} at **{_venue_name}**  ·  "
            "Game-time hours averaged (1–9 PM local)"
        )

        # ── Model-derived weather flags ────────────────────────────────────
        st.markdown("**Model Weather Flags**")
        _temp_cold = _temp_f < 50
        _temp_hot  = _temp_f > 90
        _overcast  = _cloud > 75
        _flag_defs = [
            ("Wind Out", None, "Wind blowing toward outfield (park-specific). Not modeled directly."),
            ("Wind In",  None, "Wind blowing into the infield (park-specific). Not modeled directly."),
            ("Dome Park", False, "Outdoor park — dome flag is off"),
            ("Cold Temp", _temp_cold, "Temp < 50 °F — offense may be suppressed"),
            ("Hot Temp",  _temp_hot,  "Temp > 90 °F — offense may be elevated"),
            ("Overcast",  _overcast,  "Cloud cover > 75% — overcast conditions"),
        ]
        _fc = st.columns(3)
        for _i, (_fname, _fval, _fhelp) in enumerate(_flag_defs):
            with _fc[_i % 3]:
                st.metric(
                    label=_fname,
                    value=_format_flag_value(_fval),
                    help=_fhelp,
                )

    # ── Historical weather at this venue ──────────────────────────────────
    _wx_hist_path = ROOT / "data_files" / "processed" / "weather_historical.parquet"
    with st.expander("📅 Historical Weather at This Venue"):
        if not _wx_hist_path.exists():
            st.info(
                "Historical weather data has not been built yet. "
                "Run `python scripts/fetch_weather_history.py` once to populate it."
            )
        else:
            try:
                _wx_hist_df = pd.read_parquet(_wx_hist_path)
                # load_gameinfo() translates raw codes → full names (e.g. "NYA" → "Yankees"),
                # so hometeam values match home_retro which comes from _MLB_TO_RETRO.
                _gi = load_gameinfo()[["gid", "hometeam", "season"]]
                _venue_wx = (
                    _wx_hist_df
                    .merge(_gi, on="gid", how="left")
                )
                _venue_wx = _venue_wx[_venue_wx["hometeam"] == home_retro]

                if _venue_wx.empty:
                    st.caption("No historical weather rows found for this home team/venue.")
                else:
                    _wx_num_cols = [c for c in ["temp_f", "wind_mph", "precip_mm", "humidity_pct", "cloud_cover_pct"] if c in _venue_wx.columns]
                    _by_season = _venue_wx.groupby("season").agg(
                        Games=("gid", "count"),
                        **{c: (c, "mean") for c in _wx_num_cols},
                    ).reset_index().sort_values("season", ascending=False)
                    _rename = {
                        "season": "Season", "temp_f": "Avg Temp (°F)",
                        "wind_mph": "Avg Wind (mph)", "precip_mm": "Avg Precip (mm)",
                        "humidity_pct": "Avg Humidity (%)", "cloud_cover_pct": "Avg Cloud (%)",
                    }
                    st.dataframe(
                        _by_season.rename(columns=_rename)
                        .round({"Avg Temp (°F)": 1, "Avg Wind (mph)": 1, "Avg Precip (mm)": 2, "Avg Humidity (%)": 1, "Avg Cloud (%)": 1})
                        .reset_index(drop=True),
                        hide_index=True, width="stretch",
                    )
                    st.caption(f"Averages across game-time hours (1–9 PM local) · {len(_venue_wx):,} games · {home_retro} home park")
            except Exception as _e:
                st.warning(f"Could not load historical weather: {_e}")

    if not _odds_csv.empty:
        with st.expander("📚 Multi-book comparison (Odds API)"):
            import os as _os
            _has_key = bool(_os.environ.get("ODDS_API_KEY") or st.secrets.get("ODDS_API_KEY", ""))
            if _has_key:
                st.caption("Live odds fetched automatically from The Odds API (refreshed hourly).")
            else:
                st.caption(
                    "Showing saved odds data. Set `ODDS_API_KEY` in environment or `st.secrets` "
                    "to enable automatic live fetching."
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
