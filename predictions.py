"""
Entry point for the Betting Cleanup MLB dashboard.

  - st.set_page_config()  called exactly once here
  - home_page()           landing page with per-game betting recommendations
  - st.navigation()       6-page sidebar navigation
"""

import sys
import datetime
import math
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from page_utils import (
    ROOT,
    MLB_BLUE,
    MLB_RED,
    _fetch_todays_schedule,
    _fetch_team_standings,
    _fetch_espn_odds,
    _load_precomputed,
    _load_model_results,
    _estimate_win_prob,
    _american_to_implied_prob,
    _prob_bar_html,
    init_session_state,
    add_betting_oracle_footer,
)

st.set_page_config(
    page_title="Betting Cleanup - MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f9fafb; color: #111827; }
    section[data-testid="stSidebar"] { background-color: #001f4d; }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
    h1, h2, h3 { color: #002D72; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(full_name: str) -> str:
    """Last word of a team name, e.g. 'New York Yankees' -> 'Yankees'."""
    return full_name.split()[-1] if full_name else ""


def _get_rs_g(team_full: str, hist_stnd: pd.DataFrame) -> float:
    """Team RS/G from most recent Retrosheet season. Defaults to 4.5."""
    try:
        last = team_full.split()[-1]
        sub  = hist_stnd[hist_stnd["team"].str.contains(last, case=False, na=False)]
        if not sub.empty:
            return float(sub.sort_values("season").iloc[-1]["RS_per_G"])
    except Exception:
        pass
    return 4.5


def _build_game_recs(
    g: dict,
    espn_game: dict | None,
    standings: dict,
    hist_stnd: pd.DataFrame,
) -> dict:
    """
    Build moneyline / run-line / over-under recommendations for one game.
    Returns dict with optional keys 'ml', 'rl', 'ou'.
    """
    home_full = g.get("home_name", "")
    away_full = g.get("away_name", "")
    home_prob = _estimate_win_prob(home_full, away_full, standings)
    away_prob = 1.0 - home_prob
    recs: dict = {}

    if not espn_game:
        return recs

    # -- Moneyline --
    ml_h_raw = espn_game.get("ml_home")
    ml_a_raw = espn_game.get("ml_away")
    if ml_h_raw and ml_a_raw:
        try:
            ml_h   = int(ml_h_raw)
            ml_a   = int(ml_a_raw)
            impl_h = _american_to_implied_prob(ml_h)
            impl_a = _american_to_implied_prob(ml_a)
            recs["ml"] = {
                "home": {
                    "team":     home_full,
                    "odds_str": f"+{ml_h}" if ml_h >= 0 else str(ml_h),
                    "impl":     impl_h,
                    "est_prob": home_prob,
                    "edge":     home_prob - impl_h,
                },
                "away": {
                    "team":     away_full,
                    "odds_str": f"+{ml_a}" if ml_a >= 0 else str(ml_a),
                    "impl":     impl_a,
                    "est_prob": away_prob,
                    "edge":     away_prob - impl_a,
                },
                "best": "home" if (home_prob - impl_h) >= (away_prob - impl_a) else "away",
            }
        except (TypeError, ValueError):
            pass

    # -- Run Line (+-1.5): P(home covers -1.5) ≈ home_prob^1.4 --
    spread_h_raw = espn_game.get("spread_home")
    spread_a_raw = espn_game.get("spread_away", "—")
    home_rl = home_prob ** 1.4
    away_rl = 1.0 - home_rl
    if spread_h_raw and str(spread_h_raw) not in ("—", "", "None"):
        try:
            sho    = int(str(spread_h_raw).replace("+", ""))
            impl_h = _american_to_implied_prob(sho)
            if spread_a_raw and str(spread_a_raw) not in ("—", "", "None"):
                sao    = int(str(spread_a_raw).replace("+", ""))
                impl_a = _american_to_implied_prob(sao)
                away_odds_str = f"+{sao}" if sao >= 0 else str(sao)
            else:
                impl_a = 1.0 - impl_h
                away_odds_str = "—"
            recs["rl"] = {
                "home": {
                    "pick":     f"{_short(home_full)} −1.5",
                    "odds_str": f"+{sho}" if sho >= 0 else str(sho),
                    "impl":     impl_h,
                    "est_prob": home_rl,
                    "edge":     home_rl - impl_h,
                },
                "away": {
                    "pick":     f"{_short(away_full)} +1.5",
                    "odds_str": away_odds_str,
                    "impl":     impl_a,
                    "est_prob": away_rl,
                    "edge":     away_rl - impl_a,
                },
                "best": "home" if (home_rl - impl_h) >= (away_rl - impl_a) else "away",
            }
        except (TypeError, ValueError):
            pass

    # -- Over / Under --
    ou_raw   = espn_game.get("over_under")
    ov_raw   = espn_game.get("over_odds")
    un_raw   = espn_game.get("under_odds")
    if ou_raw and ov_raw and un_raw:
        try:
            posted    = float(ou_raw)
            exp_total = _get_rs_g(home_full, hist_stnd) + _get_rs_g(away_full, hist_stnd)
            diff      = exp_total - posted
            over_prob = max(0.20, min(0.80, 0.50 + diff * 0.06))
            under_prob = 1.0 - over_prob

            def _parse(raw) -> int | None:
                try:
                    return int(str(raw).replace("+", ""))
                except (ValueError, TypeError):
                    return None

            def _fmt(raw, i) -> str:
                if i is None:
                    return "—"
                return f"+{i}" if i >= 0 else str(i)

            ov_int  = _parse(ov_raw)
            un_int  = _parse(un_raw)
            impl_ov = _american_to_implied_prob(ov_int)  if ov_int  else 0.5
            impl_un = _american_to_implied_prob(un_int) if un_int else 0.5

            recs["ou"] = {
                "posted":    posted,
                "exp_total": exp_total,
                "over": {
                    "pick":     f"Over  {posted}",
                    "odds_str": _fmt(ov_raw, ov_int),
                    "impl":     impl_ov,
                    "est_prob": over_prob,
                    "edge":     over_prob - impl_ov,
                },
                "under": {
                    "pick":     f"Under {posted}",
                    "odds_str": _fmt(un_raw, un_int),
                    "impl":     impl_un,
                    "est_prob": under_prob,
                    "edge":     under_prob - impl_un,
                },
                "best": "over" if (over_prob - impl_ov) >= (under_prob - impl_un) else "under",
            }
        except (TypeError, ValueError):
            pass

    return recs


def _rec_card_html(label: str, side: dict, exp_info: str) -> str:
    """Render one market recommendation as an HTML block."""
    edge_pct = side["edge"] * 100
    if edge_pct > 3:
        color, badge = "#16a34a", "✅ BET"
    elif edge_pct > 0:
        color, badge = "#d97706", "➡ LEAN"
    else:
        color, badge = "#dc2626", "⛔ PASS"

    if side.get("team"):
        pick_text = _short(side["team"])   # e.g. "New York Yankees" → "Yankees"
    else:
        pick_text = side.get("pick", "—")  # e.g. "Nationals +1.5" or "Over 8.5" — keep as-is
    return (
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:4px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<b style="font-size:0.88rem">{pick_text}</b>'
        f'<span style="background:{color};color:white;border-radius:6px;padding:1px 8px;'
        f'font-size:0.7rem;font-weight:700">{badge}</span></div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:2px">'
        f'Odds: <b>{side["odds_str"]}</b>'
        f' &nbsp;|&nbsp; Edge: <b style="color:{color}">{edge_pct:+.1f}%</b></div>'
        f'<div style="font-size:0.73rem;color:#888">{exp_info}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def home_page() -> None:
    """Landing page: hero metrics + per-game ML/Spread/O-U recommendations."""

    # Header
    hdr_left, hdr_right = st.columns([1, 5])
    with hdr_left:
        _logo = ROOT / "data_files" / "logo.png"
        if _logo.exists():
            st.image(str(_logo), width=110)
    with hdr_right:
        st.markdown(
            f"<h1 style='margin-bottom:0;color:#002D72'>⚾ Betting Cleanup</h1>"
            f"<p style='color:#6b7280;margin-top:2px'>MLB Predictions &nbsp;·&nbsp; "
            f"{datetime.date.today().strftime('%A, %B %d, %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Cached data
    games_today   = _fetch_todays_schedule()
    standings     = _fetch_team_standings()
    espn_odds     = _fetch_espn_odds()
    model_results = _load_model_results()
    _pre          = _load_precomputed()
    hist_stnd     = _pre["standings"]
    init_session_state()

    # Hero metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    total_games  = len(games_today)
    games_w_odds = sum(
        1 for g in games_today
        if any(g.get("home_name", "").split()[-1].lower() in eo.get("home_team", "").lower()
               for eo in espn_odds)
    )
    accuracy = model_results["moneyline"]["metrics"].get("accuracy") if model_results else None
    roc_auc  = model_results["moneyline"]["metrics"].get("roc_auc")  if model_results else None
    m1.metric("Today's Games",   total_games)
    m2.metric("Games with Odds", games_w_odds)
    m3.metric("ML Model AUC",    f"{roc_auc:.4f}" if roc_auc  else "—",
              help="Moneyline XGBoost ROC-AUC on held-out test set.")
    m4.metric("Model Accuracy",  f"{accuracy:.1%}" if accuracy else "—")
    m5.metric("Odds Source",     "ESPN" if espn_odds else "Unavailable")

    st.markdown("---")

    if not games_today:
        st.info("No MLB games scheduled today, or the MLB Stats API is unreachable.")
    else:
        st.markdown(f"### 🎯 Today's Games & Betting Recommendations")
        st.caption(
            "Win probability: current-season W% logistic model (+4% HFA). "
            "Run line: empirical cover-rate model. "
            "O/U: historical RS/G vs posted total. "
            "✅ BET = edge > 3% &nbsp;·&nbsp; ➡ LEAN = 0–3% &nbsp;·&nbsp; ⛔ PASS = negative edge."
        )

        _status_labels = {
            "Final": "🏁 Final", "Game Over": "🏁 Final",
            "In Progress": "🔴 LIVE", "Scheduled": "🕐 Scheduled",
            "Pre-Game": "⏳ Pre-Game", "Warmup": "⏳ Warmup",
            "Delayed": "⚠️ Delayed", "Postponed": "🚫 Postponed",
        }

        for idx, g in enumerate(games_today):
            away_full = g.get("away_name", "Away")
            home_full = g.get("home_name", "Home")
            away_sp   = g.get("away_probable_pitcher", "TBD") or "TBD"
            home_sp   = g.get("home_probable_pitcher", "TBD") or "TBD"
            status    = g.get("status", "Scheduled")
            venue     = g.get("venue_name", "—")

            gtime_raw = g.get("game_datetime", "")
            if gtime_raw:
                try:
                    dt_utc    = datetime.datetime.fromisoformat(gtime_raw.replace("Z", "+00:00"))
                    gtime_str = (dt_utc - datetime.timedelta(hours=4)).strftime("%I:%M %p ET")
                except Exception:
                    gtime_str = "TBD"
            else:
                gtime_str = "TBD"

            score_str = ""
            if str(status).lower() in ("final", "game over", "in progress", "live"):
                if g.get("away_score") is not None and g.get("home_score") is not None:
                    score_str = f" &nbsp;·&nbsp; **{g['away_score']}–{g['home_score']}**"

            hk = home_full.split()[-1].lower()
            espn_game = next((eo for eo in espn_odds if hk in eo.get("home_team", "").lower()), None)
            recs      = _build_game_recs(g, espn_game, standings, hist_stnd)
            home_prob = _estimate_win_prob(home_full, away_full, standings)

            with st.container(border=True):
                # ── Game header ──
                hdr_c1, hdr_c2 = st.columns([3, 2])
                with hdr_c1:
                    st.markdown(
                        f"#### {away_full} @ {home_full}"
                        + (score_str if score_str else ""),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<small>🏟️ {venue} &nbsp;·&nbsp; "
                        f"{_status_labels.get(status, status)} &nbsp;·&nbsp; "
                        f"🕐 {gtime_str}</small><br>"
                        f"<small>SP: <b>{away_sp}</b> (away) &nbsp;/&nbsp; <b>{home_sp}</b> (home)</small>",
                        unsafe_allow_html=True,
                    )
                with hdr_c2:
                    st.markdown(_prob_bar_html(home_prob, home_full, away_full), unsafe_allow_html=True)

                if not recs:
                    st.caption("⏳ Odds not yet available for this game.")
                    continue

                st.divider()

                # ── Three bet markets ──
                col_ml, col_rl, col_ou = st.columns(3)

                with col_ml:
                    st.markdown("##### 💵 Moneyline")
                    if "ml" in recs:
                        ml   = recs["ml"]
                        best = ml["best"]
                        side = ml[best]
                        other = ml["away" if best == "home" else "home"]
                        exp   = f"Est: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("ML", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {_short(other['team'])} {other['odds_str']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

                with col_rl:
                    st.markdown("##### 📏 Run Line (±1.5)")
                    if "rl" in recs:
                        rl   = recs["rl"]
                        best = rl["best"]
                        side = rl[best]
                        other = rl["away" if best == "home" else "home"]
                        exp   = f"Est cover: {side['est_prob']:.0%} · Impl: {side['impl']:.0%}"
                        st.markdown(_rec_card_html("RL", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {other['pick']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

                with col_ou:
                    st.markdown("##### 📊 Over/Under")
                    if "ou" in recs:
                        ou   = recs["ou"]
                        best = ou["best"]
                        side = ou[best]
                        other = ou["under" if best == "over" else "over"]
                        exp   = (
                            f"Exp total: {ou['exp_total']:.1f} · "
                            f"Posted: {ou['posted']} · "
                            f"Impl: {side['impl']:.0%}"
                        )
                        st.markdown(_rec_card_html("OU", side, exp), unsafe_allow_html=True)
                        st.caption(
                            f"Other side: {other['pick'].strip()} {other['odds_str']} "
                            f"(edge {other['edge']*100:+.1f}%)"
                        )
                    else:
                        st.caption("— odds unavailable —")

    st.markdown("---")

    # Navigation tiles
    st.markdown("### Explore")
    tc = st.columns(3)
    tiles = [
        ("📅", "Today",            "Full schedule with detailed game drill-down", "pages/1_Today.py"),
        ("📊", "Stats",            "Standings · Batting · Pitching · Leaders",   "pages/2_Stats.py"),
        ("🆚", "Matchup Analysis", "H2H history · Rolling win-rate charts",       "pages/3_Matchup_Analysis.py"),
    ]
    for col, (icon, title, desc, path) in zip(tc, tiles):
        with col:
            with st.container(border=True):
                st.markdown(f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>',
                            unsafe_allow_html=True)
                st.page_link(path, label=f"**{title}**")
                st.caption(desc)

    tc2 = st.columns(3)
    tiles2 = [
        ("🤖", "Models",      "XGBoost features · Evaluation · Savant research",  "pages/4_Models.py"),
        ("📈", "Performance", "Pick history · Model P&L · Kelly bankroll",         "pages/5_Performance.py"),
        ("ℹ️", "About",       "Methodology, data sources & tech stack",            "pages/6_Info.py"),
    ]
    for col, (icon, title, desc, path) in zip(tc2, tiles2):
        with col:
            with st.container(border=True):
                st.markdown(f'<div style="text-align:center;font-size:1.8rem;padding-top:4px">{icon}</div>',
                            unsafe_allow_html=True)
                st.page_link(path, label=f"**{title}**")
                st.caption(desc)

    add_betting_oracle_footer()


# ---------------------------------------------------------------------------
# Navigation (7 pages: Home + 6)
# ---------------------------------------------------------------------------
pg = st.navigation(
    {
        "": [
            st.Page(home_page, title="Home", icon="🏠", default=True),
            st.Page("pages/1_Today.py",            title="Today",            icon="📅"),
            st.Page("pages/2_Stats.py",            title="Stats",            icon="📊"),
            st.Page("pages/3_Matchup_Analysis.py", title="Matchup Analysis", icon="🆚"),
            st.Page("pages/4_Models.py",           title="Models",           icon="🤖"),
            st.Page("pages/5_Performance.py",      title="Performance",      icon="📈"),
            st.Page("pages/6_Info.py",             title="About",            icon="ℹ️"),
        ],
    }
)
pg.run()
