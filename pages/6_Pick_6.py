"""Page: Pick 6 — DraftKings Pick 6 MLB Player Props"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from page_utils import (
    render_sidebar,
    add_betting_oracle_footer,
)
from retrosheet import (
    load_batting,
    load_pitching,
    load_players,
    season_batting_leaders,
    season_pitching_leaders,
)

render_sidebar(show_year_filter=False)

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

# ─── Constants ────────────────────────────────────────────────────────────────

_CUR_YEAR = datetime.date.today().year
_SEASONS   = list(range(2020, _CUR_YEAR + 1))

# DraftKings Pick 6 batter prop categories
BATTER_PROPS = ["Hits", "Home Runs", "RBI", "Runs", "Total Bases", "Hits+Runs+RBI"]
# DraftKings Pick 6 pitcher prop categories
PITCHER_PROPS = ["Strikeouts"]

ALL_PROPS = BATTER_PROPS + PITCHER_PROPS

# Retrosheet batting column → DK prop name
_PROP_BAT_COL: dict[str, str] = {
    "Hits":         "b_h",
    "Home Runs":    "b_hr",
    "RBI":          "b_rbi",
    "Runs":         "b_r",
    "Total Bases":  "_tb",    # derived
    "Hits+Runs+RBI":"_hrr",   # derived
}
_PROP_PITCH_COL: dict[str, str] = {
    "Strikeouts": "p_k",
}

# Season avg column in leaders frame → prop
_LEADERS_BAT_COL: dict[str, str] = {
    "Hits":         ("H",   None),
    "Home Runs":    ("HR",  None),
    "RBI":          ("RBI", None),
    "Runs":         ("R",   None),
    "Total Bases":  (None,  None),   # not in leaders, computed below
    "Hits+Runs+RBI":(None,  None),   # composite
}

# Confidence tier thresholds (probability of exceeding line)
_TIER_THRESHOLDS = [(0.65, "🔥 ELITE"), (0.60, "💪 STRONG"), (0.55, "✅ GOOD"), (0.0, "⚠️ LEAN")]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tier(prob: float) -> str:
    for threshold, label in _TIER_THRESHOLDS:
        if prob >= threshold:
            return label
    return "⚠️ LEAN"


def _round_line(val: float) -> float:
    """Round to nearest 0.5 — standard DK line increment."""
    return round(val * 2) / 2


def _suggested_line(season_avg: float) -> float:
    """DraftKings typically sets lines ~85–90% of season avg, rounded to 0.5."""
    return max(0.5, _round_line(season_avg * 0.87))


def _df_height(df: pd.DataFrame, row_height: int = 35, header: int = 38, max_h: int = 600) -> int:
    return min(len(df) * row_height + header + 2, max_h)


@st.cache_data(show_spinner=False)
def _batter_game_logs(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    """Per-game batter rows with derived columns (TB, H+R+RBI)."""
    df = load_batting(min_year, max_year)
    df["_tb"]  = df["b_h"] + df["b_d"] + 2 * df["b_t"] + 3 * df["b_hr"]
    df["_hrr"] = df["b_h"] + df["b_r"] + df["b_rbi"]
    return df


@st.cache_data(show_spinner=False)
def _pitcher_game_logs(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return load_pitching(min_year, max_year)


@st.cache_data(show_spinner=False)
def _player_registry(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    pl = load_players(min_year, max_year)
    pl["full_name"] = pl["first"].str.strip() + " " + pl["last"].str.strip()
    return pl[["id", "full_name", "season"]].drop_duplicates()


@st.cache_data(show_spinner=False)
def _batting_leaders_cached(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return season_batting_leaders(min_year, max_year)


@st.cache_data(show_spinner=False)
def _pitching_leaders_cached(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return season_pitching_leaders(min_year, max_year, min_ip=20.0)


def _get_player_game_log(player_id: str, prop: str, season: int) -> pd.DataFrame:
    """Return a per-game DataFrame for the given player + prop column."""
    if prop in _PROP_BAT_COL:
        df = _batter_game_logs(season, season)
        stat_col = _PROP_BAT_COL[prop]
    else:
        df = _pitcher_game_logs(season, season)
        stat_col = _PROP_PITCH_COL[prop]

    player_df = df[df["id"] == player_id].copy()
    if player_df.empty:
        return pd.DataFrame()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(player_df["date"]):
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")

    player_df = player_df.sort_values("date")
    player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors="coerce")

    out = player_df[["date", "team", "opp", "vishome", stat_col]].rename(
        columns={"date": "Date", "opp": "Opponent", "vishome": "H/A", stat_col: "stat"}
    )
    out["H/A"] = out["H/A"].map({"h": "Home", "v": "Away"})
    return out.dropna(subset=["stat"]).reset_index(drop=True)


def _analyse_player(game_log: pd.DataFrame, dk_line: float):
    """Compute averages, hit rates, and MORE/LESS prediction via Normal distribution."""
    recent = game_log.tail(10)
    stat = recent["stat"]

    last_3_avg  = float(game_log.tail(3)["stat"].mean()) if len(game_log) >= 3 else float(stat.mean())
    last_5_avg  = float(game_log.tail(5)["stat"].mean()) if len(game_log) >= 5 else float(stat.mean())
    last_10_avg = float(stat.mean())
    season_avg  = float(game_log["stat"].mean())

    total_games = len(recent)
    games_over  = int((recent["stat"] > dk_line).sum())

    # Normal distribution probability estimate
    mu, sigma = float(stat.mean()), float(stat.std(ddof=1)) if len(stat) > 1 else 1.0
    sigma = max(sigma, 0.01)
    # Use weighted average: 70% last-10, 30% season
    mu_w = 0.7 * last_10_avg + 0.3 * season_avg

    from scipy.stats import norm  # lazy import — scipy available via scikit-learn deps
    prob_over = float(1 - norm.cdf(dk_line, loc=mu_w, scale=sigma))
    prob_over = max(0.05, min(0.95, prob_over))
    prob_under = 1.0 - prob_over

    if prob_over >= prob_under:
        recommendation, confidence = "MORE", prob_over
    else:
        recommendation, confidence = "LESS", prob_under

    return {
        "last_3_avg":  last_3_avg,
        "last_5_avg":  last_5_avg,
        "last_10_avg": last_10_avg,
        "season_avg":  season_avg,
        "total_games": total_games,
        "games_over":  games_over,
        "recommendation": recommendation,
        "confidence":     confidence,
        "tier":           _tier(confidence),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    st.title("🎯 DraftKings Pick 6 – MLB")
    st.markdown("Analyse player props vs. DraftKings Pick 6 lines using historical game logs.")
    st.markdown("---")

    tab_calc, tab_top, tab_leaders = st.tabs([
        "📊 DK Pick 6 Calculator",
        "⭐ Top Picks",
        "🏆 Season Leaders",
    ])

    # ─── preload shared data ──────────────────────────────────────────────────
    registry = _player_registry(2020, _CUR_YEAR)

    # =========================================================================
    # TAB 1 — DK Pick 6 Calculator
    # =========================================================================
    with tab_calc:
        st.subheader("📊 DraftKings Pick 6 – Line Comparison")
        st.markdown("Search for a player, enter the DK line, and get a MORE / LESS recommendation.")

        col_search, col_prop = st.columns([2, 1])

        with col_search:
            search_term = st.text_input(
                "🔍 Search Player",
                placeholder="Type player name (e.g., Aaron Judge, Shohei Ohtani)",
                key="calc_search",
            )

        selected_id   = None
        selected_name = None
        selected_season = _CUR_YEAR

        if search_term:
            lower = search_term.lower()
            matches = registry[registry["full_name"].str.lower().str.contains(lower, na=False)].copy()
            if not matches.empty:
                # Show most recent season first, deduplicate by id
                matches = (
                    matches.sort_values("season", ascending=False)
                    .drop_duplicates("id")
                    .head(30)
                )
                player_opts = {
                    f"{row['full_name']} ({int(row['season'])})": (row["id"], int(row["season"]))
                    for _, row in matches.iterrows()
                }
                with col_search:
                    sel = st.selectbox("Select Player", list(player_opts.keys()), key="calc_player_sel")
                selected_id, selected_season = player_opts[sel]
                selected_name = sel.split("(")[0].strip()
            else:
                with col_search:
                    st.info("No players found. Try a different name.")

        with col_prop:
            prop = st.selectbox("Prop Type", ALL_PROPS, key="calc_prop")

        if selected_id and prop:
            game_log = _get_player_game_log(selected_id, prop, selected_season)

            if game_log is None or game_log.empty:
                st.warning(f"No {prop} game log data found for this player in {selected_season}.")
            else:
                season_mean = float(game_log["stat"].mean())
                sug_line    = _suggested_line(season_mean)

                dk_line = st.number_input(
                    f"DraftKings Pick 6 Line for {prop}",
                    min_value=0.5,
                    max_value=50.0,
                    value=sug_line,
                    step=0.5,
                    key="calc_line",
                    help="Enter the exact over/under line from DraftKings Pick 6",
                )

                res = _analyse_player(game_log, dk_line)

                # ── Hero card ──────────────────────────────────────────────
                rec_color = "#16a34a" if res["recommendation"] == "MORE" else "#dc2626"
                st.markdown(f"""
<div style="background: linear-gradient(135deg, #002D72 0%, #D50032 100%);
            padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
  <h2 style="margin: 0; font-size: 1.8rem;">{selected_name}</h2>
  <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">{prop} · {selected_season}</p>
  <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem;">
    <div>
      <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">DraftKings Line</p>
      <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{dk_line:.1f}</p>
    </div>
    <div style="text-align: right;">
      <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Recommendation</p>
      <p style="margin: 0; font-size: 2.5rem; font-weight: bold; color: #ffd700;">{res['recommendation']}</p>
      <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{res['tier']}</p>
    </div>
  </div>
  <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
    <p style="margin: 0; font-size: 1rem;">
      Confidence: <strong>{res['confidence']:.1%}</strong> &nbsp;·&nbsp; Season avg: <strong>{res['season_avg']:.2f}</strong>
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
      Last {res['total_games']} games: {res['games_over']} over, {res['total_games'] - res['games_over']} under
    </p>
  </div>
</div>
                """, unsafe_allow_html=True)

                # ── Performance metrics ────────────────────────────────────
                st.markdown("### 📈 Performance Analysis")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(
                    "Last 3 Games Avg", f"{res['last_3_avg']:.2f}",
                    f"{res['last_3_avg'] - dk_line:+.2f} vs line",
                )
                mc2.metric(
                    "Last 5 Games Avg", f"{res['last_5_avg']:.2f}",
                    f"{res['last_5_avg'] - dk_line:+.2f} vs line",
                )
                mc3.metric(
                    "Last 10 Games Avg", f"{res['last_10_avg']:.2f}",
                    f"{res['last_10_avg'] - dk_line:+.2f} vs line",
                )
                mc4.metric(
                    "Season Avg", f"{res['season_avg']:.2f}",
                    f"{res['season_avg'] - dk_line:+.2f} vs line",
                )

                # ── Recent game log ────────────────────────────────────────
                st.markdown("### 📋 Recent Game Log (Last 10)")
                gl_disp = game_log.tail(10).copy()
                if "Date" in gl_disp.columns and pd.api.types.is_datetime64_any_dtype(gl_disp["Date"]):
                    gl_disp["Date"] = gl_disp["Date"].dt.date

                gl_disp["Result"] = gl_disp["stat"].apply(
                    lambda x: f"✅ OVER ({x:.1f})" if x > dk_line else f"❌ UNDER ({x:.1f})"
                )
                gl_disp[prop] = gl_disp["stat"].round(2)
                disp_cols = ["Date", "H/A", "Opponent", prop, "Result"]
                avail = [c for c in disp_cols if c in gl_disp.columns]
                st.dataframe(
                    gl_disp[avail].sort_values("Date", ascending=False),
                    hide_index=True,
                    width='content',
                    height=get_dataframe_height(gl_disp[avail]),
                )

                # ── Hit rate analysis ─────────────────────────────────────
                st.markdown("### 🎯 Hit Rate Analysis")
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    l3 = game_log.tail(3)
                    o3 = int((l3["stat"] > dk_line).sum())
                    hc1.metric("Last 3 Games", f"{o3}/3 Over",
                               f"{o3/3*100:.0f}% hit rate" if len(l3) >= 3 else "—")
                with hc2:
                    l5 = game_log.tail(5)
                    o5 = int((l5["stat"] > dk_line).sum())
                    hc2.metric("Last 5 Games", f"{o5}/5 Over",
                               f"{o5/5*100:.0f}% hit rate" if len(l5) >= 5 else "—")
                with hc3:
                    tg = res["total_games"]
                    ov = res["games_over"]
                    hc3.metric(f"Last {tg} Games", f"{ov}/{tg} Over",
                               f"{ov/tg*100:.0f}% hit rate" if tg > 0 else "—")

                # ── Trend chart ───────────────────────────────────────────
                st.markdown("### 📊 Trend Chart")
                chart_df = game_log.tail(20).copy()
                chart_df["Game"] = range(1, len(chart_df) + 1)
                fig = px.bar(
                    chart_df,
                    x="Game",
                    y="stat",
                    color=chart_df["stat"].apply(lambda x: "Over" if x > dk_line else "Under"),
                    color_discrete_map={"Over": "#16a34a", "Under": "#dc2626"},
                    title=f"{selected_name} — {prop} (Last {len(chart_df)} games)",
                    labels={"stat": prop},
                )
                fig.add_hline(y=dk_line, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"DK Line {dk_line:.1f}", annotation_position="top right")
                fig.update_layout(showlegend=True, legend_title_text="vs Line")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("ℹ️ How This Works"):
                    st.markdown(f"""
**Probability model** — fits a Normal distribution to this player's last 10 games (weighted 70%)
and full-season average (30%) to estimate `P(stat > line)`.

**Confidence Tiers:**
| Tier | Probability | Meaning |
|------|------------|---------|
| 🔥 ELITE  | ≥ 65% | Highest-confidence pick |
| 💪 STRONG | 60–65% | Very confident |
| ✅ GOOD   | 55–60% | Solid above breakeven |
| ⚠️ LEAN   | < 55% | Lower confidence — use sparingly |

**MORE** = model expects player to *exceed* the line  
**LESS** = model expects player to *fall short*

**Suggested Line** shown by default is 87% of season avg rounded to nearest 0.5.
Always verify against the actual DraftKings line before placing a pick.
                    """)

    # =========================================================================
    # TAB 2 — Top Picks
    # =========================================================================
    with tab_top:
        st.subheader("⭐ Top Prop Picks by Category")
        st.markdown("Season average vs. suggested DK lines — sorted by confidence tier.")

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            top_prop = st.selectbox(
                "Prop Type",
                ["Hits", "Home Runs", "RBI", "Runs", "Strikeouts"],
                key="top_prop",
            )
        with tc2:
            top_season = st.selectbox("Season", sorted(_SEASONS, reverse=True), key="top_season")
        with tc3:
            top_n = st.slider("Players to show", 10, 50, 25, key="top_n")

        st.markdown("---")

        # Load appropriate leaders table
        if top_prop == "Strikeouts":
            leaders = _pitching_leaders_cached(top_season, top_season)
            if leaders.empty:
                st.warning(f"No pitching data available for {top_season}.")
            else:
                leaders = leaders[leaders["SO"] > 0].copy()
                leaders["season_avg"] = (leaders["SO"] / leaders["GS"].where(leaders["GS"] > 0, 1)).round(2)
                leaders["Sug. Line"] = leaders["season_avg"].apply(_suggested_line)
                leaders["_ratio"]   = leaders["season_avg"] / leaders["Sug. Line"].where(leaders["Sug. Line"] > 0, 1)
                leaders["Tier"] = leaders["_ratio"].apply(
                    lambda r: "🔥 ELITE" if r >= 1.20 else "💪 STRONG" if r >= 1.12 else "✅ GOOD" if r >= 1.05 else "⚠️ LEAN"
                )
                leaders = leaders.sort_values("season_avg", ascending=False).head(top_n).copy()
                leaders.insert(0, "#", range(1, len(leaders) + 1))

                t1, t2, t3 = st.columns(3)
                t1.metric("🔥 Elite",  len(leaders[leaders["Tier"] == "🔥 ELITE"]))
                t2.metric("💪 Strong", len(leaders[leaders["Tier"] == "💪 STRONG"]))
                t3.metric("✅ Good",   len(leaders[leaders["Tier"] == "✅ GOOD"]))
                st.markdown("---")

                show = ["#", "full_name", "team", "GS", "SO", "season_avg", "Sug. Line", "Tier"]
                avail = [c for c in show if c in leaders.columns]
                disp = leaders[avail].rename(columns={"full_name": "Player", "team": "Team",
                                                       "season_avg": "K/Start"})
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(1)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(
                    f"Top {len(leaders)} K/Start candidates · {top_season} season · "
                    "Suggested Line ≈ 87% of K/start avg · Verify against live DK lines."
                )

        else:
            # Batting props
            bat_col_map = {"Hits": "H", "Home Runs": "HR", "RBI": "RBI", "Runs": "R"}
            stat_col = bat_col_map.get(top_prop)
            leaders = _batting_leaders_cached(top_season, top_season)
            if leaders.empty or stat_col is None or stat_col not in leaders.columns:
                st.warning(f"No batting data available for {top_season}.")
            else:
                leaders = leaders[leaders["PA"] >= 50].copy()
                leaders["season_avg"] = (leaders[stat_col] / leaders["PA"].where(leaders["PA"] > 0, np.nan) * 4.5).round(2)
                # Better per-game approx: stat_col / number of games (PA/4.5 ≈ games)
                g_approx = (leaders["PA"] / 4.5).clip(lower=1)
                leaders["per_game"] = (leaders[stat_col] / g_approx).round(3)
                leaders["Sug. Line"] = leaders["per_game"].apply(_suggested_line)
                leaders = leaders[leaders["Sug. Line"] >= 0.5]
                leaders["_ratio"] = leaders["per_game"] / leaders["Sug. Line"].where(leaders["Sug. Line"] > 0, 1)
                leaders["Tier"] = leaders["_ratio"].apply(
                    lambda r: "🔥 ELITE" if r >= 1.20 else "💪 STRONG" if r >= 1.12 else "✅ GOOD" if r >= 1.05 else "⚠️ LEAN"
                )
                leaders = leaders.sort_values("per_game", ascending=False).head(top_n).copy()
                leaders.insert(0, "#", range(1, len(leaders) + 1))

                t1, t2, t3 = st.columns(3)
                t1.metric("🔥 Elite",  len(leaders[leaders["Tier"] == "🔥 ELITE"]))
                t2.metric("💪 Strong", len(leaders[leaders["Tier"] == "💪 STRONG"]))
                t3.metric("✅ Good",   len(leaders[leaders["Tier"] == "✅ GOOD"]))
                st.markdown("---")

                show = ["#", "full_name", "team", "PA", stat_col, "per_game", "Sug. Line", "Tier"]
                avail = [c for c in show if c in leaders.columns]
                disp = leaders[avail].rename(columns={
                    "full_name": "Player", "team": "Team",
                    stat_col: f"{top_prop} (Season)", "per_game": f"{top_prop}/Game",
                })
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(2)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(
                    f"Top {len(leaders)} {top_prop}/game candidates · {top_season} season · "
                    "Suggested Line ≈ 87% of per-game avg · Always verify against live DK lines."
                )

        with st.expander("ℹ️ Tier calculation"):
            st.markdown("""
| Tier | Ratio (avg ÷ line) | Meaning |
|------|-------------------|---------|
| 🔥 ELITE  | ≥ 1.20× | Player consistently exceeds this line |
| 💪 STRONG | 1.12–1.20× | Very likely to hit |
| ✅ GOOD   | 1.05–1.12× | Solid edge above breakeven |
| ⚠️ LEAN   | < 1.05× | No strong season-average edge |

**Suggested Line** is a conservative estimate (~87% of per-game average).
The actual DraftKings line may differ — always check before betting.
            """)

    # =========================================================================
    # TAB 3 — Season Leaders
    # =========================================================================
    with tab_leaders:
        st.subheader("🏆 Season Leaders")

        lc1, lc2 = st.columns(2)
        with lc1:
            l_season = st.selectbox("Season", sorted(_SEASONS, reverse=True), key="leaders_season")
        with lc2:
            l_top = st.slider("Top N", 10, 50, 20, key="leaders_top")

        bat_lead = _batting_leaders_cached(l_season, l_season)
        pit_lead = _pitching_leaders_cached(l_season, l_season)

        ltab1, ltab2, ltab3, ltab4, ltab5 = st.tabs([
            "🪄 Hits", "💣 Home Runs", "🏃 Runs", "🎯 RBI", "⚡ K (Pitcher)"
        ])

        def _show_bat_leaders(df: pd.DataFrame, stat: str, label: str, top: int):
            if df.empty or stat not in df.columns:
                st.info("No data available.")
                return
            g_approx = (df["PA"] / 4.5).clip(lower=1)
            d = df.copy()
            d["per_game"] = (d[stat] / g_approx).round(3)
            d = d.sort_values(stat, ascending=False).head(top).copy()
            d.insert(0, "#", range(1, len(d) + 1))
            # Extra context cols — exclude stat itself to avoid duplicates
            extra = [c for c in ["BA", "HR"] if c != stat and c in d.columns]
            avail_cols = [c for c in ["#", "full_name", "team", "PA", stat, "per_game"] + extra if c in d.columns]
            rename = {"full_name": "Player", "team": "Team", stat: label, "per_game": f"{label}/G", "BA": "AVG"}
            disp = d[avail_cols].rename(columns=rename)
            for c in disp.select_dtypes("float").columns:
                disp[c] = disp[c].round(3 if "AVG" in disp.columns else 1)
            st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
            st.caption(f"Top {len(d)} by {label} — {l_season}")

        with ltab1:
            _show_bat_leaders(bat_lead, "H", "Hits", l_top)
        with ltab2:
            _show_bat_leaders(bat_lead, "HR", "Home Runs", l_top)
        with ltab3:
            _show_bat_leaders(bat_lead, "R", "Runs", l_top)
        with ltab4:
            _show_bat_leaders(bat_lead, "RBI", "RBI", l_top)
        with ltab5:
            if pit_lead.empty or "SO" not in pit_lead.columns:
                st.info("No pitching data available.")
            else:
                d = pit_lead.sort_values("SO", ascending=False).head(l_top).copy()
                d.insert(0, "#", range(1, len(d) + 1))
                d["K/GS"] = (d["SO"] / d["GS"].where(d["GS"] > 0, 1)).round(2)
                cols = [c for c in ["#", "full_name", "team", "GS", "IP", "SO", "K/GS", "ERA", "WHIP"] if c in d.columns]
                disp = d[cols].rename(columns={"full_name": "Player", "team": "Team"})
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(2)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(f"Top {len(d)} by Strikeouts — {l_season}")

    add_betting_oracle_footer()


main()
