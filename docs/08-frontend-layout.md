# 08 – Streamlit Dashboard & UX

Multi-page Streamlit app providing daily picks, model performance, and bankroll tracking.  

> **Status:** ❌ *No Streamlit code currently exists in the repository; the following sections describe the intended layout and wireframes for a future build.*

---

## Why Streamlit

| Criterion | Streamlit | Next.js / React |
|-----------|-----------|----------------|
| Language | Python only | TypeScript + Python API |
| Setup time | Minutes | Hours (build tooling, routing) |
| Charting | Plotly / Altair built-in | Extra libraries (Recharts, Chart.js) |
| Data access | Direct DataFrame usage | REST API + fetch layer |
| Deployment | Streamlit Cloud (free) | Vercel + separate API host |
| Customisation | Limited but sufficient | Full control |

**Recommendation: Streamlit** — single language, instant dashboards, reads Parquet files directly.

---

## App Structure (Multi-Page)

```

# The future layout would look like:

├── predictions.py              # Entry point + shared layout
├── pages/                      # individual page modules
│   ├── 1_Todays_Picks.py       # Daily picks view
│   ├── 2_Pick_History.py       # Historical ledger
│   ├── 3_Model_Performance.py  # Model leaderboard & charts
│   ├── 4_Bankroll.py           # Bankroll tracker & risk
│   └── 5_About.py              # Methodology explanation
├── components/                 # reusable UI helpers
│   ├── __init__.py
│   ├── pick_cards.py
│   ├── charts.py
│   └── metrics.py
└── .streamlit/                 # theme & server config (root-level exists)
    └── config.toml
```

> *Current repository only contains a top‑level `.streamlit/` folder; all Streamlit code must be added.*

---

## Page Wireframes

### 1. Today's Picks

```
┌──────────────────────────────────────────────────────────────┐
│  ⚾ Baseball Predictions                                     │
│  Sidebar: [Today] [History] [Models] [Bankroll] [About]      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TODAY'S PICKS — March 4, 2026           14 games · 9 picks  │
│                                                              │
│  ┌─ Quick Stats (st.columns) ────────────────────────────┐   │
│  │  Season: 127W-98L (56.4%)  │  +18.4 units  │  7.2% ROI │ │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Filter: [All Types ▾]  [All Confidence ▾]                   │
│                                                              │
│  ┌─ NYY @ BOS · Fenway Park · 7:10 PM ──────────────────┐   │
│  │  Cole vs Bello                                        │   │
│  │  🟢 HIGH  Over 8.5  │ Prob: 58.3%  Edge: +6.2%      │   │
│  │  🟡 MED   NYY +130  │ Prob: 48.2%  Edge: +4.7%      │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ LAD @ SF · Oracle Park · 9:45 PM ───────────────────┐   │
│  │  ...                                                  │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 2. Model Performance Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│  MODEL PERFORMANCE          Period: [Last 30 days ▾]         │
│                                                              │
│  ┌─ Leaderboard (st.dataframe) ──────────────────────────┐   │
│  │  Model          Type     W-L    Win%   Units   ROI    │   │
│  │  xgb_totals_v1  O/U     42-31  57.5%  +8.2   11.2%  │   │
│  │  xgb_dog_v1     ML      28-22  56.0%  +12.4   8.1%  │   │
│  │  xgb_spread_v1  Spread  35-30  53.8%  +2.1    3.2%  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ Cumulative Profit (st.plotly_chart) ─────────────────┐   │
│  │  📈 [line chart: units over time for each model]      │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ Calibration Plot ────────┐  ┌─ Confidence Breakdown ─┐   │
│  │  [scatter: pred vs actual]│  │  High:  18W-8L  +14.2  │   │
│  │  [diagonal = perfect]     │  │  Med:   52W-40L  +6.1  │   │
│  │                           │  │  Low:   35W-30L  +0.3  │   │
│  └───────────────────────────┘  └────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 3. Historical Ledger

```
┌──────────────────────────────────────────────────────────────┐
│  PICK HISTORY                                                │
│  Filter: [All Types ▾] [All Results ▾] [Last 30 days ▾]     │
│                                                              │
│  Date       Game           Pick         Conf   Result  P/L   │
│  ─────────────────────────────────────────────────────────── │
│  Mar 3      NYY @ BOS      Over 8.5     🟢     ✅ W   +0.91 │
│  Mar 3      LAD @ SF       LAD +120     🟡     ❌ L   -1.00 │
│  Mar 2      CHC @ MIL      CHC +145     🟡     ✅ W   +1.45 │
│  ...                                                         │
│                                                              │
│  Showing 30 of 225 picks  [Show More]                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Streamlit Configuration  

> **Status:** sample config only; `.streamlit/config.toml` exists at project root.

```toml
.streamlit/config.toml

[theme]
primaryColor = "#1e40af"       # Blue-700
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

---

<!-- ## 4. Entry Point  

> **Note:** This sample entry point is **not implemented** in the repo. It serves as a reference for when the Streamlit app is built.

```python

"""Main Streamlit app — entry point and shared config."""

import streamlit as st

st.set_page_config(
    page_title="Baseball Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding
st.sidebar.title("⚾ Baseball Predictions")
st.sidebar.caption("MLB picks powered by ML models")

st.title("Welcome")
st.markdown(
    """
    Use the sidebar to navigate:
    - **Today's Picks** — Model-generated picks for today's games
    - **Pick History** — Historical results ledger
    - **Model Performance** — Leaderboard and profit charts
    - **Bankroll** — Track bets and manage risk
    - **About** — Methodology and data sources
    """
)
``` -->

---

## 5. Today's Picks Page  

> **Note:** The following code is illustrative; no corresponding Python files exist yet.

```python
# streamlit_app/pages/1_Todays_Picks.py
"""Today's Picks page."""

import streamlit as st
import pandas as pd
from datetime import date
from src.data.cache import cached_todays_picks, cached_summary

st.set_page_config(page_title="Today's Picks", page_icon="⚾", layout="wide")
st.title(f"Today's Picks — {date.today().strftime('%B %d, %Y')}")

# ---- Quick Stats Row ----
summary = cached_summary(30)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Record", f"{summary['wins']}W - {summary['losses']}L")
col2.metric("Win Rate", f"{summary['win_rate']:.1%}")
col3.metric("Units Profit", f"{summary['total_units']:+.1f}")
col4.metric("ROI", f"{summary['roi']:.1%}")

st.divider()

# ---- Filters ----
fcol1, fcol2 = st.columns(2)
pick_type = fcol1.selectbox(
    "Pick Type",
    ["All", "underdog", "spread", "over_under"],
    index=0,
)
min_conf = fcol2.selectbox(
    "Min Confidence",
    ["All", "high", "medium", "low"],
    index=0,
)

picks = cached_todays_picks(
    pick_type=None if pick_type == "All" else pick_type,
    min_confidence=None if min_conf == "All" else min_conf,
)

if picks.empty:
    st.info("No picks available yet for today. Check back after 11:30 AM ET.")
    st.stop()

st.caption(f"{len(picks)} picks across today's games")

# ---- Group by game and display ----
for game_label, game_df in picks.groupby("game_label"):
    with st.expander(game_label, expanded=True):
        for _, pick in game_df.iterrows():
            _render_pick_row(pick)


def _render_pick_row(pick: pd.Series):
    """Render a single pick as a row of columns."""
    c1, c2, c3, c4 = st.columns([1, 3, 2, 2])

    # Confidence badge
    conf = pick.get("confidence", "low")
    badge_color = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(conf, "⚪")
    c1.markdown(f"**{badge_color} {conf.upper()}**")

    # Pick value
    c2.markdown(f"**{pick['pick_value']}**")

    # Probability
    prob = pick.get("predicted_prob", 0) * 100
    c3.markdown(f"Prob: **{prob:.1f}%**")

    # Edge
    edge = pick.get("edge", 0) * 100
    c4.markdown(f"Edge: **+{edge:.1f}%**")
```

---

## 6. Pick History Page

```python
# streamlit_app/pages/2_Pick_History.py
"""Historical pick results ledger."""

import streamlit as st
import pandas as pd
from src.data.cache import cached_pick_history

st.set_page_config(page_title="Pick History", page_icon="📊", layout="wide")
st.title("Pick History")

# ---- Filters ----
col1, col2, col3 = st.columns(3)
days = col1.selectbox("Period", [7, 14, 30, 60, 90], index=2)
pick_type = col2.selectbox("Type", ["All", "underdog", "spread", "over_under"])
result_filter = col3.selectbox("Result", ["All", "win", "loss", "push"])

df = cached_pick_history(
    days=days,
    pick_type=None if pick_type == "All" else pick_type,
)
if result_filter != "All":
    df = df[df["result"] == result_filter]

if df.empty:
    st.warning("No settled picks for the selected filters.")
    st.stop()

# ---- Summary metrics ----
wins = (df["result"] == "win").sum()
losses = (df["result"] == "loss").sum()
profit = df["actual_profit"].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Picks", len(df))
m2.metric("Record", f"{wins}W - {losses}L")
m3.metric("Win Rate", f"{wins / (wins + losses):.1%}" if (wins + losses) > 0 else "N/A")
m4.metric("Profit", f"{profit:+.2f} units")

st.divider()

# ---- Results table ----
display_cols = [
    "date", "game_label", "pick_type", "pick_value",
    "confidence", "result", "actual_profit",
]
available = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[available].style.applymap(
        lambda v: "color: green" if v == "win" else ("color: red" if v == "loss" else ""),
        subset=["result"] if "result" in available else [],
    ),
    width='stretch',
    hide_index=True,
)
```

---

## 7. Model Performance Page

```python
# streamlit_app/pages/3_Model_Performance.py
"""Model leaderboard and performance charts."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.data.cache import (
    cached_leaderboard,
    cached_cumulative_profit,
    cached_confidence_breakdown,
)

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")
st.title("Model Performance")

days = st.selectbox("Period", [7, 14, 30, 60, 90], index=2)

# ---- Leaderboard ----
st.subheader("Leaderboard")
lb = cached_leaderboard(days)
if lb.empty:
    st.info("No model data yet.")
    st.stop()

st.dataframe(
    lb[["model_name", "pick_type", "total_picks", "wins", "losses",
        "win_rate", "total_profit", "roi"]].style.format({
        "win_rate": "{:.1%}",
        "total_profit": "{:+.2f}",
        "roi": "{:.1%}",
        "avg_confidence": "{:.3f}",
    }),
    width='stretch',
    hide_index=True,
)

st.divider()

# ---- Cumulative Profit Chart ----
st.subheader("Cumulative Profit")
model_names = lb["model_name"].unique().tolist()
selected_model = st.selectbox("Model", ["All"] + model_names)

profit_df = cached_cumulative_profit(
    model_name=None if selected_model == "All" else selected_model,
    days=days,
)

if not profit_df.empty:
    fig = px.line(
        profit_df,
        x="date",
        y="cumulative_units",
        title="Cumulative Units Profit",
        labels={"cumulative_units": "Units", "date": "Date"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig, width='stretch')

# ---- Confidence Breakdown ----
st.subheader("Performance by Confidence Tier")
conf_df = cached_confidence_breakdown(days)

if not conf_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            conf_df,
            x="confidence",
            y=["wins", "losses"],
            barmode="group",
            title="Wins vs Losses by Confidence",
            color_discrete_map={"wins": "#10b981", "losses": "#ef4444"},
        )
        fig_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_bar, width='stretch')

    with col2:
        fig_roi = px.bar(
            conf_df,
            x="confidence",
            y="roi",
            title="ROI by Confidence Tier",
            color="roi",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        )
        fig_roi.update_layout(template="plotly_white")
        st.plotly_chart(fig_roi, width='stretch')
```

---

## 8. Bankroll Page

```python
# streamlit_app/pages/4_Bankroll.py
"""Bankroll tracking and risk management dashboard."""

import streamlit as st
import plotly.express as px
from src.bankroll.tracker import BankrollTracker
from src.bankroll.kelly import get_bet_sizing

st.set_page_config(page_title="Bankroll", page_icon="💰", layout="wide")
st.title("Bankroll Tracker")

# ---- Sidebar config ----
st.sidebar.subheader("Bankroll Settings")
starting_br = st.sidebar.number_input("Starting Bankroll ($)", value=10000, step=500)
unit_size = st.sidebar.number_input("Unit Size ($)", value=100, step=10)

# ---- Summary metrics ----
# In production, load from saved state (CSV/Parquet)
tracker = BankrollTracker(starting_bankroll=starting_br, unit_size=unit_size)
state = tracker.get_state()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Bankroll", f"${state.current_bankroll:,.0f}")
c2.metric("Total Profit", f"${state.total_profit:+,.0f}")
c3.metric("Win Rate", f"{state.win_rate:.1%}")
c4.metric("Total Bets", state.total_bets)

# ---- Bankroll Progress Bar ----
bankroll_pct = state.current_bankroll / state.starting_bankroll
color = "normal" if bankroll_pct >= 1.0 else "inverse" if bankroll_pct < 0.8 else "off"
st.progress(min(bankroll_pct, 1.0), text=f"Bankroll: {bankroll_pct:.0%} of starting")

st.divider()

# ---- Kelly Calculator ----
st.subheader("Bet Sizing Calculator (Kelly Criterion)")

kcol1, kcol2, kcol3 = st.columns(3)
prob = kcol1.slider("Predicted Win Probability", 0.30, 0.80, 0.55, 0.01)
odds = kcol2.number_input("American Odds", value=-110, step=5)
conf = kcol3.selectbox("Confidence", ["high", "medium", "low"])

sizing = get_bet_sizing(prob, odds, conf, starting_br, unit_size / starting_br)

st.info(
    f"**Recommended: {sizing.recommended_units:.2f} units "
    f"({sizing.recommended_pct:.2f}% of bankroll)** — {sizing.reason}"
)

with st.expander("Kelly Details"):
    st.write(f"- Full Kelly: {sizing.full_kelly_pct:.2f}%")
    st.write(f"- Half Kelly: {sizing.half_kelly_pct:.2f}%")
    st.write(f"- Quarter Kelly: {sizing.quarter_kelly_pct:.2f}%")

st.divider()

# ---- Daily P/L Chart ----
st.subheader("Cumulative Profit / Loss")
daily = tracker.daily_pnl()
if not daily.empty:
    fig = px.line(
        daily, x="date", y="cumulative",
        title="Cumulative P/L Over Time",
        labels={"cumulative": "Units", "date": "Date"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width='stretch')
else:
    st.info("No settled bets yet. P/L chart will appear once bets are tracked.")
```

---

## 9. About Page

```python
# streamlit_app/pages/5_About.py
"""About and methodology page."""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")
st.title("About This Project")

st.markdown("""
## Methodology

This site generates MLB betting picks using machine learning models
trained on historical game data, player statistics, and betting odds.

### Models
- **Underdog ML Model** — Identifies underdog moneyline value (XGBoost)
- **Run Line Model** — Predicts spread outcomes (XGBoost)
- **Totals Model** — Predicts over/under outcomes (XGBoost)

### Data Sources
- [MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI) — Schedule, rosters, stats
- [The Odds API](https://the-odds-api.com/) — Live betting odds
- [Retrosheet](https://retrosheet.org/) — Historical game logs
- [pybaseball](https://github.com/jldbc/pybaseball) — Statcast & advanced stats

### How Picks Are Generated
1. Fetch today's schedule and probable pitchers
2. Build feature matrices from rolling stats and odds
3. Run each model to produce win probabilities
4. Calculate edge vs. implied odds probability
5. Filter picks by minimum edge and confidence thresholds
6. Publish picks with Kelly-optimal bet sizing

### Confidence Tiers
- 🟢 **HIGH** — Edge > 6%, strong model agreement
- 🟡 **MEDIUM** — Edge 3-6%, single model pick
- ⚪ **LOW** — Edge 1-3%, included for tracking
""")

st.divider()

# ---- Responsible Gambling Notice ----
st.warning(
    "**Responsible Gambling Notice**\n\n"
    "This site provides statistical predictions for entertainment and "
    "informational purposes only. Past performance does not guarantee future "
    "results. Never bet more than you can afford to lose.\n\n"
    "If you or someone you know has a gambling problem, call "
    "**1-800-GAMBLER** or visit [ncpgambling.org](https://www.ncpgambling.org)."
)
```

---

## 10. Reusable Components

### Pick Card Helpers

```python
# streamlit_app/components/pick_cards.py
"""Reusable Streamlit components for displaying picks."""

import streamlit as st
import pandas as pd

CONF_BADGES = {"high": "🟢", "medium": "🟡", "low": "⚪"}


def render_pick_card(pick: pd.Series):
    """Render a single pick in a styled container."""
    conf = pick.get("confidence", "low")
    badge = CONF_BADGES.get(conf, "⚪")
    prob = pick.get("predicted_prob", 0) * 100
    edge = pick.get("edge", 0) * 100

    c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
    c1.markdown(f"**{badge} {conf.upper()}**")
    c2.markdown(f"**{pick['pick_value']}**")
    c3.metric("Prob", f"{prob:.1f}%", label_visibility="collapsed")
    c4.metric("Edge", f"+{edge:.1f}%", label_visibility="collapsed")


def render_game_card(game_label: str, picks_df: pd.DataFrame):
    """Render all picks for a single game inside an expander."""
    with st.expander(game_label, expanded=True):
        if picks_df.empty:
            st.caption("No picks for this game")
            return
        for _, pick in picks_df.iterrows():
            render_pick_card(pick)
            st.markdown("---")
```

### Chart Builders

```python
# streamlit_app/components/charts.py
"""Plotly chart builders for the dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def cumulative_profit_chart(df: pd.DataFrame, title: str = "Cumulative Profit") -> go.Figure:
    """Line chart of cumulative units over time."""
    fig = px.line(df, x="date", y="cumulative_units", title=title)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_white",
        height=400,
        yaxis_title="Units",
        xaxis_title="",
    )
    return fig


def confidence_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: wins vs losses by confidence tier."""
    fig = px.bar(
        df, x="confidence", y=["wins", "losses"],
        barmode="group",
        title="Record by Confidence",
        color_discrete_map={"wins": "#10b981", "losses": "#ef4444"},
    )
    fig.update_layout(template="plotly_white", height=350)
    return fig


def roi_by_confidence_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of ROI by confidence tier."""
    fig = px.bar(
        df, x="confidence", y="roi",
        title="ROI by Confidence Tier",
        color="roi",
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
    )
    fig.update_layout(template="plotly_white", height=350)
    return fig
```

---

## 11. Running the App

```bash
# From project root
streamlit run streamlit_app/app.py

# Or with custom port
streamlit run streamlit_app/app.py --server.port 8501
```

---

> **Next:** [09-deployment-ops.md](09-deployment-ops.md) – Getting it live.
