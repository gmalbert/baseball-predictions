"""Page: Info — methodology, models, confidence tiers, data sources."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from page_utils import render_sidebar, add_betting_oracle_footer

render_sidebar(show_year_filter=False)

st.subheader("About This App")
st.markdown("""
### How Picks Are Generated

1. **Data Ingestion** — MLB game logs, Statcast leaderboards, weather and odds
   are fetched daily and stored as Parquet files in `data_files/processed/`.

2. **Feature Engineering** — Per-game features (pitching matchup, team form,
   park factors, weather, implied odds) are assembled into a model-ready matrix.

3. **Model Prediction** — Three XGBoost / LightGBM models output win
   probabilities for Moneyline, Run Line, and Totals markets.

4. **Edge Calculation** — Model probability is compared to the bookmaker's
   implied probability (from American odds).
   `edge = model_prob − implied_prob`

5. **Pick Filtering** — Bets with positive edge above a minimum threshold are
   ranked by confidence tier and Kelly-sized.

---

### Models

| Market | Model | Min Edge |
|--------|-------|----------|
| Underdog Moneyline | XGBoost (`moneyline_xgb_v1`) | +120 or longer |
| Run Line ±1.5 | XGBoost (`spread_xgb_v1`) | 3 % |
| Totals O/U | LightGBM (`totals_lgbm_v1`) | 3 % |

---

### Home Page Recommendations

The home page shows three markets per game using stats-based estimates:

| Market | Estimation Method |
|--------|------------------|
| **Moneyline** | Win probability from current-season W% (logistic, +4% HFA) vs implied odds |
| **Run Line ±1.5** | Cover probability estimate = `win_prob^1.4` (empirical MLB cover model) |
| **Over/Under** | Expected total from team RS/G (historical) vs posted O/U line |

For ML-model-driven picks (XGBoost trained on 2020–2025 data), see the **Models** page.

---

### Confidence Tiers

| Tier | Edge | Kelly Fraction | Sizing |
|------|------|---------------|--------|
| HIGH | > 6 % | Half-Kelly | Full unit |
| MEDIUM | 3 – 6 % | Quarter-Kelly | Half unit |
| LOW | 1 – 3 % | Eighth-Kelly | Quarter unit |

---

### Data Sources

- **Statcast / Baseball Savant** — Pitch-level and Statcast leaderboard data
  via the `pybaseball` library.
- **MLB Stats API** — Game schedules, scores, standings, and roster data.
- **Retrosheet** — Historical game logs for long-range backtesting.
- **The Odds API** — Opening and closing lines across major bookmakers.
- **ESPN Public API** — Free live odds feed, no API key required.
- **Open-Meteo** — Historical and forecast weather at ballpark locations.

---

### Technology Stack

| Component | Technology |
|-----------|------------|
| Dashboard | Streamlit |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Data Storage | Apache Parquet (pyarrow) |
| Charts | Plotly, Altair |
| Scheduling | APScheduler, GitHub Actions |
| Language | Python 3.11+ |
""")

add_betting_oracle_footer()
