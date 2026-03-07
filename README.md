<img src="https://raw.githubusercontent.com/gmalbert/betting-oracle/main/data_files/logo.png" width="200" alt="Betting Cleanup logo">

# Betting Cleanup

All-in-one MLB betting analytics platform.

*Generate daily picks, backtest model performance, and explore results through an
interactive Streamlit dashboard.  The app preloads all data for instant tab
switching, uses a light theme for readability, and automatically retrains models
via GitHub Actions during the baseball season.*

Built in Python with a Parquet/CSV‑based data store, XGBoost models, and
Live‑Odds integration.


## Roadmap

| # | Document | Description |
|---|----------|-------------|
| 00 | [Roadmap Overview](docs/00-roadmap-overview.md) | Phases, tech stack, and project scope |
| 01 | [Data Sources](docs/01-data-sources.md) | MLB stats, odds, weather, and historical data APIs |
| 02 | [Data Ingestion](docs/02-data-ingestion.md) | Pipelines to fetch, clean, and store data as Parquet |
| 03 | [Data Schema & Storage](docs/03-database-schema.md) | Parquet schemas, file layout, and optional PostgreSQL |
| 04 | [Betting Models](docs/04-betting-models.md) | XGBoost models for underdog, spread, and totals |
| 05 | [Model Evaluation](docs/05-model-evaluation.md) | Backtesting, calibration, and performance metrics |
| 06 | [Daily Picks Engine](docs/06-daily-picks-engine.md) | End-to-end daily pipeline and result settlement |
| 07 | [Data Access Layer](docs/07-backend-api.md) | Query functions for Streamlit + optional FastAPI |
| 08 | [Streamlit Dashboard](docs/08-frontend-layout.md) | Multi-page dashboard design and components |
| 09 | [Deployment & Ops](docs/09-deployment-ops.md) | Docker, CI/CD, Streamlit Cloud, monitoring |
| 10 | [Bankroll & Risk](docs/10-bankroll-strategy.md) | Kelly criterion, bankroll tracking, responsible gambling |

---

## Highlights

* **Instant dashboard** – all primary datasets are cached on first load, so tab
  switches feel instantaneous.
* **Light theme only** – adopted after feedback; easy to read in bright
environments.
* **Automated model training** – GitHub Actions workflow retrains models daily
  during baseball season (March–November) using `scripts/train_models.py`.
* **Evaluation tab** – run walk‑forward backtests, calibration charts, and ROI
  reports directly from the UI.
* **Integrated footer** – Betting Oracle branding appears on every page via
  `footer.py`.

---

## Getting Started

1. Clone the repo (now named **Betting Cleanup**) and create a Python 3.11+
   virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Populate `data_files/raw/` by running the ingestion cron or manually using
   the functions in `src/ingestion/`.
4. Launch the dashboard:
   ```bash
   streamlit run predictions.py
   ```
5. Train models via the **Models** tab or wait for the automated workflow.

For more details see the docs linked above.
