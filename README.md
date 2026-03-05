# baseball-predictions

MLB betting predictions powered by machine learning. The goal of this project is to generate daily wagering recommendations, track historical performance, and provide a dashboard for exploring model results and bankroll metrics. Built with Python, Streamlit, and Parquet/CSV data storage.

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
