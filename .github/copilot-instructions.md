# Copilot Instructions — Baseball Predictions

## Project Overview

MLB baseball betting prediction system that uses machine learning models to generate daily picks. Built entirely in Python with a Streamlit dashboard for visualization.

## Tech Stack

- **Language:** Python 3.11+
- **Dashboard:** Streamlit (multi-page app in `streamlit_app/`)
- **Data Storage:** CSV and Parquet files (primary), PostgreSQL (optional)
- **ML Models:** XGBoost, LightGBM, scikit-learn
- **Charting:** Plotly, Altair
- **Scheduling:** APScheduler
- **API (optional):** FastAPI + Uvicorn
- **Deployment:** Streamlit Cloud, Docker, GitHub Actions CI/CD

## Project Structure

```
baseball-predictions/
├── data_files/
│   ├── raw/              # Downloaded CSVs from APIs
│   └── processed/        # Consolidated Parquet files
├── docs/                 # Roadmap documentation (00-10)
├── models/               # Serialized .joblib model files
├── src/
│   ├── ingestion/        # Data fetching & Parquet consolidation
│   ├── models/           # Feature engineering & ML training
│   ├── picks/            # Daily pipeline & settlement
│   ├── bankroll/         # Kelly criterion, risk management
│   ├── data/             # Data access layer (queries.py, cache.py)
│   └── api/              # Optional FastAPI endpoints
├── streamlit_app/
│   ├── app.py            # Entry point
│   ├── pages/            # Multi-page Streamlit pages
│   ├── components/       # Reusable UI helpers
│   └── .streamlit/       # Theme config
├── tests/
├── requirements.txt
└── docker-compose.yml
```

## Data Flow

1. **Ingestion** — Fetch MLB stats, odds, weather via APIs → save as CSV in `data_files/raw/`
2. **Consolidation** — Convert CSVs to Parquet in `data_files/processed/`
3. **Feature Engineering** — Build feature matrices from Parquet files
4. **Model Prediction** — Run XGBoost models → output probabilities
5. **Pick Generation** — Calculate edge vs. implied odds, filter by confidence
6. **Storage** — Save picks to `data_files/processed/picks_history.parquet`
7. **Dashboard** — Streamlit reads Parquet files directly via `src/data/queries.py`

## Coding Conventions

- Use **type hints** on all function signatures
- Use **dataclasses** for structured data (not plain dicts)
- Use **pathlib.Path** for file paths, never string concatenation
- Use **pyarrow schemas** when writing Parquet files for type safety
- Prefer **pandas DataFrames** as the primary data interchange format
- Use **`@st.cache_data`** decorator for all Streamlit data-loading functions
- Use **Plotly** for interactive charts in the dashboard
- Follow **ruff** for linting and formatting

## Data Storage Rules

- **Primary storage:** Parquet files in `data_files/processed/`
- **Raw downloads:** CSV files in `data_files/raw/`
- **Never commit** raw data files or `.joblib` model files to git
- Use `pyarrow` schemas defined in `src/data/schemas.py` for consistency
- PostgreSQL is optional — only reference it in clearly marked "optional" sections

## Key Files

- `src/data/queries.py` — All data access functions (reads Parquet, returns DataFrames)
- `src/data/cache.py` — Streamlit caching wrappers around query functions
- `src/picks/daily_pipeline.py` — Main daily orchestration pipeline
- `src/bankroll/kelly.py` — Kelly Criterion bet sizing
- `streamlit_app/app.py` — Streamlit entry point

## ML Models

Three XGBoost models, each targeting a different bet type:
- **Underdog Moneyline** — Predicts upset probability for dogs at +120 or longer
- **Run Line (Spread)** — Predicts whether a team covers -1.5 / +1.5
- **Totals (Over/Under)** — Predicts whether the game goes over or under the posted total

Models are trained with `scikit-learn` pipelines and serialized with `joblib`.

## Confidence Tiers

- **HIGH** — Edge > 6%, strong model agreement, half-Kelly sizing
- **MEDIUM** — Edge 3-6%, single model pick, quarter-Kelly sizing
- **LOW** — Edge 1-3%, tracked but minimal sizing

## Testing

- Use `pytest` for all tests
- Place tests in `tests/` mirroring `src/` structure
- Mock API calls in ingestion tests
- Use fixture Parquet files for data access layer tests

## What NOT to Do

- Do not use React, Next.js, or any JavaScript framework
- Do not assume PostgreSQL is available — always use Parquet as the default
- Do not hardcode API keys — use environment variables or `st.secrets`
- Do not place business logic in Streamlit pages — keep it in `src/`
