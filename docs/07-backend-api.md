# 07 – Data Access Layer

A Python data-access module that reads Parquet/CSV files and serves data to the Streamlit dashboard. An optional FastAPI layer is included if you later need a REST API.

---

## Project Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── schemas.py        # Parquet schemas (from 03-database-schema.md)
│   ├── queries.py        # High-level data query functions
│   └── cache.py          # Simple caching for Streamlit
├── api/                  # Optional: FastAPI REST endpoints
│   ├── __init__.py
│   ├── main.py
│   └── routers/
│       ├── picks.py
│       └── models.py
```

---

## 1. Data Query Functions

The primary data access layer — reads from Parquet files and returns DataFrames ready for Streamlit.

```python
# src/data/queries.py
"""Data query functions for the Streamlit dashboard.

All functions return pandas DataFrames read from Parquet files.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data_files" / "processed"


def _load(name: str) -> pd.DataFrame:
    """Load a Parquet file. Returns empty DataFrame if not found."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---- Picks ----

def get_todays_picks(
    pick_type: Optional[str] = None,
    min_confidence: Optional[str] = None,
) -> pd.DataFrame:
    """Get all picks for today's games.
    
    Args:
        pick_type: Filter by 'underdog', 'spread', 'over_under'
        min_confidence: Filter by 'high', 'medium', 'low'
    """
    picks = _load("picks_history")
    if picks.empty:
        return picks

    today = pd.Timestamp.today().normalize()
    picks["date"] = pd.to_datetime(picks["date"])
    df = picks[picks["date"] == today].copy()

    if pick_type:
        df = df[df["pick_type"] == pick_type]

    if min_confidence:
        conf_order = {"high": 3, "medium": 2, "low": 1}
        min_rank = conf_order.get(min_confidence, 0)
        df = df[df["confidence"].map(conf_order).fillna(0) >= min_rank]

    return df.sort_values("confidence_score", ascending=False)


def get_picks_by_date(target_date: date) -> pd.DataFrame:
    """Get picks for a specific date."""
    picks = _load("picks_history")
    if picks.empty:
        return picks
    picks["date"] = pd.to_datetime(picks["date"]).dt.date
    return picks[picks["date"] == target_date].sort_values(
        "confidence_score", ascending=False
    )


def get_pick_history(
    days: int = 30,
    pick_type: Optional[str] = None,
    result_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Get historical pick results."""
    picks = _load("picks_history")
    if picks.empty:
        return picks

    picks["date"] = pd.to_datetime(picks["date"]).dt.date
    cutoff = date.today() - timedelta(days=days)
    df = picks[(picks["date"] >= cutoff) & (picks["result"].notna())].copy()

    if pick_type:
        df = df[df["pick_type"] == pick_type]
    if result_filter:
        df = df[df["result"] == result_filter]

    return df.sort_values("date", ascending=False)


def get_pick_summary(days: int = 30) -> dict:
    """Get aggregated summary of pick performance."""
    df = get_pick_history(days)
    if df.empty:
        return {
            "total_picks": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "total_units": 0, "roi": 0,
        }

    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    profit = df["actual_profit"].sum()

    return {
        "total_picks": len(df),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_units": round(float(profit), 2),
        "roi": round(float(profit / len(df)), 4) if len(df) > 0 else 0,
    }


# ---- Model Performance ----

def get_model_leaderboard(days: int = 30) -> pd.DataFrame:
    """Get model performance rankings."""
    df = get_pick_history(days)
    if df.empty:
        return pd.DataFrame()

    leaderboard = df.groupby(["model_name", "pick_type"]).agg(
        total_picks=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        total_profit=("actual_profit", "sum"),
        avg_confidence=("confidence_score", "mean"),
        avg_edge=("edge", "mean"),
    ).reset_index()

    leaderboard["win_rate"] = leaderboard["wins"] / (
        leaderboard["wins"] + leaderboard["losses"]
    )
    leaderboard["roi"] = leaderboard["total_profit"] / leaderboard["total_picks"]

    return leaderboard.sort_values("total_profit", ascending=False)


def get_cumulative_profit(
    model_name: Optional[str] = None, days: int = 90
) -> pd.DataFrame:
    """Get cumulative profit over time for charting."""
    df = get_pick_history(days)
    if df.empty:
        return pd.DataFrame()

    if model_name:
        df = df[df["model_name"] == model_name]

    df = df.sort_values("date")
    df["cumulative_units"] = df["actual_profit"].cumsum()
    df["bet_count"] = range(1, len(df) + 1)

    return df[["date", "cumulative_units", "bet_count"]]


def get_game_details(target_date: date) -> pd.DataFrame:
    """Get schedule and game info for a date."""
    games = _load("schedules")
    if games.empty:
        return games
    games["date"] = pd.to_datetime(games["date"]).dt.date
    return games[games["date"] == target_date]


def get_confidence_breakdown(days: int = 30) -> pd.DataFrame:
    """Get performance breakdown by confidence tier."""
    df = get_pick_history(days)
    if df.empty:
        return pd.DataFrame()

    breakdown = df.groupby("confidence").agg(
        bets=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        profit=("actual_profit", "sum"),
    ).reset_index()

    breakdown["win_rate"] = breakdown["wins"] / (breakdown["wins"] + breakdown["losses"])
    breakdown["roi"] = breakdown["profit"] / breakdown["bets"]

    return breakdown
```

---

## 2. Streamlit Caching Layer

```python
# src/data/cache.py
"""Caching utilities for Streamlit.

Streamlit re-runs the entire script on every interaction.
@st.cache_data prevents re-reading Parquet files on every click.
"""

import streamlit as st
import pandas as pd
from datetime import date
from src.data.queries import (
    get_todays_picks,
    get_pick_history,
    get_pick_summary,
    get_model_leaderboard,
    get_cumulative_profit,
    get_confidence_breakdown,
)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_todays_picks(pick_type=None, min_confidence=None):
    return get_todays_picks(pick_type, min_confidence)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_pick_history(days=30, pick_type=None):
    return get_pick_history(days, pick_type)


@st.cache_data(ttl=600)
def cached_summary(days=30):
    return get_pick_summary(days)


@st.cache_data(ttl=600)
def cached_leaderboard(days=30):
    return get_model_leaderboard(days)


@st.cache_data(ttl=600)
def cached_cumulative_profit(model_name=None, days=90):
    return get_cumulative_profit(model_name, days)


@st.cache_data(ttl=600)
def cached_confidence_breakdown(days=30):
    return get_confidence_breakdown(days)
```

---

## 3. Optional: FastAPI REST API

If external consumers (mobile app, third-party integrations) need JSON endpoints, add a thin FastAPI layer on top of the same query functions.

```python
# src/api/main.py
"""Optional FastAPI application — use if you need a REST API."""

from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Baseball Predictions API...")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Baseball Predictions API",
    description="MLB betting picks powered by ML models",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

```python
# src/api/routers/picks.py
"""Optional REST endpoints wrapping the same query functions."""

from fastapi import APIRouter, Query
from datetime import date, timedelta
from typing import Optional
from src.data.queries import (
    get_todays_picks,
    get_picks_by_date,
    get_pick_history,
    get_pick_summary,
)

router = APIRouter()


@router.get("/today")
async def today_picks(
    pick_type: Optional[str] = None,
    min_confidence: Optional[str] = None,
):
    df = get_todays_picks(pick_type, min_confidence)
    return df.to_dict(orient="records")


@router.get("/date/{target_date}")
async def picks_by_date(target_date: date):
    df = get_picks_by_date(target_date)
    return df.to_dict(orient="records")


@router.get("/history")
async def pick_history(
    days: int = Query(30, ge=1, le=365),
    pick_type: Optional[str] = None,
    result_filter: Optional[str] = None,
):
    df = get_pick_history(days, pick_type, result_filter)
    return df.to_dict(orient="records")


@router.get("/summary")
async def summary(days: int = Query(30)):
    return get_pick_summary(days)
```

```bash
# Run the optional API (only if needed)
uvicorn src.api.main:app --reload --port 8000
```

---

## Key Endpoints Summary (Optional FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/picks/today` | Today's picks (filterable) |
| GET | `/api/picks/date/{date}` | Picks for a specific date |
| GET | `/api/picks/history` | Historical results (paginated) |
| GET | `/api/picks/summary` | Aggregated win rate, ROI |
| GET | `/api/models/leaderboard` | Model performance rankings |
| GET | `/health` | Health check |

---

> **Next:** [08-frontend-layout.md](08-frontend-layout.md) – Streamlit dashboard design and implementation.
