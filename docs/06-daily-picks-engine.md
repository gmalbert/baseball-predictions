# 06 – Daily Picks Engine

Orchestrates the end-to-end flow: fetch today's data → run models → output picks → store results.

---

## Pipeline Overview

```
 8:00 AM   Fetch schedule + probable pitchers
 8:05 AM   Fetch team stats (rolling, season-level)
11:00 AM   Fetch odds (opening lines)
11:05 AM   Fetch weather forecasts
11:10 AM   Build feature matrix for today's games
11:15 AM   Run all 3 models → raw predictions
11:20 AM   Apply filters (min edge, min confidence)
11:25 AM   Format picks, store in DB, publish to API
 4:00 PM   Re-fetch odds (updated lines), re-run models
 4:15 PM   Update picks if significant line movement
11:00 PM   Fetch final scores
11:05 PM   Settle picks (win/loss/push), update metrics
```

---

## 1. Main Daily Pipeline

```python
# src/picks/daily_pipeline.py
"""Daily picks generation pipeline."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.odds import fetch_current_odds, get_consensus_line
from src.ingestion.weather import fetch_weather_for_games
from src.models.features import (
    build_game_features, implied_probability, calculate_edge,
    UNDERDOG_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES,
)
from src.models.underdog_model import predict_underdog
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# Minimum thresholds to publish a pick
MIN_EDGE_UNDERDOG = 0.03
MIN_EDGE_SPREAD = 0.03
MIN_EDGE_TOTALS = 0.025
MIN_CONFIDENCE = 0.30


def run_daily_pipeline(
    target_date: Optional[date] = None,
) -> dict:
    """Run the full daily picks pipeline.
    
    Returns dict with picks for each model type.
    """
    target_date = target_date or date.today()
    logger.info(f"Running daily pipeline for {target_date}")
    
    # ---- Step 1: Fetch today's schedule ----
    schedule = fetch_todays_probable_pitchers()
    if schedule.empty:
        logger.warning("No games found for today")
        return {"underdog": [], "spread": [], "over_under": []}
    
    logger.info(f"Found {len(schedule)} games today")
    
    # ---- Step 2: Fetch odds ----
    odds_raw = fetch_current_odds()
    consensus = get_consensus_line(odds_raw)
    
    # Pivot odds into game-level columns
    game_odds = _pivot_odds(consensus)
    
    # ---- Step 3: Fetch weather ----
    weather = fetch_weather_for_games(schedule)
    
    # ---- Step 4: Build features ----
    # (In production, load team/pitcher stats from DB)
    features = _build_todays_features(schedule, game_odds, weather)
    
    # ---- Step 5: Run models ----
    picks = {}
    
    # Underdog picks
    underdog_preds = predict_underdog(
        model_path=str(MODEL_DIR / "underdog_xgb_v1.joblib"),
        game_features=features,
    )
    underdog_picks = _filter_picks(underdog_preds, MIN_EDGE_UNDERDOG, MIN_CONFIDENCE)
    picks["underdog"] = _format_picks(underdog_picks, "underdog")
    
    # Spread picks
    spread_preds = predict_spread(
        model_path=str(MODEL_DIR / "spread_xgb_v1.joblib"),
        game_features=features,
    )
    spread_picks = _filter_picks(spread_preds, MIN_EDGE_SPREAD, MIN_CONFIDENCE)
    picks["spread"] = _format_picks(spread_picks, "spread")
    
    # Over/Under picks
    totals_preds = predict_totals(
        model_path=str(MODEL_DIR / "totals_xgb_v1.joblib"),
        game_features=features,
    )
    totals_picks = _filter_picks(totals_preds, MIN_EDGE_TOTALS, MIN_CONFIDENCE)
    picks["over_under"] = _format_picks(totals_picks, "over_under")
    
    # ---- Step 6: Store picks ----
    _store_picks(picks, target_date)
    
    # ---- Step 7: Summary ----
    total = sum(len(v) for v in picks.values())
    logger.info(f"Generated {total} picks: "
                f"{len(picks['underdog'])} underdog, "
                f"{len(picks['spread'])} spread, "
                f"{len(picks['over_under'])} O/U")
    
    return picks


def _pivot_odds(consensus: pd.DataFrame) -> pd.DataFrame:
    """Pivot consensus odds into game-level columns.
    
    Input:  long format (one row per game/market/outcome)
    Output: wide format (one row per game with all odds as columns)
    """
    games = consensus[["game_id", "away_team", "home_team"]].drop_duplicates()
    
    # Moneyline
    ml = consensus[consensus["market"] == "h2h"]
    for _, row in ml.iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == row.get("home_team", ""):
            games.loc[mask, "home_moneyline"] = row["median_price"]
        else:
            games.loc[mask, "away_moneyline"] = row["median_price"]
    
    # Spreads
    sp = consensus[consensus["market"] == "spreads"]
    for _, row in sp.iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == row.get("home_team", ""):
            games.loc[mask, "home_spread_point"] = row["median_point"]
            games.loc[mask, "home_spread_price"] = row["median_price"]
        else:
            games.loc[mask, "away_spread_point"] = row["median_point"]
            games.loc[mask, "away_spread_price"] = row["median_price"]
    
    # Totals
    tot = consensus[consensus["market"] == "totals"]
    for _, row in tot.iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == "Over":
            games.loc[mask, "posted_total"] = row["median_point"]
            games.loc[mask, "over_price"] = row["median_price"]
        else:
            games.loc[mask, "under_price"] = row["median_price"]
    
    return games


def _build_todays_features(
    schedule: pd.DataFrame,
    odds: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge schedule, odds, and weather into a feature matrix.
    
    In production, this also loads team stats and pitcher stats from DB.
    """
    features = schedule.merge(odds, on=["away_team", "home_team"], how="left")
    
    if not weather.empty:
        features = features.merge(
            weather[["game_id", "temp_f", "wind_mph", "wind_dir_deg",
                      "precip_prob_pct", "is_dome"]],
            on="game_id",
            how="left",
        )
    
    # TODO: merge team_season_stats and pitcher_stats from DB
    # features = features.merge(team_stats, ...)
    # features = features.merge(pitcher_stats, ...)
    
    return features


def _filter_picks(
    predictions: pd.DataFrame,
    min_edge: float,
    min_confidence: float,
) -> pd.DataFrame:
    """Filter predictions to only publishable picks."""
    filtered = predictions[
        (predictions["edge"] >= min_edge) &
        (predictions["confidence_score"] >= min_confidence)
    ].copy()
    
    return filtered.sort_values("confidence_score", ascending=False)


def _format_picks(df: pd.DataFrame, pick_type: str) -> list[dict]:
    """Convert DataFrame to list of pick dicts for API/storage."""
    picks = []
    for _, row in df.iterrows():
        picks.append({
            "game_id": row.get("game_id"),
            "away_team": row.get("away_team"),
            "home_team": row.get("home_team"),
            "pick_type": pick_type,
            "pick_value": row.get("pick_value", ""),
            "predicted_prob": round(float(row.get("predicted_prob", 0)), 3),
            "confidence": row.get("confidence", "low"),
            "confidence_score": round(float(row.get("confidence_score", 0)), 3),
            "edge": round(float(row.get("edge", 0)), 3),
            "model_name": f"xgb_{pick_type}_v1",
        })
    return picks


def _store_picks(picks: dict, target_date: date):
    """Store picks to database and CSV backup."""
    all_picks = []
    for pick_type, pick_list in picks.items():
        all_picks.extend(pick_list)
    
    if not all_picks:
        return
    
    df = pd.DataFrame(all_picks)
    df["date"] = target_date.isoformat()
    
    # CSV backup
    outdir = Path(__file__).resolve().parents[2] / "data_files" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"picks_{target_date.isoformat()}.csv"
    df.to_csv(outpath, index=False)
    logger.info(f"Picks saved → {outpath}")
    
    # TODO: Also insert into daily_picks table via SQLAlchemy
```

---

## 2. Result Settlement

```python
# src/picks/settle.py
"""Settle picks after games complete: determine win/loss/push and profit."""

import pandas as pd
import statsapi
from datetime import date, timedelta
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/baseball_predictions"
)


def fetch_final_scores(target_date: date) -> pd.DataFrame:
    """Fetch final scores for all games on a given date."""
    date_str = target_date.strftime("%Y-%m-%d")
    games = statsapi.schedule(date=date_str)
    
    rows = []
    for g in games:
        if g["status"] == "Final":
            rows.append({
                "game_id": g["game_id"],
                "away_team": g["away_name"],
                "home_team": g["home_name"],
                "away_score": g["away_score"],
                "home_score": g["home_score"],
                "total_runs": g["away_score"] + g["home_score"],
            })
    
    return pd.DataFrame(rows)


def settle_underdog_pick(pick: dict, final: dict) -> dict:
    """Settle an underdog (moneyline) pick."""
    # Determine who the underdog was and if they won
    pick_value = pick["pick_value"]  # e.g., "Away +150" or "Home +130"
    
    if "Away" in pick_value:
        underdog_won = final["away_score"] > final["home_score"]
    else:
        underdog_won = final["home_score"] > final["away_score"]
    
    result = "win" if underdog_won else "loss"
    
    # Calculate profit (1 unit bet)
    odds = int(pick_value.split()[-1])
    if result == "win":
        profit = odds / 100 if odds > 0 else 100 / abs(odds)
    else:
        profit = -1.0
    
    return {"result": result, "profit": round(profit, 2)}


def settle_spread_pick(pick: dict, final: dict) -> dict:
    """Settle a spread (run line) pick."""
    pick_value = pick["pick_value"]  # e.g., "Home -1.5" or "Away +1.5"
    
    margin = final["home_score"] - final["away_score"]
    
    if "Home -1.5" in pick_value:
        # Home team must win by 2+
        if margin >= 2:
            result = "win"
        else:
            result = "loss"
    elif "Away +1.5" in pick_value:
        # Away team can lose by 1 or win
        if margin <= 1:
            result = "win"
        else:
            result = "loss"
    elif "Home +1.5" in pick_value:
        if margin >= -1:
            result = "win"
        else:
            result = "loss"
    elif "Away -1.5" in pick_value:
        if margin <= -2:
            result = "win"
        else:
            result = "loss"
    else:
        result = "push"
    
    # Standard -110 juice for spreads
    profit = 100 / 110 if result == "win" else (-1.0 if result == "loss" else 0.0)
    
    return {"result": result, "profit": round(profit, 2)}


def settle_totals_pick(pick: dict, final: dict) -> dict:
    """Settle an over/under (totals) pick."""
    pick_value = pick["pick_value"]  # e.g., "Over 8.5" or "Under 7.5"
    
    parts = pick_value.split()
    side = parts[0]        # "Over" or "Under"
    total = float(parts[1])  # 8.5
    actual = final["total_runs"]
    
    if actual == total:
        result = "push"
    elif side == "Over" and actual > total:
        result = "win"
    elif side == "Under" and actual < total:
        result = "win"
    else:
        result = "loss"
    
    # Standard -110 juice for totals
    profit = 100 / 110 if result == "win" else (-1.0 if result == "loss" else 0.0)
    
    return {"result": result, "profit": round(profit, 2)}


SETTLERS = {
    "underdog": settle_underdog_pick,
    "spread": settle_spread_pick,
    "over_under": settle_totals_pick,
}


def settle_day(target_date: date = None):
    """Settle all picks for a given date."""
    target_date = target_date or (date.today() - timedelta(days=1))
    
    logger.info(f"Settling picks for {target_date}")
    
    # Fetch final scores
    finals = fetch_final_scores(target_date)
    if finals.empty:
        logger.warning("No final scores found")
        return
    
    # Load unsettled picks for this date
    picks_path = (
        Path(__file__).resolve().parents[2]
        / "data_files" / "processed" / f"picks_{target_date.isoformat()}.csv"
    )
    
    if not picks_path.exists():
        logger.warning(f"No picks file found: {picks_path}")
        return
    
    picks_df = pd.read_csv(picks_path)
    
    settled = []
    for _, pick in picks_df.iterrows():
        pick_dict = pick.to_dict()
        
        # Match to final score
        final = finals[
            (finals["away_team"] == pick["away_team"]) &
            (finals["home_team"] == pick["home_team"])
        ]
        
        if final.empty:
            logger.warning(f"No final score for {pick['away_team']} @ {pick['home_team']}")
            continue
        
        final_dict = final.iloc[0].to_dict()
        settler = SETTLERS.get(pick["pick_type"])
        
        if settler:
            result = settler(pick_dict, final_dict)
            pick_dict.update(result)
            settled.append(pick_dict)
    
    # Save settled results
    settled_df = pd.DataFrame(settled)
    settled_path = picks_path.with_name(f"settled_{target_date.isoformat()}.csv")
    settled_df.to_csv(settled_path, index=False)
    
    # Print summary
    wins = sum(1 for s in settled if s["result"] == "win")
    losses = sum(1 for s in settled if s["result"] == "loss")
    profit = sum(s["profit"] for s in settled)
    
    logger.info(f"Settled {len(settled)} picks: "
                f"{wins}W - {losses}L, {profit:+.2f} units")
    
    return settled_df


if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    settle_day()
```

---

## 3. Pick Output Format

Each pick surfaced to users contains:

```json
{
  "game_id": 745123,
  "date": "2026-03-04",
  "away_team": "New York Yankees",
  "home_team": "Boston Red Sox",
  "venue": "Fenway Park",
  "game_time": "2026-03-04T23:10:00Z",
  "away_starter": "Gerrit Cole",
  "home_starter": "Brayan Bello",
  "picks": [
    {
      "pick_type": "underdog",
      "pick_value": "NYY +130",
      "predicted_prob": 0.482,
      "confidence": "medium",
      "confidence_score": 0.54,
      "edge": 0.047,
      "model_name": "xgb_underdog_v1"
    },
    {
      "pick_type": "spread",
      "pick_value": "BOS -1.5 (-120)",
      "predicted_prob": 0.401,
      "confidence": "low",
      "confidence_score": 0.28,
      "edge": 0.015,
      "model_name": "xgb_spread_v1"
    },
    {
      "pick_type": "over_under",
      "pick_value": "Over 8.5",
      "predicted_prob": 0.583,
      "confidence": "high",
      "confidence_score": 0.71,
      "edge": 0.062,
      "model_name": "xgb_totals_v1"
    }
  ]
}
```

---

> **Next:** [07-backend-api.md](07-backend-api.md) – Exposing picks and performance data via a REST API.
