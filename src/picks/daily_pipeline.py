"""Daily picks generation pipeline.

Run the full morning pipeline (8 AM – 11:25 AM window):
    1. Fetch schedule + probable pitchers
    2. Fetch consensus odds
    3. Fetch weather
    4. Build feature matrix
    5. Run all 3 models
    6. Filter by edge / confidence thresholds
    7. Store picks to CSV  (parquet append coming later)

The morning consensus odds snapshot is saved so the 4 PM afternoon_refresh
job can compare lines and detect significant movement.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.mlb_stats import fetch_todays_probable_pitchers
from src.ingestion.odds import fetch_current_odds, get_consensus_line
from src.ingestion.weather import fetch_weather_for_games
from src.models.underdog_model import predict_moneyline
from src.models.spread_model import predict_spread
from src.models.totals_model import predict_totals

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data_files" / "processed"

# Minimum thresholds to publish a pick
MIN_EDGE_UNDERDOG = 0.03
MIN_EDGE_SPREAD = 0.03
MIN_EDGE_TOTALS = 0.025
MIN_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_daily_pipeline(target_date: Optional[date] = None) -> dict:
    """Run the full morning picks-generation pipeline.

    Returns:
        dict with keys ``underdog``, ``spread``, ``over_under``, each a
        list of pick dicts.
    """
    target_date = target_date or date.today()
    logger.info("Running daily pipeline for %s", target_date)

    # ---- 1. Schedule --------------------------------------------------------
    schedule = fetch_todays_probable_pitchers()
    if schedule.empty:
        logger.warning("No games found for today")
        return {"underdog": [], "spread": [], "over_under": []}
    logger.info("Found %d games today", len(schedule))

    # ---- 2. Odds ------------------------------------------------------------
    odds_raw = fetch_current_odds()
    consensus = get_consensus_line(odds_raw)

    # Persist morning snapshot so afternoon_refresh can detect movement
    _save_consensus_snapshot(consensus, target_date, label="morning")

    game_odds = _pivot_odds(consensus)

    # ---- 3. Weather ---------------------------------------------------------
    weather = fetch_weather_for_games(schedule)

    # ---- 4. Feature matrix --------------------------------------------------
    features = _build_todays_features(schedule, game_odds, weather)

    # ---- 5 & 6. Models + filtering ------------------------------------------
    picks: dict = {}

    underdog_preds = predict_moneyline(
        model_or_path=MODEL_DIR / "moneyline_xgb_v1.joblib",
        game_features=features,
        home_ml_col="home_moneyline",
        away_ml_col="away_moneyline",
    )
    picks["underdog"] = _format_picks(
        _filter_picks(underdog_preds, MIN_EDGE_UNDERDOG, MIN_CONFIDENCE), "underdog"
    )

    spread_preds = predict_spread(
        model_or_path=MODEL_DIR / "spread_xgb_v1.joblib",
        game_features=features,
        spread_price_col="home_spread_price",
    )
    picks["spread"] = _format_picks(
        _filter_picks(spread_preds, MIN_EDGE_SPREAD, MIN_CONFIDENCE), "spread"
    )

    totals_preds = predict_totals(
        model_or_path=MODEL_DIR / "totals_lgbm_v1.joblib",
        game_features=features,
        over_price_col="over_price",
        under_price_col="under_price",
    )
    picks["over_under"] = _format_picks(
        _filter_picks(totals_preds, MIN_EDGE_TOTALS, MIN_CONFIDENCE), "over_under"
    )

    # ---- 7. Storage ---------------------------------------------------------
    _store_picks(picks, target_date, source="morning")

    total = sum(len(v) for v in picks.values())
    logger.info(
        "Generated %d picks: %d underdog, %d spread, %d O/U",
        total,
        len(picks["underdog"]),
        len(picks["spread"]),
        len(picks["over_under"]),
    )
    return picks


# ---------------------------------------------------------------------------
# Internal helpers (also imported by afternoon_refresh)
# ---------------------------------------------------------------------------

def _pivot_odds(consensus: pd.DataFrame) -> pd.DataFrame:
    """Pivot consensus odds into one wide row per game."""
    games = consensus[["game_id", "away_team", "home_team"]].drop_duplicates().copy()

    for _, row in consensus[consensus["market"] == "h2h"].iterrows():
        mask = games["game_id"] == row["game_id"]
        col = "home_moneyline" if row["outcome_name"] == row.get("home_team", "") else "away_moneyline"
        games.loc[mask, col] = row["median_price"]

    for _, row in consensus[consensus["market"] == "spreads"].iterrows():
        mask = games["game_id"] == row["game_id"]
        if row["outcome_name"] == row.get("home_team", ""):
            games.loc[mask, "home_spread_point"] = row["median_point"]
            games.loc[mask, "home_spread_price"] = row["median_price"]
        else:
            games.loc[mask, "away_spread_point"] = row["median_point"]
            games.loc[mask, "away_spread_price"] = row["median_price"]

    for _, row in consensus[consensus["market"] == "totals"].iterrows():
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
    """Merge schedule, odds, and weather into a feature matrix."""
    features = schedule.merge(odds, on=["away_team", "home_team"], how="left")
    if "game_id" not in features.columns:
        if "game_id_x" in features.columns:
            features = features.rename(columns={"game_id_x": "game_id"})
            if "game_id_y" in features.columns:
                features = features.drop(columns=["game_id_y"])
        elif "game_id_y" in features.columns:
            features = features.rename(columns={"game_id_y": "game_id"})

    if not weather.empty:
        weather_cols = [
            c for c in ["game_id", "temp_f", "wind_mph", "wind_dir_deg",
                         "precip_prob_pct", "is_dome"]
            if c in weather.columns
        ]
        if "game_id" in weather_cols:
            features = features.merge(weather[weather_cols], on="game_id", how="left")

    # TODO: merge team_season_stats and pitcher_stats from processed/
    return features


def _filter_picks(predictions: pd.DataFrame, min_edge: float, min_confidence: float) -> pd.DataFrame:
    """Return only rows that clear edge and confidence thresholds."""
    edge_col = next((c for c in ("edge", "edge_home", "edge_away", "edge_over") if c in predictions.columns), None)
    conf_col = next((c for c in ("confidence_score", "pick_prob") if c in predictions.columns), None)

    mask = pd.Series([True] * len(predictions), index=predictions.index)
    if edge_col:
        mask &= predictions[edge_col].fillna(0) >= min_edge
    if conf_col:
        mask &= predictions[conf_col].fillna(0) >= min_confidence

    sort_col = conf_col or edge_col
    filtered = predictions[mask].copy()
    if sort_col:
        filtered = filtered.sort_values(sort_col, ascending=False)
    return filtered


def _format_picks(df: pd.DataFrame, pick_type: str) -> list[dict]:
    """Serialize a predictions DataFrame to a list of pick dicts."""
    picks = []
    for _, row in df.iterrows():
        d = row.to_dict()
        picks.append({
            "game_id": d.get("game_id"),
            "away_team": d.get("away_team") or d.get("visteam"),
            "home_team": d.get("home_team") or d.get("hometeam"),
            "pick_type": pick_type,
            "pick_value": d.get("pick_side") or d.get("pick") or "",
            "predicted_prob": round(float(d.get("pred_home_win_prob") or d.get("pick_prob") or d.get("pred_cover_prob") or 0), 3),
            "confidence_score": round(float(d.get("pick_prob") or d.get("pred_cover_prob") or d.get("pred_home_win_prob") or 0), 3),
            "edge": round(float(d.get("edge") or d.get("edge_home") or d.get("edge_away") or d.get("edge_over") or 0), 3),
        })
    return picks


def _store_picks(picks: dict, target_date: date, source: str = "morning") -> None:
    """Write picks to the daily CSV, tagging each row with its source."""
    all_picks = [p for pick_list in picks.values() for p in pick_list]
    if not all_picks:
        return

    df = pd.DataFrame(all_picks)
    df["date"] = target_date.isoformat()
    df["source"] = source

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / f"picks_{target_date.isoformat()}.csv"
    df.to_csv(outpath, index=False)
    logger.info("Picks saved → %s", outpath)


def _save_consensus_snapshot(consensus: pd.DataFrame, target_date: date, label: str) -> None:
    """Persist a consensus odds snapshot for later comparison."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"consensus_{target_date.isoformat()}_{label}.parquet"
    consensus.to_parquet(path, index=False)
    logger.info("Consensus snapshot (%s) saved → %s", label, path)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    run_daily_pipeline()
