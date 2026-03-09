"""Afternoon picks refresh — 4 PM ET job.

Steps:
    1. Re-fetch current odds (updated lines).
    2. Compare against the morning consensus snapshot to find significant
       line movement.
    3. Re-run all three models for games that moved.
    4. Merge updated picks back into today's picks CSV, replacing morning
       picks for any affected game.

Thresholds for "significant" movement (configurable):
    - Moneyline:  |price_move| >= 10 American-odds points
    - Spread:     |point_move| >= 0.5 (half-run shift)
    - Totals:     |point_move| >= 0.5 (half-run shift in posted total)
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
from src.picks.daily_pipeline import (
    MODEL_DIR,
    MIN_CONFIDENCE,
    MIN_EDGE_SPREAD,
    MIN_EDGE_TOTALS,
    MIN_EDGE_UNDERDOG,
    PROCESSED_DIR,
    _build_todays_features,
    _filter_picks,
    _format_picks,
    _pivot_odds,
    _save_consensus_snapshot,
)

logger = logging.getLogger(__name__)

# Line-movement thresholds
MONEYLINE_MOVE_THRESHOLD: int = 10   # American-odds points (absolute)
SPREAD_MOVE_THRESHOLD: float = 0.5   # run-line point shift
TOTAL_MOVE_THRESHOLD: float = 0.5    # posted-total point shift


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_line_movement(
    morning_consensus: pd.DataFrame,
    afternoon_consensus: pd.DataFrame,
) -> pd.DataFrame:
    """Compare morning and afternoon consensus to find significant line moves.

    Args:
        morning_consensus:   Output of ``get_consensus_line()`` from the morning run.
        afternoon_consensus: Output of ``get_consensus_line()`` from the 4 PM fetch.

    Returns:
        DataFrame with one row per game/market/outcome containing movement
        metrics and a boolean ``significant`` column.
    """
    join_cols = ["game_id", "away_team", "home_team", "market", "outcome_name"]
    merged = morning_consensus.merge(
        afternoon_consensus,
        on=join_cols,
        suffixes=("_am", "_pm"),
    )

    merged["price_move"] = (merged["median_price_pm"] - merged["median_price_am"]).abs()
    merged["point_move"] = (merged["median_point_pm"] - merged["median_point_am"]).abs().fillna(0)

    ml_move = (merged["market"] == "h2h") & (merged["price_move"] >= MONEYLINE_MOVE_THRESHOLD)
    sp_move = (merged["market"] == "spreads") & (merged["point_move"] >= SPREAD_MOVE_THRESHOLD)
    tot_move = (merged["market"] == "totals") & (merged["point_move"] >= TOTAL_MOVE_THRESHOLD)

    merged["significant"] = ml_move | sp_move | tot_move

    return merged[[
        "game_id", "away_team", "home_team", "market", "outcome_name",
        "median_price_am", "median_price_pm", "price_move",
        "median_point_am", "median_point_pm", "point_move",
        "significant",
    ]]


def afternoon_picks_refresh(target_date: Optional[date] = None) -> dict:
    """Re-fetch odds, detect line movement, and update picks for moved games.

    Returns:
        dict with the same shape as ``run_daily_pipeline()`` (keys
        ``underdog``, ``spread``, ``over_under``) containing only the
        picks that were updated.  Returns ``{}`` when nothing moved.
    """
    target_date = target_date or date.today()
    logger.info("Starting afternoon picks refresh for %s", target_date)

    # ---- 1. Re-fetch odds ---------------------------------------------------
    odds_raw = fetch_current_odds()
    afternoon_consensus = get_consensus_line(odds_raw)
    _save_consensus_snapshot(afternoon_consensus, target_date, label="afternoon")

    # ---- 2. Load morning snapshot and detect movement -----------------------
    morning_path = PROCESSED_DIR / f"consensus_{target_date.isoformat()}_morning.parquet"
    if morning_path.exists():
        morning_consensus = pd.read_parquet(morning_path)
        movements = detect_line_movement(morning_consensus, afternoon_consensus)
        significant = movements[movements["significant"]]
        moved_game_ids: set = set(significant["game_id"].unique())

        if moved_game_ids:
            logger.info(
                "Significant line movement in %d game(s): %s",
                len(moved_game_ids),
                ", ".join(str(g) for g in moved_game_ids),
            )
            _log_movements(significant)
        else:
            logger.info("No significant line movement detected — skipping pick update")
            return {}
    else:
        logger.warning(
            "Morning consensus snapshot not found at %s; re-running all games",
            morning_path,
        )
        moved_game_ids = set(afternoon_consensus["game_id"].unique())

    # ---- 3. Re-run models for affected games --------------------------------
    schedule = fetch_todays_probable_pitchers()
    weather = fetch_weather_for_games(schedule)
    game_odds = _pivot_odds(afternoon_consensus)
    features = _build_todays_features(schedule, game_odds, weather)

    features_moved = features[features["game_id"].isin(moved_game_ids)].copy()
    if features_moved.empty:
        logger.warning("Moved game IDs not found in today's feature matrix — nothing to update")
        return {}

    picks: dict = {}

    underdog_preds = predict_moneyline(
        model_or_path=MODEL_DIR / "moneyline_xgb_v1.joblib",
        game_features=features_moved,
        home_ml_col="home_moneyline",
        away_ml_col="away_moneyline",
    )
    picks["underdog"] = _format_picks(
        _filter_picks(underdog_preds, MIN_EDGE_UNDERDOG, MIN_CONFIDENCE), "underdog"
    )

    spread_preds = predict_spread(
        model_or_path=MODEL_DIR / "spread_xgb_v1.joblib",
        game_features=features_moved,
        spread_price_col="home_spread_price",
    )
    picks["spread"] = _format_picks(
        _filter_picks(spread_preds, MIN_EDGE_SPREAD, MIN_CONFIDENCE), "spread"
    )

    totals_preds = predict_totals(
        model_or_path=MODEL_DIR / "totals_lgbm_v1.joblib",
        game_features=features_moved,
        over_price_col="over_price",
        under_price_col="under_price",
    )
    picks["over_under"] = _format_picks(
        _filter_picks(totals_preds, MIN_EDGE_TOTALS, MIN_CONFIDENCE), "over_under"
    )

    # ---- 4. Merge updated picks back into daily CSV -------------------------
    _merge_afternoon_picks(picks, moved_game_ids, target_date)

    total = sum(len(v) for v in picks.values())
    logger.info(
        "Afternoon refresh complete: %d updated picks (%d underdog, %d spread, %d O/U)",
        total,
        len(picks["underdog"]),
        len(picks["spread"]),
        len(picks["over_under"]),
    )
    return picks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_afternoon_picks(picks: dict, moved_game_ids: set, target_date: date) -> None:
    """Replace morning picks for moved games with the revised afternoon picks."""
    all_new = [p for pick_list in picks.values() for p in pick_list]
    if not all_new:
        # Models produced no publishable picks for moved games — still remove
        # the stale morning picks so we don't surface outdated lines.
        all_new = []

    new_df = pd.DataFrame(all_new) if all_new else pd.DataFrame()
    if not new_df.empty:
        new_df["date"] = target_date.isoformat()
        new_df["source"] = "afternoon_refresh"

    picks_path = PROCESSED_DIR / f"picks_{target_date.isoformat()}.csv"
    if picks_path.exists():
        existing = pd.read_csv(picks_path)
        existing = existing[~existing["game_id"].isin(moved_game_ids)]
        merged = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing
    else:
        merged = new_df

    if merged.empty:
        logger.info("No picks remaining after afternoon merge")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(picks_path, index=False)
    logger.info("Updated picks saved → %s", picks_path)


def _log_movements(significant: pd.DataFrame) -> None:
    """Emit one INFO line per significant movement."""
    for _, row in significant.iterrows():
        if row["market"] == "h2h":
            logger.info(
                "  ML move  | %s @ %s | %s: %+.0f → %+.0f (Δ%+.0f pts)",
                row["away_team"], row["home_team"], row["outcome_name"],
                row["median_price_am"], row["median_price_pm"], row["price_move"],
            )
        else:
            logger.info(
                "  %s move | %s @ %s | %s: %.1f → %.1f (Δ%.1f)",
                row["market"], row["away_team"], row["home_team"], row["outcome_name"],
                row["median_point_am"], row["median_point_pm"], row["point_move"],
            )


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    afternoon_picks_refresh()
