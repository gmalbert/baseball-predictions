# src/ingestion/scheduler.py
"""APScheduler-based daily data refresh.

Run directly to start the blocking scheduler:
    python -m src.ingestion.scheduler
"""

import logging

from datetime import date, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .mlb_stats import fetch_todays_probable_pitchers
from .odds import fetch_current_odds
from .season import in_season
from src.ingestion.odds import get_consensus_line
from src.picks.afternoon_refresh import afternoon_picks_refresh
from src.picks.daily_pipeline import _save_consensus_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()


@scheduler.scheduled_job(CronTrigger(hour=8, minute=0, timezone="America/New_York"))
def morning_data_pull() -> None:
    """8 AM ET – Pull today's opening consensus odds snapshot."""
    if not in_season():
        logger.info("Out of season – skipping morning data pull")
        return

    logger.info("Running morning data pull...")
    schedule = fetch_todays_probable_pitchers()
    odds_raw = fetch_current_odds()
    consensus = get_consensus_line(odds_raw)
    _save_consensus_snapshot(consensus, date.today(), label="morning")
    logger.info("Found %d games today", len(schedule))


@scheduler.scheduled_job(CronTrigger(hour=11, minute=0, timezone="America/New_York"))
def midday_odds_pull() -> None:
    """11 AM ET – Pull opening odds for today's slate."""
    if not in_season():
        logger.info("Out of season – skipping midday odds pull")
        return

    logger.info("Running midday odds pull...")
    fetch_current_odds()


@scheduler.scheduled_job(CronTrigger(hour=16, minute=0, timezone="America/New_York"))
def afternoon_update() -> None:
    """4 PM ET – Re-fetch odds, detect line movement, update affected picks."""
    if not in_season():
        logger.info("Out of season – skipping afternoon update")
        return

    logger.info("Running afternoon update...")
    afternoon_picks_refresh()


@scheduler.scheduled_job(CronTrigger(hour=1, minute=0, timezone="America/New_York"))
def overnight_results() -> None:
    """1 AM ET – Pull final scores, update model results, refresh reference data."""
    if not in_season():
        logger.info("Out of season – skipping overnight results collection")
        return

    logger.info("Running overnight results collection...")
    # TODO: fetch final scores, update pick results, recalculate metrics

    # Refresh slow-changing reference data: fg_guts, park factors, Chadwick registry
    logger.info("Refreshing reference data (fg_guts, park factors, player registry)…")
    from src.ingestion.fg_guts import fetch_fg_guts
    from src.ingestion.fg_park import fetch_fg_park_factors
    from src.ingestion.chadwick import load_player_registry
    try:
        fetch_fg_guts(save=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fg_guts refresh failed: %s", exc)
    try:
        fetch_fg_park_factors(date.today().year, save=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fg_park refresh failed: %s", exc)
    try:
        load_player_registry(force_refresh=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("player_registry refresh failed: %s", exc)
    logger.info("Reference data refresh complete")


if __name__ == "__main__":
    logger.info("Starting ingestion scheduler...")
    scheduler.start()
