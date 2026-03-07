# src/ingestion/scheduler.py
"""APScheduler-based daily data refresh.

Run directly to start the blocking scheduler:
    python -m src.ingestion.scheduler
"""

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .mlb_stats import fetch_todays_probable_pitchers
from .odds import fetch_current_odds
from .weather import fetch_weather_for_games
from .season import in_season

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()


@scheduler.scheduled_job(CronTrigger(hour=8, minute=0, timezone="America/New_York"))
def morning_data_pull() -> None:
    """8 AM ET – Pull today's schedule and probable pitchers."""
    if not in_season():
        logger.info("Out of season – skipping morning data pull")
        return

    logger.info("Running morning data pull...")
    schedule = fetch_todays_probable_pitchers()
    logger.info(f"Found {len(schedule)} games today")


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
    """4 PM ET – Pull updated odds + weather (closer to game time)."""
    if not in_season():
        logger.info("Out of season – skipping afternoon update")
        return

    logger.info("Running afternoon update...")
    fetch_current_odds()
    schedule = fetch_todays_probable_pitchers()
    fetch_weather_for_games(schedule)


@scheduler.scheduled_job(CronTrigger(hour=1, minute=0, timezone="America/New_York"))
def overnight_results() -> None:
    """1 AM ET – Pull final scores and update model results."""
    if not in_season():
        logger.info("Out of season – skipping overnight results collection")
        return

    logger.info("Running overnight results collection...")
    # TODO: fetch final scores, update pick results, recalculate metrics


if __name__ == "__main__":
    logger.info("Starting ingestion scheduler...")
    scheduler.start()
