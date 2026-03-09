"""Nightly Savant leaderboard refresh for the current MLB season.

Intended to run as a cron job or scheduled task during the season.
Only fetches the current year with a low PA minimum to capture
callups, recent performers, and relievers.

Usage:
    python scripts/nightly_savant_refresh.py          # uses current year
    python scripts/nightly_savant_refresh.py --year 2025

Schedule suggestion (crontab / Windows Task Scheduler):
    Run at 5:00 AM ET daily during April–October.
    Games finish by ~1 AM ET; Savant updates by ~4 AM ET.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.savant_leaderboard import (
    download_savant_leaderboard,
)
from src.ingestion.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# April 1 – October 31 (regular season + early postseason)
SEASON_START_MONTH = 4
SEASON_END_MONTH = 10


def games_tomorrow() -> bool:
    """Conservative check: are we in the MLB season window?"""
    today = date.today()
    return SEASON_START_MONTH <= today.month <= SEASON_END_MONTH


def nightly_refresh(year: int) -> None:
    """Pull today's cumulative Savant leaderboards and overwrite the CSVs.

    Uses a low PA/IP minimum so we capture:
      - Recently called-up hitters (min 10 PA)
      - Spot starters and relievers (min 1 IP → min_pa="1")
    """
    if not games_tomorrow():
        logger.info("Off-season (month %d) — skipping nightly refresh.", date.today().month)
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info("=== Nightly Savant refresh for %d  (%s) ===", year, timestamp)

    # ── Batters ──────────────────────────────────────────────────────────
    logger.info("Fetching batter leaderboard (min 10 PA)...")
    bat_df = download_savant_leaderboard(year, player_type="batter", min_pa="10")

    bat_path = config.raw_dir / "batting" / f"savant_batter_{year}_latest.csv"
    bat_path.parent.mkdir(parents=True, exist_ok=True)
    bat_df.to_csv(bat_path, index=False)
    logger.info("  Batters: %d rows, %d cols → %s", len(bat_df), len(bat_df.columns), bat_path)

    # ── Pitchers ─────────────────────────────────────────────────────────
    logger.info("Fetching pitcher leaderboard (min 1 IP)...")
    pit_df = download_savant_leaderboard(year, player_type="pitcher", min_pa="1")

    pit_path = config.raw_dir / "pitching" / f"savant_pitcher_{year}_latest.csv"
    pit_path.parent.mkdir(parents=True, exist_ok=True)
    pit_df.to_csv(pit_path, index=False)
    logger.info("  Pitchers: %d rows, %d cols → %s", len(pit_df), len(pit_df.columns), pit_path)

    logger.info("=== Nightly refresh complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly Savant leaderboard refresh")
    parser.add_argument(
        "--year", type=int, default=date.today().year,
        help="Season to refresh (default: current year)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run even outside the normal season window",
    )
    args = parser.parse_args()

    if args.force:
        # Override season check
        global games_tomorrow
        games_tomorrow = lambda: True  # noqa: E731

    nightly_refresh(args.year)


if __name__ == "__main__":
    main()
