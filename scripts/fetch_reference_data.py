"""Nightly refresh of slow-changing reference data.

Fetches and caches:
  1. FanGraphs Guts! wOBA / cFIP constants → data_files/processed/fg_guts.parquet
  2. FanGraphs handedness-split park factors (current year)
     → data_files/processed/fg_park_{year}.parquet
  3. Chadwick Bureau player ID registry (Retrosheet ↔ MLBAM ↔ FanGraphs)
     → data_files/processed/player_registry.parquet

Rationale:
  - fg_guts: refreshed annually; safe to re-fetch daily in case late-season corrections land.
  - fg_park: refreshed each season; fetched daily so mid-season corrections are picked up.
  - player_registry: new players are added throughout the season; weekly refresh is enough
    but daily re-fetch is cheap (CSV is ~1 MB, no auth required).

This script is idempotent — re-running it just overwrites the cached Parquet files.

Usage:
    python scripts/fetch_reference_data.py
"""
from __future__ import annotations

import sys
import logging
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def refresh_fg_guts() -> None:
    """Re-fetch FanGraphs Guts! wOBA and cFIP constants."""
    from src.ingestion.fg_guts import fetch_fg_guts
    logger.info("Fetching FanGraphs Guts! constants…")
    df = fetch_fg_guts(save=True)
    logger.info("  fg_guts: %d season rows cached", len(df))


def refresh_fg_park_factors(year: int) -> None:
    """Re-fetch FanGraphs park factors (L + R) for the given season."""
    from src.ingestion.fg_park import fetch_fg_park_factors
    logger.info("Fetching FanGraphs park factors for %d (L + R)…", year)
    df = fetch_fg_park_factors(year, save=True)
    logger.info("  fg_park %d: %d rows cached", year, len(df))


def refresh_player_registry() -> None:
    """Re-download the Chadwick Bureau player ID cross-reference."""
    from src.ingestion.chadwick import load_player_registry
    logger.info("Fetching Chadwick Bureau player registry…")
    df = load_player_registry(force_refresh=True)
    logger.info("  player_registry: %d players cached", len(df))


if __name__ == "__main__":
    year = date.today().year
    errors: list[str] = []

    for label, fn in [
        ("fg_guts",          refresh_fg_guts),
        (f"fg_park_{year}",  lambda: refresh_fg_park_factors(year)),
        ("player_registry",  refresh_player_registry),
    ]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s FAILED: %s", label, exc)
            errors.append(label)

    if errors:
        logger.warning("Some reference data fetches failed: %s", errors)
        logger.warning("Falling back to cached files (if present) or hardcoded constants.")
    else:
        logger.info("All reference data refreshed successfully.")
