# src/ingestion/season.py
"""Utilities for determining whether the current date falls within
MLB's active season window.

The goal is to avoid pulling data when there are no games scheduled
(e.g. November through February).  The check used by the GitHub Actions
workflow and by the APScheduler jobs themselves.
"""

from datetime import datetime

from .config import config


# approximate month range for the MLB regular season + postseason
# (spring training starts in February, but we don't need data then)
SEASON_MONTHS = set(range(3, 12))  # March (3) through November (11)


def in_season(today: datetime | None = None) -> bool:
    """Return ``True`` if ``today`` is within the configured season.

    The function uses three simple rules:

    1. Year must be between ``config.start_year`` and ``config.end_year``.
    2. Month must be one of ``SEASON_MONTHS``.
    3. If ``today`` is ``None`` the current date/time is used.

    The check is intentionally lightweight; it does not call any API.
    It is meant for guarding scheduled jobs and workflow runs.
    """
    if today is None:
        today = datetime.utcnow()

    if not (config.start_year <= today.year <= config.end_year):
        return False

    if today.month not in SEASON_MONTHS:
        return False

    return True
