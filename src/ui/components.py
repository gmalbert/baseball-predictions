"""Shared accessible display helpers; no external text is treated as HTML."""

from __future__ import annotations

import html
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo


def dataframe_height(
    row_count: int, row_px: int = 35, header_px: int = 38, maximum: int = 900
) -> int:
    return min(maximum, max(120, header_px + max(1, row_count) * row_px))


def freshness_label(age_seconds: float | None) -> tuple[str, str]:
    if age_seconds is None:
        return "Missing", "red"
    if age_seconds <= 180:
        return "Fresh", "green"
    if age_seconds <= 900:
        return "Aging", "orange"
    return "Stale", "red"


def safe_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def format_probability(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_decimal_odds(value: Decimal | float | None) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def local_time(value: datetime, timezone_name: str = "America/New_York") -> str:
    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")
    return value.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %I:%M %p %Z")


def recommendation_label(action: str, reasons: tuple[str, ...] = ()) -> str:
    if action == "bet":
        return "Eligible opportunity"
    if reasons:
        return f"Abstain — {', '.join(reasons)}"
    return "No opportunity"
