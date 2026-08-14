"""Closing-line value ladder with explicit book/consensus definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.contracts.domain import Quote
from src.markets.pricing import implied_probability, log_price_clv


@dataclass(frozen=True)
class ClvPoint:
    minutes_before_start: int
    quote_id: str | None
    closing_probability: float | None
    clv_probability: float | None
    clv_log_price: float | None


def closing_quote(
    quotes: list[Quote],
    *,
    game_start: datetime,
    minutes_before_start: int,
    bookmaker_id: str | None = None,
) -> Quote | None:
    cutoff = game_start - timedelta(minutes=minutes_before_start)
    eligible = [
        row
        for row in quotes
        if row.observed_at <= cutoff
        and not row.is_suspended
        and (bookmaker_id is None or row.bookmaker_id == bookmaker_id)
    ]
    return max(eligible, key=lambda row: row.observed_at, default=None)


def clv_ladder(
    execution_quote: Quote,
    quotes: list[Quote],
    *,
    game_start: datetime,
    steps: tuple[int, ...] = (60, 30, 10, 1),
    same_book: bool = True,
) -> list[ClvPoint]:
    bet_break_even = implied_probability(float(execution_quote.price_decimal))
    results = []
    matching = [
        row
        for row in quotes
        if (row.game_id, row.market_id, row.selection, row.point)
        == (
            execution_quote.game_id,
            execution_quote.market_id,
            execution_quote.selection,
            execution_quote.point,
        )
    ]
    for minutes in steps:
        close = closing_quote(
            matching,
            game_start=game_start,
            minutes_before_start=minutes,
            bookmaker_id=execution_quote.bookmaker_id if same_book else None,
        )
        if close is None:
            results.append(ClvPoint(minutes, None, None, None, None))
            continue
        close_probability = implied_probability(float(close.price_decimal))
        results.append(
            ClvPoint(
                minutes,
                close.quote_id,
                close_probability,
                close_probability - bet_break_even,
                log_price_clv(float(execution_quote.price_decimal), float(close.price_decimal)),
            )
        )
    return results
