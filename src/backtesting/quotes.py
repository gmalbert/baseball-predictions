"""Archived quote matching at a simulated decision cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.contracts.domain import Prediction, Quote
from src.markets.consensus import best_actionable_quote


@dataclass(frozen=True)
class QuoteMatch:
    prediction: Prediction
    quote: Quote | None
    eligible: bool
    reason: str | None


def match_archived_quote(
    prediction: Prediction,
    quotes: list[Quote],
    *,
    as_of: datetime,
    max_age_seconds: int,
    jurisdiction: str | None = None,
) -> QuoteMatch:
    exact = [
        row
        for row in quotes
        if (row.game_id, row.market_id, row.selection)
        == (prediction.game_id, prediction.market_id, prediction.selection)
    ]
    quote = best_actionable_quote(
        exact,
        selection=prediction.selection,
        as_of=as_of,
        max_age_seconds=max_age_seconds,
        jurisdiction=jurisdiction,
    )
    if quote is None:
        reason = "missing_quote" if not exact else "no_eligible_quote"
        return QuoteMatch(prediction, None, False, reason)
    return QuoteMatch(prediction, quote, True, None)
