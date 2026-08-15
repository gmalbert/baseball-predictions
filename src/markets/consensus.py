"""Line-aligned consensus probabilities and best-price routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.contracts.domain import Quote, Selection
from src.markets.pricing import devig_ensemble, implied_probability


@dataclass(frozen=True)
class MarketConsensus:
    game_id: str
    market_id: str
    point: str | None
    probabilities: dict[Selection, float]
    book_count: int
    observed_at: datetime
    methods: tuple[str, ...]


def build_consensus(
    quotes: list[Quote],
    *,
    methods: tuple[str, ...] = ("multiplicative", "power", "shin"),
    min_books: int = 2,
) -> MarketConsensus:
    if not quotes:
        raise ValueError("No quotes supplied")
    keys = {(row.game_id, row.market_id, row.point) for row in quotes}
    if len(keys) != 1:
        raise ValueError("Consensus quotes must share game, market, and point")
    by_book: dict[str, dict[Selection, Quote]] = {}
    for quote in quotes:
        if quote.is_suspended:
            continue
        by_book.setdefault(quote.bookmaker_id, {})[quote.selection] = quote
    complete = {book: selections for book, selections in by_book.items() if len(selections) >= 2}
    if len(complete) < min_books:
        raise ValueError(f"Consensus requires at least {min_books} complete books")
    selections = sorted(
        set.intersection(*(set(rows) for rows in complete.values())), key=lambda value: value.value
    )
    estimates: list[list[float]] = []
    for rows in complete.values():
        raw = [
            implied_probability(float(rows[selection].price_decimal)) for selection in selections
        ]
        estimates.append(devig_ensemble(raw, methods))
    probabilities = {
        selection: sum(row[index] for row in estimates) / len(estimates)
        for index, selection in enumerate(selections)
    }
    game_id, market_id, point = next(iter(keys))
    return MarketConsensus(
        game_id=game_id,
        market_id=market_id,
        point=str(point) if point is not None else None,
        probabilities=probabilities,
        book_count=len(complete),
        observed_at=max(row.observed_at for row in quotes),
        methods=methods,
    )


def best_actionable_quote(
    quotes: list[Quote],
    *,
    selection: Selection,
    as_of: datetime,
    max_age_seconds: int,
    jurisdiction: str | None = None,
) -> Quote | None:
    eligible = [
        row
        for row in quotes
        if row.selection == selection
        and row.observed_at <= as_of
        and (as_of - row.observed_at).total_seconds() <= max_age_seconds
        and not row.is_suspended
        and row.is_actionable
        and (jurisdiction is None or row.jurisdiction in {None, jurisdiction})
    ]
    return max(eligible, key=lambda row: (row.price_decimal, row.observed_at), default=None)
