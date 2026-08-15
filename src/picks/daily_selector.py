"""Selection-level candidate construction; opposite sides cannot be mixed."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from src.contracts.domain import Decision, Prediction, Quote
from src.decisions.policy import Policy, evaluate


def select_opportunities(
    predictions: list[Prediction],
    quotes: list[Quote],
    *,
    bankroll: Decimal,
    decided_at: datetime,
    policy: Policy,
) -> tuple[list[Decision], list[tuple[Prediction, str]]]:
    latest: dict[tuple[str, str, object], Quote] = {}
    for quote in sorted(quotes, key=lambda row: row.observed_at):
        if quote.observed_at <= decided_at:
            latest[(quote.game_id, quote.market_id, quote.selection)] = quote
    decisions = []
    excluded = []
    for prediction in predictions:
        quote = latest.get((prediction.game_id, prediction.market_id, prediction.selection))
        if quote is None:
            excluded.append((prediction, "missing_quote"))
            continue
        decisions.append(
            evaluate(
                prediction,
                quote,
                bankroll=bankroll,
                decided_at=decided_at,
                policy=policy,
            )
        )
    return decisions, excluded
