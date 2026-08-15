"""Deterministic execution simulation with latency, slippage, limits, and rejections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from src.contracts.domain import Decision, Execution, Quote, stable_id


@dataclass(frozen=True)
class ExecutionPolicy:
    version: str = "paper_execution_v1"
    latency_seconds: int = 2
    max_price_slippage_probability: float = 0.01
    allow_partial_fills: bool = True
    minimum_stake: Decimal = Decimal("1.00")
    available_bookmakers: tuple[str, ...] = ()


def simulate_execution(
    decision: Decision,
    decision_quote: Quote,
    later_quotes: list[Quote],
    *,
    policy: ExecutionPolicy,
) -> Execution:
    if decision.action != "bet" or decision.recommended_stake <= 0:
        raise ValueError("Only positive bet decisions can be executed")
    if decision.quote_id != decision_quote.quote_id:
        raise ValueError("Execution quote does not match decision")
    eligible_at = decision.decided_at + timedelta(seconds=policy.latency_seconds)
    candidates = [
        row
        for row in later_quotes
        if (row.game_id, row.market_id, row.selection, row.point, row.bookmaker_id)
        == (
            decision_quote.game_id,
            decision_quote.market_id,
            decision_quote.selection,
            decision_quote.point,
            decision_quote.bookmaker_id,
        )
        and row.observed_at >= eligible_at
        and not row.is_suspended
        and row.is_actionable
    ]
    quote = min(candidates, key=lambda row: row.observed_at, default=None)
    status = "rejected"
    rejection: str | None = None
    stake = decision.recommended_stake
    accepted_price = decision_quote.price_decimal
    if (
        policy.available_bookmakers
        and decision_quote.bookmaker_id not in policy.available_bookmakers
    ):
        rejection = "book_unavailable"
    elif quote is None:
        rejection = "quote_unavailable_after_latency"
    else:
        accepted_price = quote.price_decimal
        probability_slippage = float(1 / accepted_price - 1 / decision_quote.price_decimal)
        if probability_slippage > policy.max_price_slippage_probability:
            rejection = "slippage_exceeded"
        else:
            limit = quote.limit_amount
            if limit is not None and stake > limit:
                if not policy.allow_partial_fills or limit < policy.minimum_stake:
                    rejection = "limit_exceeded"
                else:
                    stake = limit
            if rejection is None and stake < policy.minimum_stake:
                rejection = "below_minimum_stake"
            if rejection is None:
                status = "accepted" if stake == decision.recommended_stake else "partially_filled"

    return Execution(
        execution_id=stable_id("execution", decision.decision_id, eligible_at),
        decision_id=decision.decision_id,
        quote_id=quote.quote_id if quote else decision_quote.quote_id,
        bookmaker_id=decision_quote.bookmaker_id,
        placed_at=eligible_at,
        accepted_price_decimal=accepted_price,
        accepted_point=decision_quote.point,
        stake=stake if status != "rejected" else decision.recommended_stake,
        status=status,
        rejection_reason=rejection,
    )
