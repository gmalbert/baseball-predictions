"""Selection-matched opportunity policy with uncertainty-aware Kelly sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.contracts.domain import Decision, Prediction, Quote, stable_id
from src.markets.pricing import expected_value, implied_probability, kelly_fraction


@dataclass(frozen=True)
class Policy:
    version: str = "conservative_v1"
    min_ev: float = 0.01
    min_edge: float = 0.01
    kelly_multiplier: float = 0.25
    uncertainty_haircut: float = 0.75
    max_bet_fraction: float = 0.01
    max_quote_age_seconds: int = 180
    require_actionable: bool = True
    allowed_jurisdictions: tuple[str, ...] = ()


def evaluate(
    prediction: Prediction,
    quote: Quote,
    *,
    bankroll: Decimal,
    decided_at: datetime,
    policy: Policy,
) -> Decision:
    if (prediction.game_id, prediction.market_id, prediction.selection) != (
        quote.game_id,
        quote.market_id,
        quote.selection,
    ):
        raise ValueError("Prediction and quote selection do not match")
    if decided_at.tzinfo is None:
        raise ValueError("decided_at must be timezone-aware")

    reasons: list[str] = []
    age = (decided_at - quote.observed_at).total_seconds()
    if age < 0:
        reasons.append("quote_from_future")
    if age > policy.max_quote_age_seconds:
        reasons.append("stale_quote")
    if quote.is_suspended:
        reasons.append("suspended_quote")
    if policy.require_actionable and not quote.is_actionable:
        reasons.append("non_actionable_quote")
    if policy.allowed_jurisdictions and quote.jurisdiction not in policy.allowed_jurisdictions:
        reasons.append("jurisdiction_unavailable")

    decimal_price = float(quote.price_decimal)
    break_even = implied_probability(decimal_price)
    edge = prediction.probability - break_even
    ev = expected_value(prediction.probability, decimal_price)
    if edge < policy.min_edge:
        reasons.append("edge_below_minimum")
    if ev < policy.min_ev:
        reasons.append("ev_below_minimum")
    if prediction.probability_low is not None and prediction.probability_low <= break_even:
        reasons.append("uncertainty_spans_break_even")

    stake = Decimal("0")
    if not reasons:
        base_fraction = kelly_fraction(prediction.probability, decimal_price)
        if prediction.probability_low is not None:
            low_fraction = kelly_fraction(prediction.probability_low, decimal_price)
            base_fraction = min(base_fraction, low_fraction / max(policy.uncertainty_haircut, 1e-9))
        fraction = min(policy.max_bet_fraction, policy.kelly_multiplier * base_fraction)
        stake = (bankroll * Decimal(str(fraction))).quantize(Decimal("0.01"))
        if quote.limit_amount is not None:
            stake = min(stake, quote.limit_amount)

    return Decision(
        decision_id=stable_id("decision", prediction.prediction_id, quote.quote_id, decided_at),
        prediction_id=prediction.prediction_id,
        quote_id=quote.quote_id,
        game_id=prediction.game_id,
        market_id=prediction.market_id,
        selection=prediction.selection,
        decided_at=decided_at,
        market_probability=break_even,
        fair_probability=prediction.probability,
        break_even_probability=break_even,
        edge=edge,
        expected_value=ev,
        recommended_stake=stake,
        bankroll_before=bankroll,
        policy_version=policy.version,
        action="bet" if stake > 0 else "abstain",
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
