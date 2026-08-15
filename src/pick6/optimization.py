"""Ticket construction under uniqueness, dependency, and stake constraints."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

from src.pick6.domain import Pick6Leg, Pick6Ticket


@dataclass(frozen=True)
class TicketPolicy:
    max_same_game_legs: int = 2
    max_ticket_fraction: Decimal = Decimal("0.0025")
    minimum_leg_probability: float = 0.50


def construct_ticket(
    candidates: list[Pick6Leg],
    *,
    payout_multiple: Decimal,
    bankroll: Decimal,
    policy: TicketPolicy = TicketPolicy(),
) -> Pick6Ticket:
    eligible = [
        row for row in candidates if row.marginal_probability >= policy.minimum_leg_probability
    ]
    best: tuple[Pick6Leg, ...] | None = None
    best_score = -1.0
    for legs in combinations(eligible, 6):
        game_counts = {
            game: sum(row.game_id == game for row in legs) for game in {row.game_id for row in legs}
        }
        if max(game_counts.values()) > policy.max_same_game_legs:
            continue
        dependency_penalty = sum(
            1
            for left, right in combinations(legs, 2)
            if set(left.driver_ids) & set(right.driver_ids)
        )
        score = sum(row.marginal_probability for row in legs) - 0.05 * dependency_penalty
        if score > best_score:
            best, best_score = legs, score
    if best is None:
        raise ValueError("No six-leg ticket satisfies policy")
    from src.contracts.domain import stable_id

    return Pick6Ticket(
        ticket_id=stable_id("pick6", *(row.leg_id for row in best)),
        legs=best,
        payout_multiple=payout_multiple,
        stake=(bankroll * policy.max_ticket_fraction).quantize(Decimal("0.01")),
    )
