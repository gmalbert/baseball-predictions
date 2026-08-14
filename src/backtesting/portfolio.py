"""Conservative correlated exposure allocation across game/team/market/book factors."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.contracts.domain import Decision, Quote


@dataclass(frozen=True)
class PortfolioPolicy:
    max_bet_fraction: Decimal = Decimal("0.010")
    max_game_fraction: Decimal = Decimal("0.020")
    max_team_fraction: Decimal = Decimal("0.025")
    max_market_fraction: Decimal = Decimal("0.035")
    max_book_fraction: Decimal = Decimal("0.040")
    max_factor_fraction: Decimal = Decimal("0.025")
    max_day_fraction: Decimal = Decimal("0.050")
    stop_loss_fraction: Decimal = Decimal("0.035")
    max_open_bets: int = 20


@dataclass(frozen=True)
class CandidateExposure:
    decision: Decision
    quote: Quote
    team_ids: tuple[str, ...] = ()
    factor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Allocation:
    decision_id: str
    requested_stake: Decimal
    allocated_stake: Decimal
    binding_constraint: str | None


def allocate_portfolio(
    candidates: list[CandidateExposure],
    *,
    bankroll: Decimal,
    policy: PortfolioPolicy,
    realized_day_loss: Decimal = Decimal("0"),
) -> list[Allocation]:
    if bankroll <= 0:
        raise ValueError("Bankroll must be positive")
    if -realized_day_loss >= bankroll * policy.stop_loss_fraction:
        return [
            Allocation(
                row.decision.decision_id, row.decision.recommended_stake, Decimal("0"), "stop_loss"
            )
            for row in candidates
        ]
    ordered = sorted(candidates, key=lambda row: row.decision.expected_value, reverse=True)
    totals: dict[tuple[str, str], Decimal] = {}
    day_total = Decimal("0")
    allocations = []
    for index, candidate in enumerate(ordered):
        requested = candidate.decision.recommended_stake
        if index >= policy.max_open_bets:
            allocations.append(
                Allocation(candidate.decision.decision_id, requested, Decimal("0"), "max_open_bets")
            )
            continue
        caps = [
            (
                "bet",
                bankroll * policy.max_bet_fraction
                - totals.get(("bet", candidate.decision.decision_id), Decimal("0")),
            ),
            (
                "game",
                bankroll * policy.max_game_fraction
                - totals.get(("game", candidate.decision.game_id), Decimal("0")),
            ),
            (
                "market",
                bankroll * policy.max_market_fraction
                - totals.get(("market", candidate.decision.market_id), Decimal("0")),
            ),
            (
                "book",
                bankroll * policy.max_book_fraction
                - totals.get(("book", candidate.quote.bookmaker_id), Decimal("0")),
            ),
            ("day", bankroll * policy.max_day_fraction - day_total),
        ]
        caps.extend(
            ("team", bankroll * policy.max_team_fraction - totals.get(("team", team), Decimal("0")))
            for team in candidate.team_ids
        )
        caps.extend(
            (
                "correlation_factor",
                bankroll * policy.max_factor_fraction
                - totals.get(("factor", factor), Decimal("0")),
            )
            for factor in candidate.factor_ids
        )
        allowed_name, allowed = min(caps, key=lambda item: item[1])
        stake = max(Decimal("0"), min(requested, allowed)).quantize(Decimal("0.01"))
        binding = allowed_name if stake < requested else None
        allocations.append(Allocation(candidate.decision.decision_id, requested, stake, binding))
        day_total += stake
        for key in (
            ("bet", candidate.decision.decision_id),
            ("game", candidate.decision.game_id),
            ("market", candidate.decision.market_id),
            ("book", candidate.quote.bookmaker_id),
        ):
            totals[key] = totals.get(key, Decimal("0")) + stake
        for team in candidate.team_ids:
            totals[("team", team)] = totals.get(("team", team), Decimal("0")) + stake
        for factor in candidate.factor_ids:
            totals[("factor", factor)] = totals.get(("factor", factor), Decimal("0")) + stake
    return allocations
