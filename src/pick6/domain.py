"""Pick 6 rules and immutable ticket representation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Pick6Leg:
    leg_id: str
    game_id: str
    player_id: str
    market: str
    selection: str
    line: Decimal
    marginal_probability: float
    driver_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.marginal_probability <= 1:
            raise ValueError("Leg probability must be in [0, 1]")


@dataclass(frozen=True)
class Pick6Ticket:
    ticket_id: str
    legs: tuple[Pick6Leg, ...]
    payout_multiple: Decimal
    stake: Decimal

    def __post_init__(self) -> None:
        if len(self.legs) != 6:
            raise ValueError("Pick 6 tickets require exactly six legs")
        if len({row.leg_id for row in self.legs}) != 6:
            raise ValueError("Pick 6 legs must be unique")
        if self.payout_multiple <= 1 or self.stake <= 0:
            raise ValueError("Payout and stake must be positive")

    @property
    def break_even_probability(self) -> float:
        return 1 / float(self.payout_multiple)

    @property
    def dependency_warning(self) -> bool:
        games = [row.game_id for row in self.legs]
        drivers = [driver for row in self.legs for driver in row.driver_ids]
        return len(games) != len(set(games)) or len(drivers) != len(set(drivers))
