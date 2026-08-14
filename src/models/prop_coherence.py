"""Player prop coherence: player PA volume and event rates share the game
distribution produced by the simulator.

``prop_distributions_from_game`` allocates each team's simulated plate
appearances across the projected lineup by weight, then derives per-player
event rates (hits, home runs, total bases, strikeouts as a batter) from the
same per-PA outcome mix used to simulate the game.  This keeps props on the
same distribution as the moneyline/run-line/totals rather than independent
Poisson guesses.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.game_simulator import _OUTCOMES, LEAGUE_AVERAGE_MIX
from src.models.plate_appearance import OUTCOME_BASES, PaOutcome
from src.models.props import SUPPORTED_PROPS, PropDistribution, poisson_prop_distribution


@dataclass(frozen=True)
class PropAllocation:
    player_id: str
    team_id: str
    market: str
    distribution: PropDistribution


def _event_rate_for_market(outcome_probabilities: dict[PaOutcome, float], market: str) -> float:
    """Event rate per PA for a supported prop market from the outcome mix."""
    if market == "hits":
        return sum(
            outcome_probabilities.get(outcome, 0.0)
            for outcome in (
                PaOutcome.SINGLE,
                PaOutcome.DOUBLE,
                PaOutcome.TRIPLE,
                PaOutcome.HOME_RUN,
            )
        )
    if market == "home_runs":
        return outcome_probabilities.get(PaOutcome.HOME_RUN, 0.0)
    if market == "total_bases":
        return sum(
            OUTCOME_BASES[outcome][1] * outcome_probabilities.get(outcome, 0.0)
            for outcome in (
                PaOutcome.SINGLE,
                PaOutcome.DOUBLE,
                PaOutcome.TRIPLE,
                PaOutcome.HOME_RUN,
            )
        )
    if market == "strikeouts":
        return outcome_probabilities.get(PaOutcome.STRIKEOUT, 0.0)
    if market == "outs_recorded":
        return outcome_probabilities.get(PaOutcome.OUT, 0.0) + outcome_probabilities.get(
            PaOutcome.STRIKEOUT, 0.0
        )
    raise KeyError(f"Unsupported prop market: {market}")


def prop_distributions_from_game(
    *,
    away_pa: int,
    home_pa: int,
    away_lineup: dict[str, float],
    home_lineup: dict[str, float],
    away_outcome_mix: dict[PaOutcome, float] | None = None,
    home_outcome_mix: dict[PaOutcome, float] | None = None,
    markets: tuple[str, ...] = ("hits", "home_runs", "total_bases", "strikeouts"),
    availability_probability: float = 1.0,
    maximum: int = 20,
) -> list[PropAllocation]:
    """Allocate simulated PA totals across projected lineups and derive props.

    ``away_lineup``/``home_lineup`` map player_id to projected PA weight.
    Per-player PA volume is weight / total_weight * team PA total; per-player
    event rates come from the team outcome mix (default league average).
    """
    if not markets:
        return []
    unsupported = set(markets) - SUPPORTED_PROPS
    if unsupported:
        raise KeyError(f"Unsupported prop markets: {sorted(unsupported)}")

    allocations: list[PropAllocation] = []
    for team_id, team_pa, lineup, mix in (
        ("away", away_pa, away_lineup, away_outcome_mix or LEAGUE_AVERAGE_MIX),
        ("home", home_pa, home_lineup, home_outcome_mix or LEAGUE_AVERAGE_MIX),
    ):
        total_weight = sum(lineup.values())
        if total_weight <= 0:
            continue
        for player_id, weight in lineup.items():
            expected_pa = weight / total_weight * team_pa
            for market in markets:
                event_rate = _event_rate_for_market(mix, market)
                distribution = poisson_prop_distribution(
                    market,
                    expected_opportunities=expected_pa,
                    event_rate=event_rate,
                    availability_probability=availability_probability,
                    maximum=maximum,
                )
                allocations.append(
                    PropAllocation(
                        player_id=player_id,
                        team_id=team_id,
                        market=market,
                        distribution=distribution,
                    )
                )
    return allocations


def outcome_mix_from_rates(
    *,
    hits_rate: float,
    hr_rate: float,
    walk_rate: float,
    strikeout_rate: float,
) -> dict[PaOutcome, float]:
    """Build a per-PA outcome mix from observed rates (used to fit a model's
    output to the simulator's fixed outcome vector)."""
    mix = dict(LEAGUE_AVERAGE_MIX)
    mix[PaOutcome.SINGLE] = max(hits_rate - hr_rate - 0.042 - 0.004, 0.0)
    mix[PaOutcome.DOUBLE] = 0.042
    mix[PaOutcome.TRIPLE] = 0.004
    mix[PaOutcome.HOME_RUN] = hr_rate
    mix[PaOutcome.WALK] = walk_rate
    mix[PaOutcome.STRIKEOUT] = strikeout_rate
    mix[PaOutcome.OUT] = 1.0 - sum(mix.get(o, 0.0) for o in _OUTCOMES if o != PaOutcome.OUT)
    return mix
