"""First-five, NRFI/YRFI, and arbitrary-line pricing from generative scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from src.models.score_distribution import ScoreDistribution, shared_environment_score_distribution


@dataclass(frozen=True)
class MarketProbabilities:
    market_id: str
    home: float | None = None
    away: float | None = None
    over: float | None = None
    under: float | None = None
    yes: float | None = None
    no: float | None = None
    push: float = 0.0


def price_full_game(
    distribution: ScoreDistribution,
    *,
    run_line: float = -1.5,
    total_line: float = 8.5,
) -> dict[str, MarketProbabilities]:
    home, away, run_push = distribution.run_line_probabilities(run_line)
    over, under, total_push = distribution.total_probabilities(total_line)
    tie = distribution.tie_probability()
    return {
        "moneyline_full_game": MarketProbabilities(
            "moneyline_full_game",
            home=distribution.home_moneyline(),
            away=distribution.away_moneyline(),
            push=tie,
        ),
        "run_line_full_game": MarketProbabilities(
            "run_line_full_game",
            home=home,
            away=away,
            push=run_push,
        ),
        "total_full_game": MarketProbabilities(
            "total_full_game",
            over=over,
            under=under,
            push=total_push,
        ),
    }


def first_five_distribution(
    away_full_game_rate: float,
    home_full_game_rate: float,
    *,
    starter_share: float = 5 / 9,
    environment_sd: float = 0.12,
    seed: int = 42,
) -> ScoreDistribution:
    if not 0 < starter_share < 1:
        raise ValueError("starter_share must be in (0, 1)")
    return shared_environment_score_distribution(
        away_full_game_rate * starter_share,
        home_full_game_rate * starter_share,
        environment_sd=environment_sd,
        seed=seed,
    )


def price_first_five(
    distribution: ScoreDistribution,
    *,
    run_line: float = -0.5,
    total_line: float = 4.5,
) -> dict[str, MarketProbabilities]:
    priced = price_full_game(distribution, run_line=run_line, total_line=total_line)
    return {
        key.replace("full_game", "first_5"): MarketProbabilities(
            value.market_id.replace("full_game", "first_5"),
            home=value.home,
            away=value.away,
            over=value.over,
            under=value.under,
            yes=value.yes,
            no=value.no,
            push=value.push,
        )
        for key, value in priced.items()
    }


def nrfi_yrfi_probability(
    away_first_inning_rate: float,
    home_first_inning_rate: float,
    *,
    shared_environment_variance: float = 0.0,
) -> MarketProbabilities:
    if min(away_first_inning_rate, home_first_inning_rate) < 0:
        raise ValueError("Inning rates cannot be negative")
    # Gamma-Poisson mixing provides an overdispersed shared-environment correction.
    total_rate = away_first_inning_rate + home_first_inning_rate
    if shared_environment_variance > 0:
        shape = 1 / shared_environment_variance
        no = (shape / (shape + total_rate)) ** shape
    else:
        no = exp(-total_rate)
    return MarketProbabilities("nrfi_yrfi", yes=1 - no, no=no)
