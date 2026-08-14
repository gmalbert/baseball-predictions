"""Game simulator: PA-level outcome probabilities to one coherent score
distribution for moneyline, run line, totals, team totals, and props.

The simulator walks the ``PaState`` machine for both halves of each inning,
drawing each PA's outcome from a per-PA outcome probability vector (from the
trained per-PA model, or a fixed league-average mix).  Existing runners advance
stochastically via a ``RunnerAdvancement`` model.  The result is a
``ScoreDistribution`` whose moneyline/run-line/total probabilities are derived
from the same joint distribution, so all markets are coherent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.models.plate_appearance import (
    MAX_RUNS,
    PaOutcome,
    PaState,
    RunnerAdvancement,
    SimpleRunnerAdvancement,
    advance,
)
from src.models.score_distribution import ScoreDistribution

_OUTCOMES = list(PaOutcome)

# League-average outcome mix (per-PA probabilities), used as the default when
# no model probabilities are supplied.
LEAGUE_AVERAGE_MIX = {
    PaOutcome.OUT: 0.624762,
    PaOutcome.STRIKEOUT: 0.071429,
    PaOutcome.WALK: 0.080952,
    PaOutcome.HBP: 0.009524,
    PaOutcome.SINGLE: 0.140952,
    PaOutcome.DOUBLE: 0.040000,
    PaOutcome.TRIPLE: 0.003810,
    PaOutcome.HOME_RUN: 0.028571,
}


class OutcomeModel(Protocol):
    """Per-PA outcome probability provider."""

    def outcome_probabilities(self, batter_id: str, pitcher_id: str, state: PaState) -> np.ndarray:
        """Return a length-8 probability vector aligned to ``_OUTCOMES``."""
        ...


class FixedOutcomeModel:
    """Constant outcome mix, used for tests and as a baseline."""

    def __init__(self, mix: dict[PaOutcome, float] | None = None) -> None:
        selected = mix or {}
        self.mix = np.array([selected.get(outcome, 0.0) for outcome in _OUTCOMES], dtype=float)
        total = float(self.mix.sum())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Outcome mix must sum to one, got {total}")

    def outcome_probabilities(self, batter_id: str, pitcher_id: str, state: PaState) -> np.ndarray:
        return self.mix


@dataclass(frozen=True)
class SimulatedGame:
    away_runs: int
    home_runs: int
    away_pa: int
    home_pa: int


def _sample_outcome(rng: np.random.Generator, probabilities: np.ndarray) -> PaOutcome:
    return _OUTCOMES[int(rng.choice(len(_OUTCOMES), p=probabilities))]


def _half_inning(
    *,
    rng: np.random.Generator,
    model: OutcomeModel,
    runner_advancement: RunnerAdvancement,
    max_outs: int = 3,
) -> tuple[int, int]:
    """Simulate one half-inning; returns (runs, plate appearances)."""
    state = PaState()
    runs = 0
    pa = 0
    while state.outs < max_outs:
        probabilities = model.outcome_probabilities("batter", "pitcher", state)
        outcome = _sample_outcome(rng, probabilities)
        distribution = advance(state, outcome, runner_advancement)
        # Sample the next state from the (stochastic) advancement distribution.
        states = list(distribution)
        weights = np.array([distribution[s] for s in states], dtype=float)
        next_state = states[int(rng.choice(len(states), p=weights))]
        # Runs scored on the play = the increment the state machine recorded.
        runs += next_state.score_diff
        pa += 1
        if next_state.outs >= max_outs:
            break
        state = PaState(
            outs=next_state.outs,
            on_1b=next_state.on_1b,
            on_2b=next_state.on_2b,
            on_3b=next_state.on_3b,
            inning=state.inning,
            score_diff=0,
        )
    return runs, pa


def simulate_game(
    *,
    rng: np.random.Generator,
    model: OutcomeModel,
    runner_advancement: RunnerAdvancement | None = None,
    max_innings: int = 9,
) -> SimulatedGame:
    """Simulate one game; walk-off and extra innings use the fixed 9-inning
    grain and extra frames are drawn the same way (ties allowed in the raw
    simulation; the distribution handles settlement)."""
    advancement = runner_advancement or SimpleRunnerAdvancement()
    away_runs = 0
    home_runs = 0
    away_pa = 0
    home_pa = 0
    for inning in range(1, max_innings + 1):
        away_runs_inning, away_pa_inning = _half_inning(
            rng=rng, model=model, runner_advancement=advancement
        )
        away_runs += away_runs_inning
        away_pa += away_pa_inning
        home_runs_inning, home_pa_inning = _half_inning(
            rng=rng, model=model, runner_advancement=advancement
        )
        home_runs += home_runs_inning
        home_pa += home_pa_inning
    return SimulatedGame(
        away_runs=away_runs,
        home_runs=home_runs,
        away_pa=away_pa,
        home_pa=home_pa,
    )


def simulate_score_distribution(
    *,
    model: OutcomeModel,
    n_simulations: int = 20_000,
    runner_advancement: RunnerAdvancement | None = None,
    seed: int = 42,
    max_runs: int = MAX_RUNS,
) -> ScoreDistribution:
    """Monte Carlo the full game and return a coherent ``ScoreDistribution``.

    The matrix is clipped to ``max_runs`` per side; the tail probability is
    folded into the final cell so the matrix still sums to one.
    """
    rng = np.random.default_rng(seed)
    matrix = np.zeros((max_runs, max_runs), dtype=float)
    for _ in range(n_simulations):
        game = simulate_game(rng=rng, model=model, runner_advancement=runner_advancement)
        away = min(game.away_runs, max_runs - 1)
        home = min(game.home_runs, max_runs - 1)
        matrix[away, home] += 1.0
    matrix /= matrix.sum()
    return ScoreDistribution(matrix)


def outcome_mix_from_frame(frame: np.ndarray) -> dict[PaOutcome, float]:
    """Convert a model probability vector (or empirical count vector) to an
    outcome mix dict aligned to ``_OUTCOMES``."""
    values = np.asarray(frame, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError("Outcome mix cannot be empty")
    return {outcome: float(values[i] / total) for i, outcome in enumerate(_OUTCOMES)}
