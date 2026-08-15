"""Generative plate-appearance state machine.

``PaState`` is the blueprint's ``(outs, on_1b, on_2b, on_3b, inning, score_diff)``
state.  ``advance`` takes a PA outcome plus a runner-advancement model and
returns a probability distribution over next states; advancement is stochastic
because runners do not always take the same number of bases on a hit.

Outcomes are the mutually exclusive terminal result of a plate appearance
(excluding the count path, which the per-PA model collapses to an outcome).
A batter either reaches base or records an out; reaching outcomes carry the
base(s) awarded and an optional advancement probability distribution for the
existing runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

MAX_RUNS = 21  # Consistent with src.models.score_distribution.


class PaOutcome(StrEnum):
    OUT = "out"
    STRIKEOUT = "strikeout"
    WALK = "walk"
    HBP = "hit_by_pitch"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"


# Base effects per outcome: (runs scored by batter, bases awarded to batter,
# bases awarded to existing runners as a default advancement).
OUTCOME_BASES: dict[PaOutcome, tuple[int, int, int]] = {
    PaOutcome.OUT: (0, 0, 0),
    PaOutcome.STRIKEOUT: (0, 0, 0),
    PaOutcome.WALK: (0, 1, 1),
    PaOutcome.HBP: (0, 1, 1),
    PaOutcome.SINGLE: (0, 1, 2),
    PaOutcome.DOUBLE: (0, 2, 2),
    PaOutcome.TRIPLE: (0, 3, 3),
    PaOutcome.HOME_RUN: (1, 4, 4),
}

# Innings that can contain a PA (regulation 9, plus extras for the model grain).
INNINGS = tuple(range(1, 10))


@dataclass(frozen=True)
class PaState:
    """Base/out/inning/differential state before a plate appearance."""

    outs: int = 0
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False
    inning: int = 1
    score_diff: int = 0  # home - away; away-team PA sees the negative.

    def __post_init__(self) -> None:
        if not 0 <= self.outs <= 3:
            raise ValueError(f"outs must be 0..3, got {self.outs}")
        if self.inning < 1:
            raise ValueError(f"inning must be >= 1, got {self.inning}")

    @property
    def runners(self) -> int:
        return int(self.on_1b) + int(self.on_2b) + int(self.on_3b)

    def with_runner_advance(self, bases: int) -> "PaState":
        """Return the state after all runners advance ``bases`` with no force.

        A runner on first reaches third on a two-base advance, scores on three,
        etc.  Runners already on base never move backward.
        """
        on_1b = self.on_1b and bases < 1
        on_2b = (self.on_1b and bases >= 1 and bases < 2) or (self.on_2b and bases < 2)
        on_3b = (
            (self.on_1b and bases >= 2 and bases < 3)
            or (self.on_2b and bases >= 2 and bases < 3)
            or (self.on_3b and bases < 3)
        )
        return PaState(
            outs=self.outs,
            on_1b=on_1b,
            on_2b=on_2b,
            on_3b=on_3b,
            inning=self.inning,
            score_diff=self.score_diff,
        )

    def with_out(self) -> "PaState":
        return PaState(
            outs=self.outs + 1,
            on_1b=self.on_1b,
            on_2b=self.on_2b,
            on_3b=self.on_3b,
            inning=self.inning,
            score_diff=self.score_diff,
        )


def _runners_scored(state: PaState, bases: int) -> int:
    scored = 0
    if state.on_1b and bases >= 3:
        scored += 1
    if state.on_2b and bases >= 2:
        scored += 1
    if state.on_3b and bases >= 1:
        scored += 1
    return scored


def advance(
    state: PaState,
    outcome: PaOutcome,
    runner_model: "RunnerAdvancement | None" = None,
) -> dict[PaState, float]:
    """Return a probability distribution over next states for ``outcome``.

    ``runner_model`` supplies per-outcome advancement distributions for the
    runners already on base.  If omitted, deterministic default advancement
    (``OUTCOME_BASES`` third element) is used.  When a state is terminal (third
    out), it is returned as-is: the inning-half boundary is the simulator's job.
    """
    if outcome in (PaOutcome.OUT, PaOutcome.STRIKEOUT):
        return {state.with_out(): 1.0}

    _, batter_bases, default_advance = OUTCOME_BASES[outcome]

    # Walks/HBP force runners; hits do not.
    if outcome in (PaOutcome.WALK, PaOutcome.HBP):
        forced = state.on_1b
        advanced = state.with_runner_advance(1)
        advanced = PaState(
            outs=advanced.outs,
            on_1b=True,
            on_2b=advanced.on_2b or (state.on_1b and forced),
            on_3b=advanced.on_3b or (state.on_2b and state.on_1b and forced),
            inning=advanced.inning,
            score_diff=advanced.score_diff,
        )
        return {advanced: 1.0}

    advancement = runner_model.advancement(state, outcome) if runner_model else None
    if advancement is None:
        next_state = state.with_runner_advance(default_advance)
        next_state = PaState(
            outs=next_state.outs,
            on_1b=next_state.on_1b or (batter_bases == 1),
            on_2b=next_state.on_2b or (batter_bases == 2),
            on_3b=next_state.on_3b or (batter_bases == 3),
            inning=next_state.inning,
            score_diff=next_state.score_diff
            + _runners_scored(state, default_advance)
            + int(batter_bases >= 4),
        )
        return {next_state: 1.0}

    distribution: dict[PaState, float] = {}
    for bases, probability in advancement.items():
        next_state = state.with_runner_advance(bases)
        next_state = PaState(
            outs=next_state.outs,
            on_1b=next_state.on_1b or (batter_bases == 1),
            on_2b=next_state.on_2b or (batter_bases == 2),
            on_3b=next_state.on_3b or (batter_bases == 3),
            inning=next_state.inning,
            score_diff=next_state.score_diff
            + _runners_scored(state, bases)
            + int(batter_bases >= 4),
        )
        distribution[next_state] = distribution.get(next_state, 0.0) + probability
    return distribution


class RunnerAdvancement:
    """Stochastic runner advancement on hits, per outcome and base state.

    ``advancement(state, outcome)`` returns {bases: probability} for how far
    existing runners advance.  The distribution must sum to one.
    """

    def advancement(self, state: PaState, outcome: PaOutcome) -> dict[int, float] | None:
        raise NotImplementedError


class SimpleRunnerAdvancement(RunnerAdvancement):
    """League-average advancement: most runners take default bases; a fraction
    are held or take an extra base."""

    def advancement(self, state: PaState, outcome: PaOutcome) -> dict[int, float]:
        default = OUTCOME_BASES[outcome][2]
        if default == 0:
            return {0: 1.0}
        distribution: dict[int, float] = {default: 0.80}
        held = max(default - 1, 0)
        if held != default:
            distribution[held] = distribution.get(held, 0.0) + 0.15
        extra = default + 1
        if extra <= 4 and extra != default:
            distribution[extra] = distribution.get(extra, 0.0) + 0.05
        elif extra > 4:
            # No extra base beyond home; fold the extra-base probability into
            # the default (every runner scores on a home run regardless).
            distribution[default] = distribution.get(default, 0.0) + 0.05
        return distribution


def base_out_states() -> list[PaState]:
    """Enumerate the 24 base/out states plus inning/differential defaults."""
    states: list[PaState] = []
    for outs in range(3):
        for on_1b in (False, True):
            for on_2b in (False, True):
                for on_3b in (False, True):
                    states.append(PaState(outs=outs, on_1b=on_1b, on_2b=on_2b, on_3b=on_3b))
    return states


def transition_matrix(
    outcome_probabilities: "dict[PaOutcome, float]",
    runner_model: RunnerAdvancement | None = None,
) -> np.ndarray:
    """Build the 24x24 base/out transition matrix for a fixed outcome mix.

    Rows are current base/out state, columns are next base/out state.  The
    ``score_diff`` and ``inning`` dimensions are left to the simulator, which
    carries them explicitly; the matrix only moves base/out.
    """
    states = base_out_states()
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for state in states:
        for outcome, probability in outcome_probabilities.items():
            if probability <= 0:
                continue
            for next_state, weight in advance(state, outcome, runner_model).items():
                # The matrix only tracks base/out; runs and inning are carried
                # separately by the simulator.  A terminal (3-out) state maps to
                # the 2-out, bases-empty row: no further PA occurs in that half.
                if next_state.outs >= 3:
                    terminal = PaState(outs=2)
                    matrix[index[state], index[terminal]] += probability * weight
                    continue
                base_out = PaState(
                    outs=next_state.outs,
                    on_1b=next_state.on_1b,
                    on_2b=next_state.on_2b,
                    on_3b=next_state.on_3b,
                )
                matrix[index[state], index[base_out]] += probability * weight
    for row in matrix:
        total = row.sum()
        if total > 0:
            row /= total
    return matrix
