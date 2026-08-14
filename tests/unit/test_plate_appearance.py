"""Tests for the generative plate-appearance state machine."""

from __future__ import annotations

import pytest

from src.models.plate_appearance import (
    PaOutcome,
    PaState,
    SimpleRunnerAdvancement,
    advance,
    base_out_states,
    transition_matrix,
)


def test_advance_single_empty_puts_batter_on_first() -> None:
    state = PaState()
    result = advance(state, PaOutcome.SINGLE)
    assert len(result) == 1
    ((next_state, weight),) = result.items()
    assert weight == 1.0
    assert next_state.on_1b is True
    assert next_state.on_2b is False
    assert next_state.on_3b is False
    assert next_state.outs == 0


def test_advance_out_increments_outs_and_third_out_is_terminal() -> None:
    state = PaState(outs=2, on_1b=True)
    result = advance(state, PaOutcome.STRIKEOUT)
    ((next_state, weight),) = result.items()
    assert weight == 1.0
    assert next_state.outs == 3  # half-inning terminal sentinel


def test_advance_walk_forces_runners() -> None:
    state = PaState(on_1b=True, on_2b=True)
    result = advance(state, PaOutcome.WALK)
    ((next_state, weight),) = result.items()
    assert weight == 1.0
    assert next_state.on_1b is True  # batter
    assert next_state.on_2b is True  # forced from first
    assert next_state.on_3b is True  # forced from second


def test_advance_home_run_scores_all_runners() -> None:
    state = PaState(on_1b=True, on_2b=True, on_3b=True)
    result = advance(state, PaOutcome.HOME_RUN)
    ((next_state, weight),) = result.items()
    assert weight == 1.0
    assert next_state.on_1b is False
    assert next_state.on_2b is False
    assert next_state.on_3b is False
    assert next_state.score_diff == 4  # grand slam


def test_advance_double_with_runner_on_first() -> None:
    state = PaState(on_1b=True)
    result = advance(state, PaOutcome.DOUBLE)
    ((next_state, weight),) = result.items()
    assert weight == 1.0
    # Runner on first scores on a double? No: 1B->2B->3B->home is 3 bases,
    # but default advancement for a double is 2 bases -> runner to third.
    assert next_state.on_3b is True
    assert next_state.on_2b is True  # batter
    assert next_state.score_diff == 0


def test_advance_distribution_normalizes_for_all_states_and_outcomes() -> None:
    runner_model = SimpleRunnerAdvancement()
    for state in base_out_states():
        for outcome in PaOutcome:
            result = advance(state, outcome, runner_model)
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-9, f"{state} {outcome} sums to {total}"


def test_transition_matrix_rows_sum_to_one() -> None:
    matrix = transition_matrix(
        {
            PaOutcome.SINGLE: 0.2,
            PaOutcome.DOUBLE: 0.05,
            PaOutcome.HOME_RUN: 0.03,
            PaOutcome.WALK: 0.10,
            PaOutcome.STRIKEOUT: 0.12,
            PaOutcome.OUT: 0.50,
        },
        SimpleRunnerAdvancement(),
    )
    assert matrix.shape == (24, 24)
    for row in matrix:
        assert abs(row.sum() - 1.0) < 1e-9


def test_base_out_states_has_24_states() -> None:
    assert len(base_out_states()) == 24


def test_pa_state_invariant() -> None:
    with pytest.raises(ValueError):
        PaState(outs=4)
    with pytest.raises(ValueError):
        PaState(inning=0)
