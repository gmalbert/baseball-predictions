"""Tests for the game simulator and prop coherence."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.game_simulator import (
    LEAGUE_AVERAGE_MIX,
    FixedOutcomeModel,
    simulate_game,
    simulate_score_distribution,
)
from src.models.prop_coherence import (
    PropAllocation,
    prop_distributions_from_game,
)
from src.models.score_distribution import ScoreDistribution


def test_fixed_outcome_model_mix_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        FixedOutcomeModel({})


def test_simulated_game_produces_valid_runs() -> None:
    model = FixedOutcomeModel(LEAGUE_AVERAGE_MIX)
    rng = np.random.default_rng(7)
    game = simulate_game(rng=rng, model=model)
    assert game.away_runs >= 0
    assert game.home_runs >= 0
    assert game.away_pa > 0
    assert game.home_pa > 0


def test_score_distribution_is_coherent() -> None:
    model = FixedOutcomeModel(LEAGUE_AVERAGE_MIX)
    distribution = simulate_score_distribution(model=model, n_simulations=2_000, seed=1)
    assert isinstance(distribution, ScoreDistribution)
    assert abs(distribution.matrix.sum() - 1.0) < 1e-8
    home = distribution.home_moneyline()
    away = distribution.away_moneyline()
    tie = distribution.tie_probability()
    assert abs(home + away + tie - 1.0) < 1e-8
    # Neutral league-average mix: both sides near 0.5
    assert 0.35 < home < 0.65
    assert 0.35 < away < 0.65


def test_totals_reconcile_with_joint_distribution() -> None:
    model = FixedOutcomeModel(LEAGUE_AVERAGE_MIX)
    distribution = simulate_score_distribution(model=model, n_simulations=2_000, seed=2)
    over, under, push = distribution.total_probabilities(9.0)
    assert abs(over + under + push - 1.0) < 1e-8


def test_prop_allocations_share_game_distribution() -> None:
    allocations = prop_distributions_from_game(
        away_pa=40,
        home_pa=40,
        away_lineup={"a1": 4.5, "a2": 4.0},
        home_lineup={"h1": 4.5, "h2": 4.0},
        markets=("hits", "home_runs", "total_bases", "strikeouts"),
    )
    assert len(allocations) == 4 * 4  # 4 players x 4 markets
    assert all(isinstance(allocation, PropAllocation) for allocation in allocations)
    # PA volume proportional to weight
    a1 = next(a for a in allocations if a.player_id == "a1" and a.market == "hits")
    a2 = next(a for a in allocations if a.player_id == "a2" and a.market == "hits")
    assert a1.distribution.probabilities.sum() == pytest.approx(1.0)
    # a1 has more weight than a2 -> more expected PA -> wider distribution
    assert a1.distribution.probabilities[0] < a2.distribution.probabilities[0]


def test_prop_pa_totals_match_team_pa() -> None:
    allocations = prop_distributions_from_game(
        away_pa=40,
        home_pa=40,
        away_lineup={"a1": 4.5, "a2": 4.0},
        home_lineup={"h1": 4.5, "h2": 4.0},
        markets=("hits",),
    )
    away_hits = [a for a in allocations if a.team_id == "away"]
    # Expected hits per player = mean of the Poisson = PA volume * event rate.
    total_expected_hits = sum(
        float(np.arange(len(a.distribution.probabilities)) @ a.distribution.probabilities)
        for a in away_hits
    )
    # With ~40 team PA and a ~0.213 hit rate, expected hits ≈ 8.5.
    assert 0 < total_expected_hits < 20
    # The higher-weighted player gets more PA volume -> higher expected hits.
    a1 = next(a for a in away_hits if a.player_id == "a1")
    a2 = next(a for a in away_hits if a.player_id == "a2")
    mean_a1 = float(np.arange(len(a1.distribution.probabilities)) @ a1.distribution.probabilities)
    mean_a2 = float(np.arange(len(a2.distribution.probabilities)) @ a2.distribution.probabilities)
    assert mean_a1 > mean_a2
