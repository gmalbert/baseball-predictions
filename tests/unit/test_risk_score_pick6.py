from decimal import Decimal

import numpy as np
import pytest

from src.backtesting.risk import block_bootstrap_risk, required_stress_scenarios
from src.models.score_distribution import independent_poisson_score_distribution
from src.pick6.domain import Pick6Leg
from src.pick6.optimization import construct_ticket
from src.pick6.simulation import simulate_ticket


def test_score_distribution_prices_total_push_and_sums_to_one():
    distribution = independent_poisson_score_distribution(4.2, 4.5)
    over, under, push = distribution.total_probabilities(9)
    assert over + under + push == pytest.approx(1)
    assert (
        distribution.home_moneyline()
        + distribution.away_moneyline()
        + distribution.tie_probability()
        == pytest.approx(1)
    )


def test_risk_bootstrap_and_required_stress_scenarios_are_reproducible():
    returns = np.array([0.01, -0.012, 0.008, -0.005, 0.014, -0.01, 0.003] * 3)
    assert block_bootstrap_risk(returns, simulations=100, seed=7) == block_bootstrap_risk(
        returns, simulations=100, seed=7
    )
    names = {row.scenario for row in required_stress_scenarios(returns)}
    assert {"base", "20c_worse_fills", "source_outage_week", "regime_shift"} <= names


def test_pick6_uses_joint_simulation_and_enforces_ticket_cap():
    legs = [
        Pick6Leg(
            f"l{i}",
            f"g{i // 2}",
            f"p{i}",
            "hits",
            "over",
            Decimal("0.5"),
            0.60,
            (f"driver{i // 2}",),
        )
        for i in range(8)
    ]
    ticket = construct_ticket(legs, payout_multiple=Decimal("20"), bankroll=Decimal("1000"))
    result = simulate_ticket(ticket, simulations=5_000, seed=1)
    assert len(ticket.legs) == 6
    assert ticket.stake == Decimal("2.50")
    assert 0 <= result.joint_probability <= 1
    assert ticket.dependency_warning
