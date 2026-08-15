"""One shared simulation preserves dependence across all ticket legs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pick6.domain import Pick6Ticket


@dataclass(frozen=True)
class TicketSimulation:
    joint_probability: float
    expected_value_per_unit: float
    standard_error: float
    simulations: int


def simulate_ticket(
    ticket: Pick6Ticket,
    *,
    correlation: np.ndarray | None = None,
    simulations: int = 100_000,
    seed: int = 42,
) -> TicketSimulation:
    probabilities = np.asarray([row.marginal_probability for row in ticket.legs])
    if correlation is None:
        correlation = np.eye(6)
        for i, left in enumerate(ticket.legs):
            for j, right in enumerate(ticket.legs):
                if i != j and (
                    left.game_id == right.game_id or set(left.driver_ids) & set(right.driver_ids)
                ):
                    correlation[i, j] = 0.25
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (6, 6) or not np.allclose(correlation, correlation.T):
        raise ValueError("Correlation must be a symmetric 6x6 matrix")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues.min() < -1e-8:
        raise ValueError("Correlation matrix must be positive semidefinite")
    from scipy.stats import norm

    thresholds = norm.ppf(np.clip(probabilities, 1e-9, 1 - 1e-9))
    generator = np.random.default_rng(seed)
    latent = generator.multivariate_normal(np.zeros(6), correlation, size=simulations)
    success = (latent <= thresholds).all(axis=1)
    probability = float(success.mean())
    standard_error = float(np.sqrt(probability * (1 - probability) / simulations))
    ev = probability * (float(ticket.payout_multiple) - 1) - (1 - probability)
    return TicketSimulation(probability, ev, standard_error, simulations)
