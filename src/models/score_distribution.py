"""Coherent joint score distributions used to price sides, lines, and totals."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np


@dataclass(frozen=True)
class ScoreDistribution:
    matrix: np.ndarray  # [away_runs, home_runs]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or np.any(self.matrix < 0):
            raise ValueError("Invalid score matrix")
        if not np.isclose(self.matrix.sum(), 1.0, atol=1e-8):
            raise ValueError("Score matrix must sum to one")

    def home_moneyline(self) -> float:
        return float(np.triu(self.matrix, k=1).sum())

    def away_moneyline(self) -> float:
        return float(np.tril(self.matrix, k=-1).sum())

    def tie_probability(self) -> float:
        return float(np.trace(self.matrix))

    def run_line_probabilities(self, home_point: float) -> tuple[float, float, float]:
        away, home = np.indices(self.matrix.shape)
        adjusted = home - away + home_point
        return (
            float(self.matrix[adjusted > 0].sum()),
            float(self.matrix[adjusted < 0].sum()),
            float(self.matrix[np.isclose(adjusted, 0)].sum()),
        )

    def total_probabilities(self, point: float) -> tuple[float, float, float]:
        away, home = np.indices(self.matrix.shape)
        totals = away + home
        return (
            float(self.matrix[totals > point].sum()),
            float(self.matrix[totals < point].sum()),
            float(self.matrix[np.isclose(totals, point)].sum()),
        )

    def exact_score(self, away_runs: int, home_runs: int) -> float:
        return float(self.matrix[away_runs, home_runs])


def _poisson(rate: float, max_runs: int) -> np.ndarray:
    values = np.array([exp(-rate) * rate**k / factorial(k) for k in range(max_runs)])
    values[-1] += max(0.0, 1.0 - values.sum())
    return values


def independent_poisson_score_distribution(
    away_rate: float,
    home_rate: float,
    max_runs: int = 21,
) -> ScoreDistribution:
    if away_rate <= 0 or home_rate <= 0:
        raise ValueError("Run rates must be positive")
    away = _poisson(away_rate, max_runs)
    home = _poisson(home_rate, max_runs)
    matrix = np.outer(away, home)
    matrix /= matrix.sum()
    return ScoreDistribution(matrix)


def shared_environment_score_distribution(
    away_rate: float,
    home_rate: float,
    *,
    environment_sd: float = 0.15,
    samples: int = 5_000,
    max_runs: int = 21,
    seed: int = 42,
) -> ScoreDistribution:
    """Monte Carlo correlated/overdispersed score distribution.

    A shared log-normal environment moves both team rates coherently while team
    Poisson noise remains independent conditional on that environment.
    """
    if min(away_rate, home_rate, environment_sd) <= 0 or samples <= 0:
        raise ValueError("Rates, environment_sd, and samples must be positive")
    generator = np.random.default_rng(seed)
    environment = generator.lognormal(
        mean=-(environment_sd**2) / 2, sigma=environment_sd, size=samples
    )
    away = np.minimum(generator.poisson(away_rate * environment), max_runs - 1)
    home = np.minimum(generator.poisson(home_rate * environment), max_runs - 1)
    matrix = np.zeros((max_runs, max_runs), dtype=float)
    np.add.at(matrix, (away, home), 1.0)
    return ScoreDistribution(matrix / matrix.sum())
