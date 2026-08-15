"""Opportunity/rate decomposition for supported player prop distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np

SUPPORTED_PROPS = {"strikeouts", "outs_recorded", "hits", "total_bases", "home_runs"}


@dataclass(frozen=True)
class PropDistribution:
    market: str
    probabilities: np.ndarray
    availability_probability: float

    def over(self, line: float) -> float:
        values = np.arange(len(self.probabilities))
        return float(self.availability_probability * self.probabilities[values > line].sum())

    def under(self, line: float) -> float:
        values = np.arange(len(self.probabilities))
        return float(self.availability_probability * self.probabilities[values < line].sum())

    def void_probability(self) -> float:
        return 1 - self.availability_probability


def poisson_prop_distribution(
    market: str,
    *,
    expected_opportunities: float,
    event_rate: float,
    availability_probability: float,
    maximum: int = 20,
) -> PropDistribution:
    if market not in SUPPORTED_PROPS:
        raise KeyError(market)
    if expected_opportunities < 0 or event_rate < 0 or not 0 <= availability_probability <= 1:
        raise ValueError("Invalid prop inputs")
    rate = expected_opportunities * event_rate
    values = np.array([exp(-rate) * rate**k / factorial(k) for k in range(maximum)])
    values[-1] += max(0.0, 1 - values.sum())
    return PropDistribution(market, values, availability_probability)
