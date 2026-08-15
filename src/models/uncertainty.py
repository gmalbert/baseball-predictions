"""Bootstrap/scenario uncertainty intervals and abstention helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbabilityInterval:
    mean: float
    low: float
    high: float
    standard_deviation: float
    samples: int

    def spans(self, threshold: float) -> bool:
        return self.low <= threshold <= self.high


def probability_interval(samples: np.ndarray, confidence: float = 0.9) -> ProbabilityInterval:
    values = np.asarray(samples, dtype=float).reshape(-1)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
    ):
        raise ValueError("Probability samples must be finite values in [0, 1]")
    alpha = (1 - confidence) / 2
    return ProbabilityInterval(
        mean=float(values.mean()),
        low=float(np.quantile(values, alpha)),
        high=float(np.quantile(values, 1 - alpha)),
        standard_deviation=float(values.std(ddof=0)),
        samples=len(values),
    )


def scenario_samples(
    base_probability: float,
    *,
    model_disagreement: list[float] | None = None,
    lineup_adjustments: list[float] | None = None,
    weather_adjustments: list[float] | None = None,
    samples: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    result = np.full(samples, base_probability, dtype=float)
    for alternatives in (model_disagreement, lineup_adjustments, weather_adjustments):
        if alternatives:
            result += generator.choice(np.asarray(alternatives, dtype=float), size=samples)
    return np.clip(result, 0, 1)
