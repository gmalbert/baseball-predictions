"""Explanation stability and scenario sensitivity diagnostics (non-causal)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sensitivity:
    feature: str
    base_probability: float
    lower_probability: float
    upper_probability: float
    maximum_change: float


def scenario_sensitivity(
    predict_probability,
    row: pd.DataFrame,
    scenarios: dict[str, tuple[float, float]],
) -> list[Sensitivity]:
    if len(row) != 1:
        raise ValueError("Sensitivity requires exactly one feature row")
    base = float(np.asarray(predict_probability(row))[0])
    results = []
    for feature, (lower, upper) in scenarios.items():
        if feature not in row:
            raise KeyError(feature)
        low_row, high_row = row.copy(), row.copy()
        low_row.loc[:, feature] = lower
        high_row.loc[:, feature] = upper
        low = float(np.asarray(predict_probability(low_row))[0])
        high = float(np.asarray(predict_probability(high_row))[0])
        results.append(
            Sensitivity(feature, base, low, high, max(abs(low - base), abs(high - base)))
        )
    return sorted(results, key=lambda result: result.maximum_change, reverse=True)
