"""Feature PSI, probability calibration drift, and incident thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if not len(reference) or not len(current):
        raise ValueError("Both samples are required")
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, boundaries)
    cur_counts, _ = np.histogram(current, boundaries)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass(frozen=True)
class DriftAlert:
    metric: str
    value: float
    threshold: float
    severity: str
    affected_scope: str


def drift_alert(
    metric: str, value: float, *, warning: float, critical: float, scope: str
) -> DriftAlert | None:
    if value >= critical:
        return DriftAlert(metric, value, critical, "critical", scope)
    if value >= warning:
        return DriftAlert(metric, value, warning, "warning", scope)
    return None
