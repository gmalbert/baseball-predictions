"""Probability and bankroll metrics used by model promotion reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class ProbabilityMetrics:
    observations: int
    log_loss: float
    brier_score: float
    brier_skill_vs_market: float | None
    calibration_intercept: float
    calibration_slope: float
    expected_calibration_error: float
    maximum_calibration_error: float
    sharpness: float
    roc_auc: float | None


def _bounded(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite one-dimensional array")
    return np.clip(values, 1e-6, 1 - 1e-6)


def calibration_intercept_slope(
    outcomes: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Fit outcome ~ logit(probability), returning intercept and slope."""
    target = np.asarray(outcomes, dtype=int)
    probability = _bounded(probabilities)
    if len(np.unique(target)) < 2:
        return float("nan"), float("nan")
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logits, target)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def reliability_table(
    outcomes: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    """Return weighted calibration bins with beta-binomial credible intervals."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    target = np.asarray(outcomes, dtype=int)
    probability = _bounded(probabilities)
    if len(target) != len(probability):
        raise ValueError("Outcomes and probabilities must have equal length")
    bucket = np.minimum((probability * bins).astype(int), bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = bucket == index
        count = int(mask.sum())
        if count == 0:
            continue
        wins = int(target[mask].sum())
        rows.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": float(probability[mask].mean()),
                "observed_rate": wins / count,
                "credible_low": float(beta.ppf(0.025, wins + 0.5, count - wins + 0.5)),
                "credible_high": float(beta.ppf(0.975, wins + 0.5, count - wins + 0.5)),
            }
        )
    return pd.DataFrame(rows)


def probability_metrics(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    market_probabilities: np.ndarray | None = None,
    bins: int = 10,
) -> ProbabilityMetrics:
    target = np.asarray(outcomes, dtype=int)
    probability = _bounded(probabilities)
    if len(target) != len(probability) or len(target) == 0:
        raise ValueError("Outcomes and probabilities must be non-empty and equal length")
    table = reliability_table(target, probability, bins=bins)
    gaps = (table["observed_rate"] - table["mean_probability"]).abs()
    ece = float((gaps * table["count"] / len(target)).sum())
    brier = float(brier_score_loss(target, probability))
    skill: float | None = None
    if market_probabilities is not None:
        market_brier = float(brier_score_loss(target, _bounded(market_probabilities)))
        skill = float(1 - brier / market_brier) if market_brier > 0 else None
    intercept, slope = calibration_intercept_slope(target, probability)
    auc = float(roc_auc_score(target, probability)) if len(np.unique(target)) > 1 else None
    return ProbabilityMetrics(
        observations=len(target),
        log_loss=float(log_loss(target, probability)),
        brier_score=brier,
        brier_skill_vs_market=skill,
        calibration_intercept=intercept,
        calibration_slope=slope,
        expected_calibration_error=ece,
        maximum_calibration_error=float(gaps.max()),
        sharpness=float(np.var(probability)),
        roc_auc=auc,
    )


def expected_log_growth(stakes: np.ndarray, profits: np.ndarray, bankroll: float) -> float:
    """Mean log bankroll growth for chronological settled wagers."""
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    stakes_array = np.asarray(stakes, dtype=float)
    profits_array = np.asarray(profits, dtype=float)
    if stakes_array.shape != profits_array.shape:
        raise ValueError("stakes and profits must have equal shape")
    running = bankroll
    growth: list[float] = []
    for stake, profit in zip(stakes_array, profits_array, strict=True):
        if stake < 0 or stake > running:
            raise ValueError("stake must be non-negative and cannot exceed bankroll")
        next_bankroll = running + profit
        if next_bankroll <= 0:
            return float("-inf")
        growth.append(float(np.log(next_bankroll / running)))
        running = next_bankroll
    return float(np.mean(growth)) if growth else 0.0
