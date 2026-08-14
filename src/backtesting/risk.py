"""Block-bootstrap drawdown, ruin, recovery, and deployment stress scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskSummary:
    scenario: str
    median_return: float
    probability_of_loss: float
    probability_of_ruin: float
    median_max_drawdown: float
    drawdown_95: float
    median_recovery_periods: float


def _path_metrics(returns: np.ndarray, ruin_threshold: float) -> tuple[float, float, bool, int]:
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[:-1]
    drawdowns = 1 - wealth / peak
    maximum = float(drawdowns.max(initial=0))
    ruined = bool((wealth <= ruin_threshold).any())
    trough = int(drawdowns.argmax()) if len(drawdowns) else 0
    recovered = np.where(wealth[trough:] >= peak[trough])[0] if len(wealth) else np.array([])
    recovery = int(recovered[0]) if len(recovered) else len(returns) - trough
    return float(wealth[-1] - 1) if len(wealth) else 0.0, maximum, ruined, recovery


def block_bootstrap_risk(
    returns: np.ndarray,
    *,
    scenario: str = "base",
    simulations: int = 2_000,
    block_size: int = 7,
    ruin_threshold: float = 0.5,
    seed: int = 42,
) -> RiskSummary:
    values = np.asarray(returns, dtype=float)
    if not len(values) or np.any(values <= -1):
        raise ValueError("Returns must be non-empty and greater than -100%")
    generator = np.random.default_rng(seed)
    metrics = []
    starts = np.arange(max(len(values) - block_size + 1, 1))
    for _ in range(simulations):
        sampled = []
        while len(sampled) < len(values):
            start = int(generator.choice(starts))
            sampled.extend(values[start : start + block_size])
        metrics.append(_path_metrics(np.asarray(sampled[: len(values)]), ruin_threshold))
    array = np.asarray(metrics)
    return RiskSummary(
        scenario=scenario,
        median_return=float(np.median(array[:, 0])),
        probability_of_loss=float((array[:, 0] < 0).mean()),
        probability_of_ruin=float(array[:, 2].mean()),
        median_max_drawdown=float(np.median(array[:, 1])),
        drawdown_95=float(np.quantile(array[:, 1], 0.95)),
        median_recovery_periods=float(np.median(array[:, 3])),
    )


def required_stress_scenarios(returns: np.ndarray) -> list[RiskSummary]:
    values = np.asarray(returns, dtype=float)
    scenarios = {
        "base": values,
        "25pct_overconfident": values * 0.75,
        "50pct_overconfident": values * 0.50,
        "100pct_overconfident": np.minimum(values, 0),
        "5c_worse_fills": values - 0.005,
        "10c_worse_fills": values - 0.010,
        "20c_worse_fills": values - 0.020,
        "edge_decay_half": values * 0.50,
        "double_correlation": np.repeat(values.reshape(-1, 2).sum(axis=1), 2)[: len(values)]
        if len(values) % 2 == 0
        else values,
        "top_book_unavailable": np.where(np.arange(len(values)) % 3 == 0, 0, values),
        "source_outage_week": np.where(np.arange(len(values)) % 28 < 7, 0, values),
        "regime_shift": values - np.std(values),
        "largest_losing_block_twice": values.copy(),
    }
    if len(values) >= 7:
        rolling = np.convolve(values, np.ones(7), mode="valid")
        start = int(rolling.argmin())
        worst = values[start : start + 7]
        scenarios["largest_losing_block_twice"] = np.r_[values, worst, worst]
    return [block_bootstrap_risk(result, scenario=name) for name, result in scenarios.items()]
