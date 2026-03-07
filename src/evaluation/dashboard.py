# src/evaluation/dashboard.py
"""Generate data structures for the model performance dashboard."""

from __future__ import annotations

import numpy as np

from .backtester import BacktestResult
from .calibration import calibration_plot_data
from .profitability import cumulative_profit_data, monthly_breakdown


def generate_dashboard_data(
    backtest_results: dict[str, BacktestResult],
    y_true_map: dict[str, list],
    y_prob_map: dict[str, list],
) -> dict:
    """Generate all data needed for the frontend performance dashboard.

    Args:
        backtest_results: Mapping of ``model_name → BacktestResult``.
        y_true_map:       Mapping of ``model_name → actual outcomes``.
        y_prob_map:       Mapping of ``model_name → predicted probabilities``.

    Returns:
        Dict structure ready for JSON serialisation with keys:
        ``leaderboard``, ``cumulative_charts``, ``calibration_charts``,
        ``monthly_tables``.
    """
    dashboard: dict = {
        "leaderboard": [],
        "cumulative_charts": {},
        "calibration_charts": {},
        "monthly_tables": {},
    }

    for name, result in backtest_results.items():
        # Leaderboard row
        dashboard["leaderboard"].append(result.summary())

        # Cumulative profit chart data
        dashboard["cumulative_charts"][name] = cumulative_profit_data(result)

        # Monthly breakdown
        monthly = monthly_breakdown(result)
        dashboard["monthly_tables"][name] = monthly.to_dict(orient="records")

        # Calibration chart
        if name in y_true_map and name in y_prob_map:
            dashboard["calibration_charts"][name] = calibration_plot_data(
                np.array(y_true_map[name]),
                np.array(y_prob_map[name]),
            )

    # Sort leaderboard by ROI descending
    dashboard["leaderboard"].sort(key=lambda x: x["roi"], reverse=True)

    return dashboard
