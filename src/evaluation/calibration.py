# src/evaluation/calibration.py
"""Calibration analysis: are predicted probabilities accurate?

A well-calibrated model: when it predicts 60%, events happen ~60% of the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "model",
) -> pd.DataFrame:
    """Generate a calibration report showing predicted vs. actual rates.

    Args:
        y_true:     Binary ground-truth labels (0 or 1).
        y_prob:     Predicted probabilities in [0, 1].
        n_bins:     Number of calibration bins.
        model_name: Label used in printed output.

    Returns:
        DataFrame with one row per bin showing count, avg predicted,
        actual rate, and calibration error.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}",
            "count": int(mask.sum()),
            "avg_predicted": float(y_prob[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
            "calibration_error": float(abs(y_prob[mask].mean() - y_true[mask].mean())),
        })

    report = pd.DataFrame(rows)

    brier = brier_score_loss(y_true, y_prob)
    ece = report["calibration_error"].mean() if not report.empty else float("nan")

    print(f"\n=== Calibration Report: {model_name} ===")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(report.to_string(index=False))

    return report


def calibration_plot_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Return data needed to render a calibration plot.

    The caller can plot:
    - X axis: mean predicted probability per bin
    - Y axis: actual fraction of positives per bin
    - A diagonal line represents perfect calibration

    Returns:
        Dict with keys ``mean_predicted``, ``fraction_positive``,
        ``perfect_line``, and ``brier_score``.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    return {
        "mean_predicted": mean_predicted.tolist(),
        "fraction_positive": fraction_positive.tolist(),
        "perfect_line": [[0, 0], [1, 1]],
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
