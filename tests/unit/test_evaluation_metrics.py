import numpy as np

from src.evaluation.metrics import expected_log_growth, probability_metrics, reliability_table


def test_probability_metrics_weight_calibration_bins_by_observations() -> None:
    target = np.array([0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    metrics = probability_metrics(target, probability, market_probabilities=np.full(5, 0.5))
    table = reliability_table(target, probability)
    expected = float(
        (
            (table["observed_rate"] - table["mean_probability"]).abs()
            * table["count"]
            / len(target)
        ).sum()
    )
    assert metrics.expected_calibration_error == expected
    assert metrics.brier_skill_vs_market is not None


def test_expected_log_growth_tracks_chronological_bankroll() -> None:
    growth = expected_log_growth(
        stakes=np.array([10.0, 10.0]), profits=np.array([10.0, -10.0]), bankroll=100.0
    )
    assert abs(growth) < 1e-12
