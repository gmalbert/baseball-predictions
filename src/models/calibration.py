"""Chronological out-of-fold probability calibration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class BetaCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1_000, max_iter=2_000)

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> BetaCalibrator:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        design = np.column_stack([np.log(clipped), -np.log1p(-clipped)])
        self.model.fit(design, targets)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        design = np.column_stack([np.log(clipped), -np.log1p(-clipped)])
        return self.model.predict_proba(design)[:, 1]


class TemperatureCalibrator:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> TemperatureCalibrator:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        y = np.asarray(targets)
        result = minimize(
            lambda x: log_loss(y, 1 / (1 + np.exp(-logits / np.exp(x[0])))),
            x0=np.array([0.0]),
            method="Nelder-Mead",
        )
        self.temperature = float(np.exp(result.x[0]))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        return 1 / (1 + np.exp(-logits / self.temperature))


@dataclass(frozen=True)
class CalibrationCandidate:
    name: str
    calibrator: object
    brier: float
    log_loss: float


def fit_calibration_candidates(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> list[CalibrationCandidate]:
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    candidates: list[tuple[str, object]] = []
    platt = LogisticRegression(C=1_000, max_iter=2_000).fit(probabilities.reshape(-1, 1), targets)
    candidates.append(("platt", platt))
    candidates.append(("beta", BetaCalibrator().fit(probabilities, targets)))
    candidates.append(
        ("isotonic", IsotonicRegression(out_of_bounds="clip").fit(probabilities, targets))
    )
    candidates.append(("temperature", TemperatureCalibrator().fit(probabilities, targets)))
    results = []
    for name, calibrator in candidates:
        if name == "platt":
            calibrated = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        else:
            calibrated = calibrator.predict(probabilities)
        results.append(
            CalibrationCandidate(
                name=name,
                calibrator=calibrator,
                brier=float(brier_score_loss(targets, calibrated)),
                log_loss=float(log_loss(targets, calibrated)),
            )
        )
    return sorted(results, key=lambda candidate: (candidate.log_loss, candidate.brier))
