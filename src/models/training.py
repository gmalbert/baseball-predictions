"""Chronological fit/calibrate/final-test workflow and immutable model bundles."""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.calibration import CalibrationCandidate, fit_calibration_candidates
from src.models.manifest import FeatureSpec, ModelManifest, sha256_file, write_manifest


@dataclass(frozen=True)
class ChronologicalPartition:
    train: pd.Index
    calibration: pd.Index
    final_test: pd.Index


def chronological_partition(
    frame: pd.DataFrame,
    *,
    date_col: str,
    calibration_fraction: float = 0.15,
    final_test_fraction: float = 0.15,
    embargo_days: int = 1,
) -> ChronologicalPartition:
    if not 0 < calibration_fraction < 1 or not 0 < final_test_fraction < 1:
        raise ValueError("Fractions must be in (0, 1)")
    ordered = frame.sort_values(date_col)
    dates = pd.to_datetime(ordered[date_col], utc=True)
    final_position = int(len(ordered) * (1 - final_test_fraction))
    calibration_position = int(len(ordered) * (1 - final_test_fraction - calibration_fraction))
    if calibration_position <= 0 or final_position <= calibration_position:
        raise ValueError("Insufficient rows for chronological partitions")
    calibration_start = dates.iloc[calibration_position]
    final_start = dates.iloc[final_position]
    embargo = timedelta(days=embargo_days)
    train = ordered.index[dates < calibration_start - embargo]
    calibration = ordered.index[(dates >= calibration_start) & (dates < final_start - embargo)]
    final_test = ordered.index[dates >= final_start]
    if min(len(train), len(calibration), len(final_test)) == 0:
        raise ValueError("Chronological partition produced an empty split")
    return ChronologicalPartition(train, calibration, final_test)


@dataclass(frozen=True)
class TrainedBundle:
    artifact_path: Path
    calibration_path: Path
    manifest_path: Path
    calibration_method: str
    final_metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]


def _calibrate(candidate: CalibrationCandidate, probabilities: np.ndarray) -> np.ndarray:
    if candidate.name == "platt":
        return candidate.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    return candidate.calibrator.predict(probabilities)


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    values = {
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
    }
    if len(np.unique(target)) > 1:
        values["roc_auc"] = float(roc_auc_score(target, probability))
    return values


def train_calibrated_bundle(
    frame: pd.DataFrame,
    *,
    estimator_factory: Callable[[], object],
    feature_columns: list[str],
    target_column: str,
    date_column: str,
    market_probability_column: str | None,
    output_dir: Path,
    model_run_id: str,
    model_name: str,
    model_version: str,
    market_id: str,
    feature_set_version: str,
    data_schema_version: str,
    source_commit: str,
    dependency_lock: Path,
    supported_snapshot_types: tuple[str, ...] = ("morning", "confirmed_lineup", "pregame_30m"),
) -> TrainedBundle:
    partition = chronological_partition(frame, date_col=date_column)
    estimator = estimator_factory()
    estimator.fit(
        frame.loc[partition.train, feature_columns], frame.loc[partition.train, target_column]
    )
    estimator.feature_cols_ = feature_columns
    raw_calibration = estimator.predict_proba(frame.loc[partition.calibration, feature_columns])[
        :, 1
    ]
    candidates = fit_calibration_candidates(
        raw_calibration, frame.loc[partition.calibration, target_column].to_numpy()
    )
    selected = candidates[0]
    raw_test = estimator.predict_proba(frame.loc[partition.final_test, feature_columns])[:, 1]
    calibrated_test = _calibrate(selected, raw_test)
    target_test = frame.loc[partition.final_test, target_column].to_numpy()
    final_metrics = _metrics(target_test, calibrated_test)
    baseline_metrics = {
        "constant": _metrics(
            target_test, np.full(len(target_test), frame.loc[partition.train, target_column].mean())
        )
    }
    if market_probability_column:
        baseline_metrics["market"] = _metrics(
            target_test, frame.loc[partition.final_test, market_probability_column].to_numpy(float)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{model_run_id}.joblib"
    calibration_path = output_dir / f"{model_run_id}.calibration.joblib"
    manifest_path = output_dir / f"{model_run_id}.manifest.json"
    joblib.dump(estimator, artifact_path)
    joblib.dump(selected.calibrator, calibration_path)
    manifest = ModelManifest(
        model_run_id=model_run_id,
        model_name=model_name,
        model_version=model_version,
        market_id=market_id,
        supported_snapshot_types=supported_snapshot_types,
        feature_set_version=feature_set_version,
        data_schema_version=data_schema_version,
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        dependency_lock_sha256=sha256_file(dependency_lock),
        dependency_versions={
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        artifact_sha256=sha256_file(artifact_path),
        calibration_sha256=sha256_file(calibration_path),
        calibration_method=selected.name,
        source_commit=source_commit,
        random_seed=42,
        validation_definition={
            "kind": "chronological_fit_calibration_final_test",
            "calibration_fraction": 0.15,
            "final_test_fraction": 0.15,
            "embargo_days": 1,
        },
        training_start=str(pd.to_datetime(frame.loc[partition.train, date_column]).min().date()),
        training_end=str(pd.to_datetime(frame.loc[partition.train, date_column]).max().date()),
        metrics={**final_metrics, "calibration_method": selected.name},
        features=tuple(
            FeatureSpec(
                name=column,
                dtype=str(frame[column].dtype),
                nullable=bool(frame[column].isna().any()),
            )
            for column in feature_columns
        ),
    )
    write_manifest(manifest_path, manifest)
    return TrainedBundle(
        artifact_path,
        calibration_path,
        manifest_path,
        selected.name,
        final_metrics,
        baseline_metrics,
    )
