"""Content-addressed model bundles with fail-closed compatibility checks."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field


class FeatureSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    dtype: str
    nullable: bool
    default: float | int | str | bool | None = None
    categories: tuple[str, ...] = ()
    transform: str | None = None


class ModelManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model_run_id: str
    model_name: str
    model_version: str
    market_id: str
    supported_snapshot_types: tuple[str, ...] = ()
    feature_set_version: str
    data_schema_version: str
    python_version: str
    sklearn_version: str
    dependency_lock_sha256: str
    dependency_versions: dict[str, str]
    artifact_sha256: str
    calibration_sha256: str | None = None
    calibration_method: str | None = None
    source_commit: str
    random_seed: int
    validation_definition: dict[str, Any]
    training_start: str
    training_end: str
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    features: tuple[FeatureSpec, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LoadedModel:
    def __init__(
        self,
        artifact_path: Path,
        manifest_path: Path,
        calibration_path: Path | None = None,
        dependency_lock: Path | None = None,
    ) -> None:
        if not artifact_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Both model artifact and manifest are required")
        self.manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(artifact_path) != self.manifest.artifact_sha256:
            raise ValueError("Model artifact checksum mismatch")
        runtime_python = ".".join(platform.python_version().split(".")[:2])
        manifest_python = ".".join(self.manifest.python_version.split(".")[:2])
        if runtime_python != manifest_python:
            raise RuntimeError(f"Python mismatch: {runtime_python} != {manifest_python}")
        if sklearn.__version__ != self.manifest.sklearn_version:
            raise RuntimeError(
                f"scikit-learn mismatch: {sklearn.__version__} != {self.manifest.sklearn_version}"
            )
        lock = dependency_lock or Path(__file__).resolve().parents[2] / "uv.lock"
        if not lock.is_file():
            raise FileNotFoundError(f"Dependency lock is required: {lock}")
        if sha256_file(lock) != self.manifest.dependency_lock_sha256:
            raise RuntimeError("Dependency lock checksum mismatch")
        if self.manifest.calibration_sha256:
            if calibration_path is None:
                calibration_path = artifact_path.with_name(
                    f"{artifact_path.stem}.calibration.joblib"
                )
            if (
                calibration_path is None
                or sha256_file(calibration_path) != self.manifest.calibration_sha256
            ):
                raise ValueError("Calibration artifact checksum mismatch")
        self.estimator = joblib.load(artifact_path)
        self.calibrator = joblib.load(calibration_path) if calibration_path else None

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        names = [feature.name for feature in self.manifest.features]
        missing = [name for name in names if name not in frame]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        matrix = frame.loc[:, names].copy()
        for spec in self.manifest.features:
            if not spec.nullable and matrix[spec.name].isna().any():
                raise ValueError(f"Non-nullable feature contains nulls: {spec.name}")
            if spec.categories:
                unknown = set(matrix[spec.name].dropna().astype(str)) - set(spec.categories)
                if unknown:
                    raise ValueError(f"Unknown categories for {spec.name}: {sorted(unknown)}")
            try:
                matrix[spec.name] = matrix[spec.name].astype(spec.dtype)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Invalid dtype for {spec.name}: expected {spec.dtype}") from exc
        expected = getattr(self.estimator, "n_features_in_", len(names))
        if expected != len(names):
            raise ValueError(
                f"Estimator feature count {expected} != manifest feature count {len(names)}"
            )
        return matrix

    def predict_probability(self, frame: pd.DataFrame) -> pd.Series:
        matrix = self.matrix(frame)
        raw = self.estimator.predict_proba(matrix)[:, 1]
        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                raw = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            else:
                raw = self.calibrator.predict(raw)
        return pd.Series(raw, index=frame.index, name="probability")


def write_manifest(path: Path, manifest: ModelManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def quarantine_reason(artifact_path: Path, manifest_path: Path | None) -> str | None:
    if manifest_path is None or not manifest_path.is_file():
        return "missing_manifest"
    try:
        manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "invalid_manifest"
    if not artifact_path.is_file():
        return "missing_artifact"
    if sha256_file(artifact_path) != manifest.artifact_sha256:
        return "checksum_mismatch"
    return None


def compatibility_reason(
    artifact_path: Path,
    manifest_path: Path | None,
    *,
    dependency_lock: Path | None = None,
) -> str | None:
    """Return a stable fail-closed reason covering files, checksums, and runtime."""
    reason = quarantine_reason(artifact_path, manifest_path)
    if reason is not None:
        return reason
    assert manifest_path is not None
    try:
        LoadedModel(
            artifact_path,
            manifest_path,
            dependency_lock=dependency_lock,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"runtime_incompatible:{type(exc).__name__}:{exc}"
    return None
