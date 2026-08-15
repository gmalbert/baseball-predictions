from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from src.models.manifest import FeatureSpec, LoadedModel, ModelManifest, sha256_file, write_manifest


def test_manifest_bundle_recreates_prediction_and_rejects_wrong_lock(tmp_path: Path) -> None:
    frame = pd.DataFrame({"strength": [0.1, 0.3, 0.8, 0.9]})
    estimator = LogisticRegression().fit(frame, np.array([0, 0, 1, 1]))
    artifact = tmp_path / "model.joblib"
    joblib.dump(estimator, artifact)
    project_lock = Path(__file__).resolve().parents[2] / "uv.lock"
    manifest = ModelManifest(
        model_run_id="test",
        model_name="logistic",
        model_version="2.0.0",
        market_id="moneyline_full_game",
        supported_snapshot_types=("morning",),
        feature_set_version="v2",
        data_schema_version="2.0.0",
        python_version=".".join(__import__("platform").python_version().split(".")),
        sklearn_version=sklearn.__version__,
        dependency_lock_sha256=sha256_file(project_lock),
        dependency_versions={"scikit-learn": sklearn.__version__},
        artifact_sha256=sha256_file(artifact),
        source_commit="test",
        random_seed=42,
        validation_definition={"kind": "chronological"},
        training_start="2024-01-01",
        training_end="2025-12-31",
        features=(FeatureSpec(name="strength", dtype="float64", nullable=False),),
    )
    manifest_path = tmp_path / "model.manifest.json"
    write_manifest(manifest_path, manifest)
    loaded = LoadedModel(artifact, manifest_path, dependency_lock=project_lock)
    assert loaded.predict_probability(frame).between(0, 1).all()

    wrong_lock = tmp_path / "wrong.lock"
    wrong_lock.write_text("not the production lock", encoding="utf-8")
    try:
        LoadedModel(artifact, manifest_path, dependency_lock=wrong_lock)
    except RuntimeError as exc:
        assert "lock checksum" in str(exc)
    else:
        raise AssertionError("A mismatched dependency lock was accepted")
