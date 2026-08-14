from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.manifest import LoadedModel
from src.models.training import train_calibrated_bundle


def test_chronological_training_writes_loadable_calibrated_bundle(tmp_path: Path) -> None:
    rows = 120
    frame = pd.DataFrame(
        {
            "as_of_time": pd.date_range("2024-01-01", periods=rows, tz="UTC"),
            "strength": np.linspace(-2, 2, rows),
            "market_probability": np.linspace(0.35, 0.65, rows),
            "target": np.tile([0, 1], rows // 2),
        }
    )
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    bundle = train_calibrated_bundle(
        frame,
        estimator_factory=lambda: LogisticRegression(),
        feature_columns=["strength", "market_probability"],
        target_column="target",
        date_column="as_of_time",
        market_probability_column="market_probability",
        output_dir=tmp_path,
        model_run_id="training-smoke",
        model_name="logistic",
        model_version="2.0.0",
        market_id="moneyline_full_game",
        feature_set_version="v2",
        data_schema_version="2.0.0",
        source_commit="test",
        dependency_lock=lock,
    )
    loaded = LoadedModel(
        bundle.artifact_path,
        bundle.manifest_path,
        dependency_lock=lock,
    )
    probabilities = loaded.predict_probability(frame.tail(5))
    assert probabilities.between(0, 1).all()
    assert bundle.calibration_method in {"platt", "beta", "isotonic", "temperature"}
    assert "market" in bundle.baseline_metrics
