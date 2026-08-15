"""Per-PA outcome model: calibrated classifiers that produce a per-PA outcome
mix consumed by the game simulator.

One binary calibrated bundle is trained per ``PaOutcome`` (excluding ``OUT``,
which is the residual), each through ``train_calibrated_bundle`` with its own
manifest, checksums, and registry lifecycle.  ``PlateAppearanceOutcomeModel``
loads the bundles, normalizes the per-outcome probabilities to a valid mix,
and implements the simulator's ``OutcomeModel`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.models.game_simulator import OutcomeModel
from src.models.manifest import LoadedModel
from src.models.plate_appearance import PaOutcome, PaState
from src.models.training import train_calibrated_bundle

# Outcomes modeled explicitly; OUT is the residual so the mix sums to one.
TRAINED_OUTCOMES = (
    PaOutcome.STRIKEOUT,
    PaOutcome.WALK,
    PaOutcome.HBP,
    PaOutcome.SINGLE,
    PaOutcome.DOUBLE,
    PaOutcome.TRIPLE,
    PaOutcome.HOME_RUN,
)


def train_pa_outcome_bundles(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    date_column: str,
    output_dir: Path,
    model_run_prefix: str,
    market_id: str,
    feature_set_version: str,
    data_schema_version: str,
    source_commit: str,
    dependency_lock: Path,
    model_version: str = "2.0.0",
    estimator_factory: Callable[[], object] | None = None,
) -> list[Path]:
    """Train one calibrated bundle per modeled outcome and return manifests.

    ``frame`` must contain ``outcome_<value>`` binary columns from
    ``src.labels.plate_appearances.pa_outcome_target`` plus the feature set.
    """
    if estimator_factory is None:

        def _default_estimator() -> object:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=42
            )

        estimator_factory = _default_estimator
    trained: list[Path] = []
    for outcome in TRAINED_OUTCOMES:
        target = f"outcome_{outcome.value}"
        if target not in frame:
            raise KeyError(f"Frame missing target column {target}")
        bundle = train_calibrated_bundle(
            frame,
            estimator_factory=estimator_factory,
            feature_columns=feature_columns,
            target_column=target,
            date_column=date_column,
            market_probability_column=None,
            output_dir=output_dir,
            model_run_id=f"{model_run_prefix}_{outcome.value}",
            model_name="plate_appearance",
            model_version=model_version,
            market_id=market_id,
            feature_set_version=feature_set_version,
            data_schema_version=data_schema_version,
            source_commit=source_commit,
            dependency_lock=dependency_lock,
        )
        trained.append(bundle.manifest_path)
    return trained


@dataclass
class PlateAppearanceOutcomeModel(OutcomeModel):
    """Loads per-outcome calibrated bundles and produces a normalized mix."""

    bundles: dict[PaOutcome, LoadedModel]
    residual_outcome: PaOutcome = PaOutcome.OUT

    @classmethod
    def load(cls, output_dir: Path, model_run_prefix: str) -> "PlateAppearanceOutcomeModel":
        bundles: dict[PaOutcome, LoadedModel] = {}
        for outcome in TRAINED_OUTCOMES:
            artifact = output_dir / f"{model_run_prefix}_{outcome.value}.joblib"
            manifest = artifact.with_suffix(".manifest.json")
            calibration = artifact.with_suffix(".calibration.joblib")
            bundles[outcome] = LoadedModel(artifact, manifest, calibration)
        return cls(bundles=bundles)

    def outcome_probabilities(self, batter_id: str, pitcher_id: str, state: PaState) -> np.ndarray:
        # The simulator passes placeholder ids; a real deployment builds a
        # feature row per PA from the snapshot.  This contract returns the
        # residual mix when features are unavailable, and the subclass
        # (feature-driven) overrides this method.
        raise NotImplementedError(
            "PlateAppearanceOutcomeModel requires a feature row; use "
            "FeaturePlateAppearanceOutcomeModel or the snapshot-aware adapter."
        )


@dataclass
class FeaturePlateAppearanceOutcomeModel(PlateAppearanceOutcomeModel):
    """Per-PA outcome model that builds a mix from a feature row lookup."""

    feature_columns: tuple[str, ...] = ()
    # (batter_id, pitcher_id) -> feature row (dict or pd.Series)
    feature_rows: dict[tuple[str, str], object] | None = None

    def _mix_from_row(self, row: object) -> np.ndarray:
        features = pd.DataFrame([row])[list(self.feature_columns)]
        probabilities = np.zeros(len(TRAINED_OUTCOMES))
        for i, outcome in enumerate(TRAINED_OUTCOMES):
            model = self.bundles[outcome]
            probabilities[i] = model.predict_probability(features).to_numpy()[0]
        residual = max(0.0, 1.0 - probabilities.sum())
        total = probabilities.sum() + residual
        result = np.zeros(len(list(PaOutcome)))
        outcome_list = list(PaOutcome)
        for i, outcome in enumerate(TRAINED_OUTCOMES):
            result[outcome_list.index(outcome)] = probabilities[i] / total
        result[outcome_list.index(self.residual_outcome)] = residual / total
        return result

    def outcome_probabilities(self, batter_id: str, pitcher_id: str, state: PaState) -> np.ndarray:
        if self.feature_rows is None:
            raise ValueError("feature_rows must be provided for a feature-driven model")
        row = self.feature_rows.get((batter_id, pitcher_id))
        if row is None:
            raise KeyError(f"No feature row for ({batter_id}, {pitcher_id})")
        return self._mix_from_row(row)
