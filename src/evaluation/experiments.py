"""Append-only experiment registry and simple overfit diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    hypothesis: str
    feature_version: str
    data_version: str
    code_commit: str
    chronological_folds: tuple[str, ...]
    search_space: dict[str, object]
    metrics: dict[str, float]
    decision: str
    registered_at: str


class ExperimentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, experiment: Experiment) -> None:
        rows = [] if not self.path.exists() else json.loads(self.path.read_text(encoding="utf-8"))
        if any(row["experiment_id"] == experiment.experiment_id for row in rows):
            raise ValueError(f"Duplicate experiment: {experiment.experiment_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([*rows, asdict(experiment)], indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)


def multiple_testing_haircut(observed_z: float, trials: int) -> float:
    if trials < 1:
        raise ValueError("trials must be positive")
    expected_maximum = NormalDist().inv_cdf(max(0.5, 1 - 1 / (2 * trials)))
    return observed_z - expected_maximum


def probability_of_backtest_overfit(train_ranks: list[int], test_ranks: list[int]) -> float:
    if len(train_ranks) != len(test_ranks) or not train_ranks:
        raise ValueError("Train/test ranks must have equal non-zero length")
    median_test = sorted(test_ranks)[len(test_ranks) // 2]
    selected = min(range(len(train_ranks)), key=lambda index: train_ranks[index])
    return float(test_ranks[selected] > median_test)
