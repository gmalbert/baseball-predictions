"""Versioned feature metadata, information-time contracts, and null policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: str
    dtype: str
    entity_grain: str
    source: str
    availability_delay: str
    null_policy: str
    formula: str
    leakage_classification: str = "pregame_only"
    owner: str = "feature_model"
    version: str = "1.0.0"


class FeatureRegistry:
    def __init__(self, definitions: list[FeatureDefinition] | None = None) -> None:
        self._definitions = {definition.name: definition for definition in definitions or []}

    def register(self, definition: FeatureDefinition) -> None:
        existing = self._definitions.get(definition.name)
        if existing is not None and existing != definition:
            raise ValueError(
                f"Feature already registered with different semantics: {definition.name}"
            )
        self._definitions[definition.name] = definition

    def get(self, name: str) -> FeatureDefinition:
        if name not in self._definitions:
            raise KeyError(name)
        return self._definitions[name]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self._definitions.values()])

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().sort_values("name").to_json(path, orient="records", indent=2)
