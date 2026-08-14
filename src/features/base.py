"""Common feature-family interface and safe composition."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from src.features.asof import assert_feature_watermarks


class FeatureBuilder(Protocol):
    name: str
    version: str
    required_sources: tuple[str, ...]

    def build(self, entities: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame: ...

    def validate(self, frame: pd.DataFrame, *, as_of: datetime) -> None: ...


def combine_feature_families(
    base: pd.DataFrame,
    builders: list[FeatureBuilder],
    *,
    as_of: datetime,
) -> pd.DataFrame:
    result = base.copy()
    for builder in builders:
        family = builder.build(base, as_of=as_of)
        builder.validate(family, as_of=as_of)
        keys = [column for column in ("game_id", "team_id", "player_id") if column in family]
        if not keys:
            raise ValueError(f"{builder.name} exposes no entity key")
        payload = [column for column in family if column not in result or column in keys]
        result = result.merge(family[payload], on=keys, how="left", validate="one_to_one")
    return result


def validate_family(frame: pd.DataFrame, *, as_of: datetime) -> None:
    if frame.empty:
        raise ValueError("Feature family produced no rows")
    assert_feature_watermarks(frame)
    if (pd.to_datetime(frame["as_of_time"], utc=True) != pd.Timestamp(as_of)).any():
        raise ValueError("Feature family used a different cutoff")
