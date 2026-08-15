"""Leakage-safe joins and rolling features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def point_in_time_join(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    by: str | Sequence[str],
    target_time: str = "as_of_time",
    observed_time: str = "observed_at",
    event_time: str | None = None,
    target_event_time: str | None = None,
    suffix: str = "_observation",
) -> pd.DataFrame:
    """Join each target to the latest observation eligible at its cutoff.

    If event columns are supplied, observations from the target event or later
    are rejected in addition to the observed-time cutoff.
    """
    keys = [by] if isinstance(by, str) else list(by)
    required_left = set(keys + [target_time])
    required_right = set(keys + [observed_time])
    if target_event_time:
        required_left.add(target_event_time)
    if event_time:
        required_right.add(event_time)
    if missing := required_left - set(targets):
        raise KeyError(f"Targets missing columns: {sorted(missing)}")
    if missing := required_right - set(observations):
        raise KeyError(f"Observations missing columns: {sorted(missing)}")

    left = targets.copy()
    right = observations.copy()
    left[target_time] = pd.to_datetime(left[target_time], utc=True)
    right[observed_time] = pd.to_datetime(right[observed_time], utc=True)
    if target_event_time:
        left[target_event_time] = pd.to_datetime(left[target_event_time], utc=True)
    if event_time:
        right[event_time] = pd.to_datetime(right[event_time], utc=True)
    left["__order"] = range(len(left))
    left = left.sort_values([target_time, *keys])
    right = right.sort_values([observed_time, *keys])
    joined = pd.merge_asof(
        left,
        right,
        left_on=target_time,
        right_on=observed_time,
        by=keys,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", suffix),
    )
    invalid = joined[observed_time].notna() & (joined[observed_time] > joined[target_time])
    if invalid.any():
        raise AssertionError("Point-in-time join selected future observations")
    if event_time and target_event_time:
        invalid_event = joined[event_time].notna() & (
            joined[event_time] >= joined[target_event_time]
        )
        if invalid_event.any():
            raise ValueError("Point-in-time join selected target or future event data")
    return joined.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def shifted_rolling_features(
    games: pd.DataFrame,
    *,
    entity_col: str,
    event_time_col: str,
    value_cols: Sequence[str],
    windows: Sequence[int] = (5, 15, 30),
    half_lives: Sequence[int] = (5, 15, 30),
) -> pd.DataFrame:
    ordered = games.sort_values([entity_col, event_time_col]).copy()
    grouped = ordered.groupby(entity_col, sort=False)
    for value in value_cols:
        prior = grouped[value].shift(1)
        for window in windows:
            ordered[f"{value}_mean_{window}"] = (
                prior.groupby(ordered[entity_col])
                .rolling(window, min_periods=max(2, window // 3))
                .mean()
                .reset_index(level=0, drop=True)
            )
        for half_life in half_lives:
            ordered[f"{value}_ewm_h{half_life}"] = prior.groupby(ordered[entity_col]).transform(
                lambda series, h=half_life: series.ewm(
                    halflife=h, min_periods=3, adjust=False
                ).mean()
            )
    return ordered


def assert_feature_watermarks(frame: pd.DataFrame) -> None:
    required = {"as_of_time", "source_max_observed_at"}
    if missing := required - set(frame):
        raise KeyError(f"Missing watermark columns: {sorted(missing)}")
    future = pd.to_datetime(frame["source_max_observed_at"], utc=True) > pd.to_datetime(
        frame["as_of_time"], utc=True
    )
    if future.any():
        sample = frame.loc[future].head(5)
        raise ValueError(f"Feature leakage: {len(frame.loc[future])} future rows\n{sample}")
