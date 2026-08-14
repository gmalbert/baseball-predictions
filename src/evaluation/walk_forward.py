"""Purged chronological train/calibration/test folds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_index: pd.Index
    calibration_index: pd.Index
    test_index: pd.Index
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_folds(
    frame: pd.DataFrame,
    *,
    date_col: str,
    first_test_year: int,
    calibration_days: int = 60,
    embargo_days: int = 1,
    test_month: int = 7,
) -> list[Fold]:
    if date_col not in frame:
        raise KeyError(date_col)
    dates = pd.to_datetime(frame[date_col], utc=True)
    folds: list[Fold] = []
    for year in sorted(year for year in dates.dt.year.unique() if year >= first_test_year):
        test_start = pd.Timestamp(year=year, month=test_month, day=1, tz="UTC")
        test_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        calibration_end = test_start - timedelta(days=embargo_days)
        calibration_start = calibration_end - timedelta(days=calibration_days)
        train_end = calibration_start - timedelta(days=embargo_days)
        train = frame.index[dates < train_end]
        calibration = frame.index[(dates >= calibration_start) & (dates < calibration_end)]
        test = frame.index[(dates >= test_start) & (dates < test_end)]
        if min(len(train), len(calibration), len(test)) > 0:
            folds.append(Fold(train, calibration, test, test_start, test_end))
    if not folds:
        raise ValueError("No non-empty chronological folds produced")
    return folds


def assert_disjoint(fold: Fold) -> None:
    train, calibration, test = map(set, (fold.train_index, fold.calibration_index, fold.test_index))
    if train & calibration or train & test or calibration & test:
        raise ValueError("Walk-forward fold partitions overlap")
