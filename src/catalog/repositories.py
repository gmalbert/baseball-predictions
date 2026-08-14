"""Parquet-backed canonical repositories with exact selection/cutoff queries."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from src.contracts.domain import Prediction, Quote

RecordT = TypeVar("RecordT", bound=BaseModel)


def write_records(path: Path, rows: Iterable[RecordT]) -> int:
    records = [row.model_dump(mode="json") for row in rows]
    if not records:
        raise ValueError("Refusing to write an empty canonical dataset")
    frame = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return len(frame)


class ParquetQuoteRepository:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)

    def latest_actionable(
        self,
        game_ids: Sequence[str],
        as_of: datetime,
        max_age_seconds: int,
    ) -> list[Quote]:
        if not self.paths:
            return []
        frames = [pd.read_parquet(path) for path in self.paths if path.is_file()]
        if not frames:
            return []
        frame = pd.concat(frames, ignore_index=True)
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        cutoff = pd.Timestamp(as_of)
        eligible = frame[
            frame["game_id"].isin(game_ids)
            & (frame["observed_at"] <= cutoff)
            & ((cutoff - frame["observed_at"]).dt.total_seconds() <= max_age_seconds)
            & ~frame["is_suspended"].astype(bool)
            & frame["is_actionable"].astype(bool)
        ].copy()
        eligible = eligible.sort_values("observed_at").drop_duplicates(
            ["game_id", "bookmaker_id", "market_id", "selection", "point"], keep="last"
        )
        return [Quote.model_validate(record) for record in eligible.to_dict("records")]


class ParquetPredictionRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, predictions: list[Prediction], *, target_date: str, model_run_id: str) -> Path:
        path = (
            self.root
            / f"target_date={target_date}"
            / f"model_run_id={model_run_id}"
            / "predictions.parquet"
        )
        write_records(path, predictions)
        return path
