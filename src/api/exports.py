"""Canonical CSV/JSON exports with one explicit schema."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

EXPORT_COLUMNS = [
    "prediction_id",
    "decision_id",
    "game_id",
    "game_date",
    "market_id",
    "selection",
    "bookmaker_id",
    "point",
    "price_decimal",
    "price_american",
    "quote_time",
    "probability",
    "probability_low",
    "probability_high",
    "break_even_probability",
    "edge",
    "expected_value",
    "recommended_stake",
    "action",
    "reason_codes",
    "snapshot_id",
    "model_run_id",
    "policy_version",
    "quality_status",
]


def validate_export(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := set(EXPORT_COLUMNS) - set(frame):
        raise ValueError(f"Canonical export missing columns: {sorted(missing)}")
    result = frame.loc[:, EXPORT_COLUMNS].copy()
    required = [
        "game_date",
        "market_id",
        "selection",
        "bookmaker_id",
        "price_decimal",
        "quote_time",
        "snapshot_id",
        "model_run_id",
    ]
    nulls = [column for column in required if result[column].isna().any()]
    if nulls:
        raise ValueError(f"Canonical export has null required fields: {nulls}")
    return result


def write_exports(frame: pd.DataFrame, *, json_path: Path, csv_path: Path) -> None:
    result = validate_export(frame)
    for path in (json_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(result.to_dict("records"), indent=2, default=str) + "\n", encoding="utf-8"
    )
    result.to_csv(temporary_csv, index=False)
    os.replace(temporary_json, json_path)
    os.replace(temporary_csv, csv_path)
