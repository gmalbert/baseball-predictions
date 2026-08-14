"""Read-only UI data access; pages never load models or join raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")
