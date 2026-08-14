"""Detect inserts, updates, and deletes without destroying prior releases."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def _row_hash(row: pd.Series, value_columns: list[str]) -> str:
    value = json.dumps(
        {column: row[column] for column in value_columns}, sort_keys=True, default=str
    )
    return hashlib.sha256(value.encode()).hexdigest()


def compare_revisions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    if not set(key_columns) <= set(previous) or not set(key_columns) <= set(current):
        raise KeyError("Revision keys are missing")
    value_columns = sorted((set(previous) | set(current)) - set(key_columns))
    before = previous.copy()
    after = current.copy()
    for column in value_columns:
        if column not in before:
            before[column] = None
        if column not in after:
            after[column] = None
    before["prior_hash"] = before.apply(_row_hash, axis=1, value_columns=value_columns)
    after["current_hash"] = after.apply(_row_hash, axis=1, value_columns=value_columns)
    joined = before[key_columns + ["prior_hash"]].merge(
        after[key_columns + ["current_hash"]],
        on=key_columns,
        how="outer",
        indicator=True,
    )
    joined["change_type"] = (
        joined["_merge"]
        .map({"left_only": "delete", "right_only": "insert", "both": "unchanged"})
        .astype(str)
    )
    joined.loc[
        (joined["_merge"] == "both") & (joined["prior_hash"] != joined["current_hash"]),
        "change_type",
    ] = "update"
    return (
        joined[joined["change_type"] != "unchanged"].drop(columns="_merge").reset_index(drop=True)
    )
