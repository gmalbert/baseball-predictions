"""YAML dataset-contract loading and supported rule validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api import types as pdt

from src.quality.gate import CheckResult


@dataclass(frozen=True)
class DatasetContract:
    dataset: str
    version: str
    primary_key: tuple[str, ...]
    partition_by: tuple[str, ...]
    required: dict[str, str]
    checks: tuple[dict[str, Any], ...]
    freshness: dict[str, str]


def load_contract(path: Path) -> DatasetContract:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetContract(
        dataset=raw["dataset"],
        version=str(raw["version"]),
        primary_key=tuple(raw.get("primary_key", ())),
        partition_by=tuple(raw.get("partition_by", ())),
        required=dict(raw.get("required", {})),
        checks=tuple(raw.get("checks", ())),
        freshness=dict(raw.get("freshness", {})),
    )


def validate_contract(frame: pd.DataFrame, contract: DatasetContract) -> list[CheckResult]:
    results = []
    missing = sorted(set(contract.required) - set(frame))
    results.append(
        CheckResult(
            "required_columns",
            not missing,
            "error",
            f"missing={missing}",
            f"present={sorted(contract.required)}",
            len(missing),
        )
    )
    for column, expected in contract.required.items():
        if column not in frame:
            continue
        series = frame[column]
        if expected == "string":
            passed = (
                pdt.is_string_dtype(series.dtype)
                or series.dropna().map(lambda value: isinstance(value, str)).all()
            )
        elif expected.startswith("timestamp"):
            converted = pd.to_datetime(series, utc=True, errors="coerce")
            passed = converted.notna().sum() == series.notna().sum()
        elif expected.startswith("float"):
            passed = pdt.is_numeric_dtype(series.dtype)
        elif expected.startswith("int"):
            numeric = pd.to_numeric(series, errors="coerce")
            passed = (
                numeric.notna().sum() == series.notna().sum() and (numeric.dropna() % 1 == 0).all()
            )
        elif expected == "bool":
            passed = pdt.is_bool_dtype(series.dtype)
        elif expected.startswith("decimal"):
            passed = (
                series.dropna().map(lambda value: isinstance(value, (Decimal, int, float))).all()
            )
        elif expected.startswith("list"):
            passed = series.dropna().map(lambda value: isinstance(value, (list, tuple))).all()
        else:
            passed = True
        results.append(
            CheckResult(
                f"dtype:{column}",
                bool(passed),
                "error",
                str(series.dtype),
                expected,
                0 if passed else int(series.notna().sum()),
            )
        )
    if not missing and contract.primary_key:
        duplicates = int(frame.duplicated(list(contract.primary_key), keep=False).sum())
        results.append(
            CheckResult(
                "primary_key",
                duplicates == 0,
                "error",
                f"duplicate_rows={duplicates}",
                "duplicate_rows=0",
                duplicates,
            )
        )
    for check in contract.checks:
        if "unique" in check:
            columns = list(check["unique"])
            duplicates = (
                int(frame.duplicated(columns, keep=False).sum())
                if set(columns) <= set(frame)
                else len(frame)
            )
            results.append(
                CheckResult(
                    f"unique:{','.join(columns)}",
                    duplicates == 0,
                    "error",
                    f"duplicate_rows={duplicates}",
                    "duplicate_rows=0",
                    duplicates,
                )
            )
        elif "expression" in check:
            expression = str(check["expression"])
            try:
                valid = frame.eval(expression).fillna(False)
                invalid = int((~valid).sum())
            except (KeyError, ValueError, TypeError, SyntaxError):
                invalid = len(frame)
            results.append(
                CheckResult(
                    f"expression:{expression}",
                    invalid == 0,
                    "error",
                    f"invalid_rows={invalid}",
                    expression,
                    invalid,
                )
            )
    failures = [row for row in results if not row.passed and row.severity == "error"]
    if failures:
        raise ValueError(
            "Contract failed: " + "; ".join(f"{row.name} {row.observed}" for row in failures)
        )
    return results
