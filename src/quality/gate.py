"""Composable checks that fail publication on contract violations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    severity: str
    observed: str
    expected: str
    affected_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Check = Callable[[pd.DataFrame], CheckResult]


def require_columns(*columns: str) -> Check:
    def check(frame: pd.DataFrame) -> CheckResult:
        missing = sorted(set(columns) - set(frame))
        return CheckResult(
            name="required_columns",
            passed=not missing,
            severity="error",
            observed=f"missing={missing}",
            expected=f"present={list(columns)}",
            affected_rows=len(missing),
        )

    return check


def unique_key(*columns: str) -> Check:
    def check(frame: pd.DataFrame) -> CheckResult:
        missing = set(columns) - set(frame)
        if missing:
            return CheckResult(
                name=f"unique:{','.join(columns)}",
                passed=False,
                severity="error",
                observed=f"missing_columns={sorted(missing)}",
                expected="key columns present",
                affected_rows=len(missing),
            )
        duplicates = int(frame.duplicated(list(columns), keep=False).sum())
        return CheckResult(
            name=f"unique:{','.join(columns)}",
            passed=duplicates == 0,
            severity="error",
            observed=f"duplicate_rows={duplicates}",
            expected="duplicate_rows=0",
            affected_rows=duplicates,
        )

    return check


def no_future_observations(as_of: datetime, column: str = "observed_at") -> Check:
    def check(frame: pd.DataFrame) -> CheckResult:
        if column not in frame:
            return CheckResult(
                name="no_future_observations",
                passed=False,
                severity="error",
                observed=f"missing_column={column}",
                expected=f"column={column}",
                affected_rows=1,
            )
        timestamps = pd.to_datetime(frame[column], utc=True)
        count = int((timestamps > as_of).sum())
        return CheckResult(
            name="no_future_observations",
            passed=count == 0,
            severity="error",
            observed=f"future_rows={count}",
            expected="future_rows=0",
            affected_rows=count,
        )

    return check


def numeric_range(
    column: str, *, minimum: float | None = None, maximum: float | None = None
) -> Check:
    def check(frame: pd.DataFrame) -> CheckResult:
        if column not in frame:
            return CheckResult(
                name=f"range:{column}",
                passed=False,
                severity="error",
                observed="missing column",
                expected=f"{minimum} <= value <= {maximum}",
                affected_rows=1,
            )
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna()
        if minimum is not None:
            invalid |= values < minimum
        if maximum is not None:
            invalid |= values > maximum
        count = int(invalid.sum())
        return CheckResult(
            name=f"range:{column}",
            passed=count == 0,
            severity="error",
            observed=f"invalid_rows={count}",
            expected=f"{minimum} <= value <= {maximum}",
            affected_rows=count,
        )

    return check


def known_values(column: str, values: set[str]) -> Check:
    """Require every non-null value to be part of a declared vocabulary."""

    def check(frame: pd.DataFrame) -> CheckResult:
        if column not in frame:
            return CheckResult(
                name=f"known_values:{column}",
                passed=False,
                severity="error",
                observed="missing column",
                expected=f"values={sorted(values)}",
                affected_rows=1,
            )
        unknown = ~frame[column].isna() & ~frame[column].astype(str).isin(values)
        count = int(unknown.sum())
        observed = sorted(frame.loc[unknown, column].astype(str).unique().tolist())
        return CheckResult(
            name=f"known_values:{column}",
            passed=count == 0,
            severity="error",
            observed=f"unknown={observed}",
            expected=f"values={sorted(values)}",
            affected_rows=count,
        )

    return check


def max_missing(column: str, maximum_fraction: float) -> Check:
    """Enforce an explicit missingness budget instead of broad zero imputation."""
    if not 0 <= maximum_fraction <= 1:
        raise ValueError("maximum_fraction must be in [0, 1]")

    def check(frame: pd.DataFrame) -> CheckResult:
        if column not in frame:
            return CheckResult(
                name=f"missingness:{column}",
                passed=False,
                severity="error",
                observed="missing column",
                expected=f"fraction<={maximum_fraction}",
                affected_rows=1,
            )
        missing = int(frame[column].isna().sum())
        fraction = missing / len(frame) if len(frame) else 1.0
        return CheckResult(
            name=f"missingness:{column}",
            passed=fraction <= maximum_fraction,
            severity="error",
            observed=f"fraction={fraction:.6f}",
            expected=f"fraction<={maximum_fraction}",
            affected_rows=missing,
        )

    return check


def valid_decimal_odds(column: str = "decimal_odds") -> Check:
    """Validate executable decimal prices without inventing a conventional price."""
    return numeric_range(column, minimum=1.000001)


def schema_exact(*columns: str) -> Check:
    """Require an exact canonical column set for current/historical parity."""

    def check(frame: pd.DataFrame) -> CheckResult:
        expected = set(columns)
        actual = set(frame.columns)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        affected = len(missing) + len(extra)
        return CheckResult(
            name="schema_exact",
            passed=affected == 0,
            severity="error",
            observed=f"missing={missing},extra={extra}",
            expected=f"columns={list(columns)}",
            affected_rows=affected,
        )

    return check


def freshness(
    column: str, *, as_of: datetime, warn_after_seconds: int, fail_after_seconds: int
) -> Check:
    def check(frame: pd.DataFrame) -> CheckResult:
        if frame.empty or column not in frame:
            return CheckResult(
                name=f"freshness:{column}",
                passed=False,
                severity="error",
                observed="no timestamp",
                expected=f"age <= {fail_after_seconds}s",
                affected_rows=1,
            )
        latest = pd.to_datetime(frame[column], utc=True).max().to_pydatetime()
        age = (as_of - latest).total_seconds()
        severity = (
            "error"
            if age > fail_after_seconds
            else "warning"
            if age > warn_after_seconds
            else "info"
        )
        return CheckResult(
            name=f"freshness:{column}",
            passed=age <= fail_after_seconds,
            severity=severity,
            observed=f"age_seconds={age:.0f}",
            expected=f"age_seconds<={fail_after_seconds}",
            affected_rows=int(age > fail_after_seconds),
        )

    return check


def run_gate(frame: pd.DataFrame, checks: list[Check]) -> list[CheckResult]:
    results = [check(frame) for check in checks]
    failures = [result for result in results if not result.passed and result.severity == "error"]
    if failures:
        summary = "; ".join(f"{item.name}: {item.observed}" for item in failures)
        raise ValueError(f"Data-quality gate failed: {summary}")
    return results
