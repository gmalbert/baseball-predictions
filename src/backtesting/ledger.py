"""Idempotent decision/execution/settlement storage and bankroll reconciliation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.backtesting.settlement import (
    FinalScore,
    MarketRule,
    profit_loss,
    settle_moneyline,
    settle_run_line,
    settle_total,
)
from src.contracts.domain import Decision, Execution, Quote, Settlement


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


class LedgerRepository:
    """Small-file repository used until DuckDB/object storage is promoted."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def append(self, name: str, rows: list[BaseModel], *, key: str | list[str]) -> Path:
        if not rows:
            return self._path(name)
        new = pd.DataFrame([row.model_dump(mode="json") for row in rows])
        path = self._path(name)
        existing = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=new.columns)
        combined = pd.concat([existing, new], ignore_index=True)
        # An identical retry is idempotent. Conflicting records with the same key fail closed.
        keys = [key] if isinstance(key, str) else key
        conflicts = combined.groupby(keys, dropna=False).filter(
            lambda group: len(group.astype(str).drop_duplicates()) > 1
        )
        if not conflicts.empty:
            sample = conflicts[keys].drop_duplicates().head(10).to_dict("records")
            raise ValueError(f"Conflicting ledger records for {keys}: {sample}")
        combined = combined.drop_duplicates(keys, keep="first")
        _atomic_parquet(combined, path)
        return path

    def decisions(self, rows: list[Decision]) -> Path:
        return self.append("decisions", rows, key="decision_id")

    def executions(self, rows: list[Execution]) -> Path:
        return self.append("executions", rows, key="execution_id")

    def settlements(self, rows: list[Settlement]) -> Path:
        return self.append("settlements", rows, key=["execution_id", "settlement_version"])

    def read(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


@dataclass(frozen=True)
class SettlementInput:
    execution: Execution
    decision: Decision
    quote: Quote
    score: FinalScore
    rule: MarketRule


def settle_execution(
    item: SettlementInput, *, source: str, source_reference: str | None = None
) -> Settlement:
    market_id = item.decision.market_id
    if item.execution.status not in {"accepted", "partially_filled"}:
        raise ValueError("Only accepted executions can settle")
    if item.decision.selection != item.quote.selection:
        raise ValueError("Decision and quote selections differ")
    if market_id.startswith("moneyline"):
        result = settle_moneyline(item.decision.selection, item.score, item.rule)
    elif market_id.startswith("run_line"):
        point = item.execution.accepted_point
        if point is None:
            raise ValueError("Run-line execution has no accepted point")
        result = settle_run_line(item.decision.selection, point, item.score, item.rule)
    elif market_id.startswith("total"):
        point = item.execution.accepted_point
        if point is None:
            raise ValueError("Total execution has no accepted point")
        result = settle_total(item.decision.selection, point, item.score, item.rule)
    else:
        raise KeyError(f"Unsupported settlement market: {market_id}")
    return Settlement(
        execution_id=item.execution.execution_id,
        settled_at=datetime.now(UTC),
        result=result,
        profit_loss=profit_loss(
            result, item.execution.stake, item.execution.accepted_price_decimal
        ),
        settlement_rule=item.rule.rule_id,
        source=source,
        source_reference=source_reference,
        settlement_version=1,
    )


@dataclass(frozen=True)
class BankrollReconciliation:
    starting_bankroll: Decimal
    total_stake: Decimal
    profit_loss: Decimal
    ending_bankroll: Decimal
    accepted_executions: int
    settled_executions: int
    balanced: bool


def reconcile_bankroll(
    executions: list[Execution],
    settlements: list[Settlement],
    *,
    starting_bankroll: Decimal,
) -> BankrollReconciliation:
    accepted = [row for row in executions if row.status in {"accepted", "partially_filled"}]
    settlement_by_id = {row.execution_id: row for row in settlements}
    unknown = set(settlement_by_id) - {row.execution_id for row in accepted}
    if unknown:
        raise ValueError(f"Settlements reference unknown accepted executions: {sorted(unknown)}")
    total_stake = sum((row.stake for row in accepted), Decimal("0"))
    pnl = sum((row.profit_loss for row in settlements), Decimal("0"))
    ending = starting_bankroll + pnl
    balanced = len(settlement_by_id) == len(accepted)
    return BankrollReconciliation(
        starting_bankroll=starting_bankroll,
        total_stake=total_stake,
        profit_loss=pnl,
        ending_bankroll=ending,
        accepted_executions=len(accepted),
        settled_executions=len(settlements),
        balanced=balanced,
    )
