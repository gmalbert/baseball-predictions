"""Settle pending canonical executions from official schedule results."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.ledger import LedgerRepository, SettlementInput, settle_execution
from src.backtesting.settlement import FinalScore, MarketRule
from src.contracts.domain import Decision, Execution, Quote

FINAL_STATUSES = {"Final", "Game Over", "Completed Early"}


def settle_pending_from_schedule(
    *,
    ledger: LedgerRepository,
    quote_path: Path,
    schedule: pd.DataFrame,
    target_date: date,
    source: str = "mlb_statsapi",
) -> int:
    executions_frame = ledger.read("executions")
    decisions_frame = ledger.read("decisions")
    settlements_frame = ledger.read("settlements")
    if executions_frame.empty or decisions_frame.empty or not quote_path.is_file():
        return 0
    settled_ids = set(settlements_frame.get("execution_id", pd.Series(dtype=str)))
    pending = executions_frame[
        executions_frame["status"].isin(["accepted", "partially_filled"])
        & ~executions_frame["execution_id"].isin(settled_ids)
    ]
    if pending.empty:
        return 0
    quotes_frame = pd.read_parquet(quote_path)
    decisions = {
        row["decision_id"]: Decision.model_validate(row)
        for row in decisions_frame.to_dict("records")
    }
    quotes = {row["quote_id"]: Quote.model_validate(row) for row in quotes_frame.to_dict("records")}
    schedule_by_game = {str(row["game_id"]): row for row in schedule.to_dict("records")}
    rows = []
    for raw in pending.to_dict("records"):
        execution = Execution.model_validate(raw)
        decision = decisions.get(execution.decision_id)
        quote = quotes.get(execution.quote_id)
        game = schedule_by_game.get(decision.game_id if decision else "")
        if (
            decision is None
            or quote is None
            or game is None
            or game.get("status") not in FINAL_STATUSES
        ):
            continue
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            continue
        score = FinalScore(
            home_runs=int(game["home_score"]),
            away_runs=int(game["away_score"]),
            official=True,
            innings_played=int(game.get("innings_played", 9) or 9),
            listed_pitchers_started=bool(game.get("listed_pitchers_started", True)),
        )
        rule = MarketRule(
            rule_id=quote.action_rule or "default_full_game_v1",
            min_innings=5 if decision.market_id.startswith("total") else 0,
            listed_pitchers_action=bool(
                quote.action_rule and "listed" in quote.action_rule.lower()
            ),
        )
        rows.append(
            settle_execution(
                SettlementInput(execution, decision, quote, score, rule),
                source=source,
                source_reference=f"{target_date.isoformat()}:{decision.game_id}",
            )
        )
    ledger.settlements(rows)
    return len(rows)
