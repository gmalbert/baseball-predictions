"""True market labels derived from exact selection points and official results."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtesting.settlement import (
    FinalScore,
    MarketRule,
    settle_moneyline,
    settle_run_line,
    settle_total,
)
from src.contracts.domain import Selection, SettlementResult


@dataclass(frozen=True)
class MarketLabel:
    game_id: str
    market_id: str
    selection: Selection
    point: Decimal | None
    result: SettlementResult
    eligible: bool
    reason: str | None = None


def label_market(
    *,
    game_id: str,
    market_id: str,
    selection: Selection,
    point: Decimal | None,
    score: FinalScore,
    rule: MarketRule | None = None,
) -> MarketLabel:
    if market_id.startswith("moneyline"):
        result = settle_moneyline(selection, score, rule)
    elif market_id.startswith("run_line"):
        if point is None:
            raise ValueError("Run-line label requires point")
        result = settle_run_line(selection, point, score, rule)
    elif market_id.startswith("total"):
        if point is None:
            raise ValueError("Total label requires point")
        result = settle_total(selection, point, score, rule)
    else:
        raise KeyError(f"Unsupported market: {market_id}")
    return MarketLabel(
        game_id=game_id,
        market_id=market_id,
        selection=selection,
        point=point,
        result=result,
        eligible=result != SettlementResult.VOID,
        reason="market_void" if result == SettlementResult.VOID else None,
    )
