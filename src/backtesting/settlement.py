"""Pure market settlement functions with explicit push and void outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.contracts.domain import Selection, SettlementResult


@dataclass(frozen=True)
class FinalScore:
    home_runs: int
    away_runs: int
    official: bool
    innings_played: int = 9
    void_market: bool = False
    listed_pitchers_started: bool = True


@dataclass(frozen=True)
class MarketRule:
    rule_id: str
    require_official: bool = True
    min_innings: int = 0
    listed_pitchers_action: bool = False
    include_extra_innings: bool = True


def _is_void(score: FinalScore, rule: MarketRule | None) -> bool:
    if score.void_market:
        return True
    if rule is None:
        return not score.official
    if rule.require_official and not score.official:
        return True
    if score.innings_played < rule.min_innings:
        return True
    return rule.listed_pitchers_action and not score.listed_pitchers_started


def settle_moneyline(
    selection: Selection,
    score: FinalScore,
    rule: MarketRule | None = None,
) -> SettlementResult:
    if _is_void(score, rule):
        return SettlementResult.VOID
    if selection not in {Selection.HOME, Selection.AWAY}:
        raise ValueError("Moneyline requires home or away selection")
    if score.home_runs == score.away_runs:
        return SettlementResult.PUSH
    home_won = score.home_runs > score.away_runs
    selected_won = home_won if selection == Selection.HOME else not home_won
    return SettlementResult.WIN if selected_won else SettlementResult.LOSS


def settle_run_line(
    selection: Selection,
    point: Decimal,
    score: FinalScore,
    rule: MarketRule | None = None,
) -> SettlementResult:
    if _is_void(score, rule):
        return SettlementResult.VOID
    if selection == Selection.HOME:
        value = Decimal(score.home_runs - score.away_runs) + point
    elif selection == Selection.AWAY:
        value = Decimal(score.away_runs - score.home_runs) + point
    else:
        raise ValueError("Run line requires home or away selection")
    return _settle_signed(value)


def settle_total(
    selection: Selection,
    point: Decimal,
    score: FinalScore,
    rule: MarketRule | None = None,
) -> SettlementResult:
    if _is_void(score, rule):
        return SettlementResult.VOID
    actual = Decimal(score.home_runs + score.away_runs)
    if selection == Selection.OVER:
        return _settle_signed(actual - point)
    if selection == Selection.UNDER:
        return _settle_signed(point - actual)
    raise ValueError("Total requires over or under selection")


def _settle_signed(value: Decimal) -> SettlementResult:
    if value > 0:
        return SettlementResult.WIN
    if value < 0:
        return SettlementResult.LOSS
    return SettlementResult.PUSH


def profit_loss(result: SettlementResult, stake: Decimal, price_decimal: Decimal) -> Decimal:
    if stake < 0 or price_decimal <= 1:
        raise ValueError("Invalid stake or decimal price")
    if result == SettlementResult.WIN:
        return (stake * (price_decimal - Decimal("1"))).quantize(Decimal("0.01"))
    if result == SettlementResult.LOSS:
        return -stake
    return Decimal("0")
