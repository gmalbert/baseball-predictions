from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.backtesting.settlement import (
    FinalScore,
    MarketRule,
    profit_loss,
    settle_run_line,
    settle_total,
)
from src.contracts.domain import Prediction, Quote, Selection, SettlementResult
from src.decisions.policy import Policy, evaluate
from src.markets.pricing import (
    american_to_decimal,
    decimal_to_american,
    devig_additive,
    devig_ensemble,
    devig_multiplicative,
    devig_power,
    devig_shin,
    expected_value,
    kelly_fraction,
)

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


def prediction(selection: Selection, probability: float, low: float | None = None) -> Prediction:
    return Prediction(
        prediction_id=f"p-{selection}",
        snapshot_id="s",
        game_id="g",
        model_run_id="m",
        market_id="moneyline_full_game",
        selection=selection,
        probability_raw=probability,
        probability=probability,
        probability_low=low,
        probability_high=min(1, probability + 0.05),
        predicted_at=NOW,
        feature_row_hash="h",
    )


def quote(selection: Selection, price: int) -> Quote:
    return Quote(
        quote_id=f"q-{selection}",
        game_id="g",
        bookmaker_id="book",
        market_id="moneyline_full_game",
        selection=selection,
        price_decimal=Decimal(str(american_to_decimal(price))),
        price_american=price,
        observed_at=NOW,
        jurisdiction="NY",
    )


def test_market_math_and_devig_methods_are_coherent():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert decimal_to_american(american_to_decimal(+135)) == 135
    raw = [1 / american_to_decimal(-130), 1 / american_to_decimal(+115)]
    for method in (devig_multiplicative, devig_additive, devig_power, devig_shin):
        assert sum(method(raw)) == pytest.approx(1)
    assert sum(devig_ensemble(raw)) == pytest.approx(1)
    assert expected_value(0.60, 2.10) == pytest.approx(0.26)
    assert 0 < kelly_fraction(0.60, 2.10) < 1


def test_decision_matches_exact_side_and_abstains_on_uncertainty():
    away = evaluate(
        prediction(Selection.AWAY, 0.60, 0.58),
        quote(Selection.AWAY, +110),
        bankroll=Decimal("1000"),
        decided_at=NOW,
        policy=Policy(),
    )
    assert away.selection == Selection.AWAY
    assert away.fair_probability == 0.60
    assert away.action == "bet"
    uncertain = evaluate(
        prediction(Selection.AWAY, 0.60, 0.45),
        quote(Selection.AWAY, +110),
        bankroll=Decimal("1000"),
        decided_at=NOW,
        policy=Policy(),
    )
    assert uncertain.action == "abstain"
    assert "uncertainty_spans_break_even" in uncertain.reason_codes
    with pytest.raises(ValueError, match="do not match"):
        evaluate(
            prediction(Selection.AWAY, 0.60),
            quote(Selection.HOME, -120),
            bankroll=Decimal("1000"),
            decided_at=NOW,
            policy=Policy(),
        )


def test_integer_push_void_and_away_margin_settlement():
    score = FinalScore(home_runs=5, away_runs=4, official=True, innings_played=9)
    assert settle_total(Selection.OVER, Decimal("9"), score) == SettlementResult.PUSH
    assert settle_total(Selection.UNDER, Decimal("9"), score) == SettlementResult.PUSH
    assert settle_run_line(Selection.AWAY, Decimal("1.5"), score) == SettlementResult.WIN
    shortened = FinalScore(home_runs=3, away_runs=2, official=True, innings_played=4)
    assert (
        settle_total(Selection.OVER, Decimal("8"), shortened, MarketRule("five", min_innings=5))
        == SettlementResult.VOID
    )
    assert profit_loss(SettlementResult.WIN, Decimal("10"), Decimal("2.50")) == Decimal("15.00")
