"""Performance attribution by skill, execution, market, and data tier."""

from __future__ import annotations

import pandas as pd


def performance_attribution(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {
        "profit_loss",
        "stake",
        "model_probability",
        "market_probability",
        "execution_price",
        "decision_price",
    }
    if missing := required - set(ledger):
        raise KeyError(f"Ledger missing attribution columns: {sorted(missing)}")
    frame = ledger.copy()
    frame["probability_skill"] = frame["model_probability"] - frame["market_probability"]
    frame["line_shopping_value"] = frame["execution_price"] - frame.get(
        "consensus_price", frame["decision_price"]
    )
    frame["execution_slippage"] = frame["execution_price"] - frame["decision_price"]
    frame["roi"] = frame["profit_loss"] / frame["stake"].where(frame["stake"] != 0)
    dimensions = [
        column
        for column in (
            "market_id",
            "bookmaker_id",
            "season",
            "edge_bucket",
            "uncertainty_bucket",
            "quality_tier",
        )
        if column in frame
    ]
    if not dimensions:
        dimensions = ["market_id"] if "market_id" in frame else []
    if not dimensions:
        frame["dimension"] = "all"
        dimensions = ["dimension"]
    return frame.groupby(dimensions, dropna=False, as_index=False).agg(
        bets=("profit_loss", "size"),
        total_stake=("stake", "sum"),
        profit_loss=("profit_loss", "sum"),
        roi=("roi", "mean"),
        probability_skill=("probability_skill", "mean"),
        line_shopping_value=("line_shopping_value", "mean"),
        execution_slippage=("execution_slippage", "mean"),
    )
