# src/evaluation/profitability.py
"""Profitability and ROI analysis for betting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtester import BacktestResult, BetResult


def profitability_report(result: BacktestResult) -> pd.DataFrame:
    """Detailed profitability breakdown by confidence tier.

    Args:
        result: Completed :class:`BacktestResult` from a backtest run.

    Returns:
        DataFrame with one row per confidence tier (high / medium / low).
    """
    if not result.bets:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "date": b.date,
        "pick_type": b.pick_type,
        "confidence": b.confidence,
        "confidence_score": b.confidence_score,
        "edge": b.edge,
        "odds": b.american_odds,
        "result": b.result,
        "profit": b.profit_units,
    } for b in result.bets])

    tiers = df.groupby("confidence").agg(
        total_bets=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        total_profit=("profit", "sum"),
        avg_odds=("odds", "mean"),
        avg_edge=("edge", "mean"),
    ).reset_index()

    tiers["win_rate"] = tiers["wins"] / (tiers["wins"] + tiers["losses"])
    tiers["roi"] = tiers["total_profit"] / tiers["total_bets"]

    print(f"\n=== Profitability Report: {result.model_name} ({result.pick_type}) ===")
    print(f"Period: {result.period}")
    print(
        f"\nOverall: {result.total_bets} bets, {result.win_rate:.1%} win rate, "
        f"{result.total_units:+.2f} units, {result.roi:.1%} ROI"
    )
    print(f"Max Drawdown: {result.max_drawdown:.2f} units\n")
    print(tiers.to_string(index=False))

    return tiers


def cumulative_profit_data(result: BacktestResult) -> dict:
    """Return data for a cumulative profit line chart.

    Returns:
        Dict with ``dates``, ``cumulative_profit``, ``total_bets``,
        and ``final_profit``.
    """
    profits = [b.profit_units for b in result.bets]
    dates = [str(b.date) for b in result.bets]
    cumulative = np.cumsum(profits).tolist()

    return {
        "dates": dates,
        "cumulative_profit": cumulative,
        "total_bets": len(profits),
        "final_profit": cumulative[-1] if cumulative else 0,
    }


def monthly_breakdown(result: BacktestResult) -> pd.DataFrame:
    """Month-by-month performance breakdown.

    Returns:
        DataFrame with columns: month, bets, wins, losses, profit,
        win_rate, roi.
    """
    if not result.bets:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "date": b.date,
        "profit": b.profit_units,
        "result": b.result,
    } for b in result.bets])

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    monthly = df.groupby("month").agg(
        bets=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        profit=("profit", "sum"),
    ).reset_index()

    monthly["win_rate"] = monthly["wins"] / (monthly["wins"] + monthly["losses"])
    monthly["roi"] = monthly["profit"] / monthly["bets"]

    return monthly


def edge_filter_analysis(result: BacktestResult) -> pd.DataFrame:
    """Show how performance changes at different minimum edge thresholds.

    Helps determine the optimal minimum edge to filter bets.

    Returns:
        DataFrame with one row per edge threshold tested.
    """
    thresholds = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    rows = []

    for threshold in thresholds:
        filtered = [b for b in result.bets if b.edge >= threshold]
        if not filtered:
            continue

        wins = sum(1 for b in filtered if b.result == "win")
        losses = sum(1 for b in filtered if b.result == "loss")
        profit = sum(b.profit_units for b in filtered)
        decided = wins + losses

        rows.append({
            "min_edge": threshold,
            "total_bets": len(filtered),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / decided, 4) if decided > 0 else 0.0,
            "total_profit": round(profit, 2),
            "roi": round(profit / len(filtered), 4),
        })

    report = pd.DataFrame(rows)
    print("\n=== Edge Filter Analysis ===")
    print(report.to_string(index=False))
    return report
