# src/evaluation/backtester.py
"""Walk-forward backtesting engine for betting models.

Critical: Uses time-series splits to prevent look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BetResult:
    """Single bet outcome."""

    game_id: int | str
    date: date
    pick_type: str    # 'underdog', 'spread', 'over_under'
    pick_value: str   # 'NYY +150', 'Over 8.5', etc.
    predicted_prob: float
    confidence_score: float
    confidence: str   # 'high', 'medium', 'low'
    edge: float
    american_odds: int
    result: str       # 'win', 'loss', 'push'
    profit_units: float  # +1.50, -1.00, 0.00


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""

    model_name: str
    pick_type: str
    period: str
    bets: list[BetResult] = field(default_factory=list)

    @property
    def total_bets(self) -> int:
        return len(self.bets)

    @property
    def wins(self) -> int:
        return sum(1 for b in self.bets if b.result == "win")

    @property
    def losses(self) -> int:
        return sum(1 for b in self.bets if b.result == "loss")

    @property
    def pushes(self) -> int:
        return sum(1 for b in self.bets if b.result == "push")

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0.0

    @property
    def total_units(self) -> float:
        return sum(b.profit_units for b in self.bets)

    @property
    def roi(self) -> float:
        return self.total_units / self.total_bets if self.total_bets > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough decline in cumulative units."""
        cumulative = np.cumsum([b.profit_units for b in self.bets])
        if len(cumulative) == 0:
            return 0.0
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(drawdown.max())

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "pick_type": self.pick_type,
            "period": self.period,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "win_rate": round(self.win_rate, 4),
            "total_units": round(self.total_units, 2),
            "roi": round(self.roi, 4),
            "max_drawdown": round(self.max_drawdown, 2),
        }


def calculate_profit(american_odds: int, result: str) -> float:
    """Calculate profit for a 1-unit bet.

    Win:  +profit based on odds
    Loss: -1.00
    Push:  0.00
    """
    if result == "push":
        return 0.0
    if result == "loss":
        return -1.0
    # result == "win"
    if american_odds > 0:
        return american_odds / 100
    return 100 / abs(american_odds)


def walk_forward_backtest(
    features_df: pd.DataFrame,
    train_fn: Callable,
    predict_fn: Callable,
    target_col: str,
    odds_col: str,
    pick_type: str,
    model_name: str,
    min_edge: float = 0.0,
    min_confidence: float = 0.0,
    train_window_games: int = 1500,
    test_window_games: int = 200,
    step_size: int = 100,
) -> BacktestResult:
    """Walk-forward backtesting with rolling retrain windows.

    1. Train on games [start : train_end]
    2. Predict on games [train_end : train_end + test_window]
    3. Slide forward by ``step_size`` and repeat

    Args:
        features_df:          Feature matrix sorted by date.
        train_fn:             ``(X_train, y_train) → model``
        predict_fn:           ``(model, X_test) → probabilities array``
        target_col:           Column with actual outcomes (0/1).
        odds_col:             Column with American odds for bet sizing.
        pick_type:            ``'underdog'``, ``'spread'``, or ``'over_under'``.
        model_name:           Identifier string.
        min_edge:             Minimum edge to place a bet.
        min_confidence:       Minimum confidence score to place a bet.
        train_window_games:   Number of games in training window.
        test_window_games:    Number of games in test window.
        step_size:            Slide forward by this many games each iteration.
    """
    df = features_df.sort_values("date").reset_index(drop=True)
    drop_cols = [target_col, "game_id", "date"]
    all_bets: list[BetResult] = []

    start = 0
    while start + train_window_games + test_window_games <= len(df):
        train_end = start + train_window_games
        test_end = train_end + test_window_games

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df.drop(columns=drop_cols, errors="ignore")
        y_train = train_df[target_col]
        model = train_fn(X_train, y_train)

        X_test = test_df.drop(columns=drop_cols, errors="ignore")
        probs = predict_fn(model, X_test)

        for i, (idx, row) in enumerate(test_df.iterrows()):
            prob = float(probs[i])
            odds = int(row.get(odds_col, -110))

            # Edge = our probability minus market implied probability
            if odds < 0:
                implied = abs(odds) / (abs(odds) + 100)
            else:
                implied = 100 / (odds + 100)
            edge = prob - implied

            # Confidence score
            prob_strength = abs(prob - 0.5) * 2
            edge_strength = max(0.0, min(edge / 0.15, 1.0))
            conf_score = 0.45 * prob_strength + 0.55 * edge_strength

            if edge < min_edge or conf_score < min_confidence:
                continue

            actual = int(row[target_col])
            predicted_pick = 1 if prob >= 0.5 else 0
            result = "win" if actual == predicted_pick else "loss"
            profit = calculate_profit(odds, result)

            if conf_score >= 0.65:
                conf_label = "high"
            elif conf_score >= 0.35:
                conf_label = "medium"
            else:
                conf_label = "low"

            all_bets.append(BetResult(
                game_id=row.get("game_id", idx),
                date=row.get("date"),
                pick_type=pick_type,
                pick_value=f"Pred={prob:.3f}",
                predicted_prob=prob,
                confidence_score=conf_score,
                confidence=conf_label,
                edge=edge,
                american_odds=odds,
                result=result,
                profit_units=profit,
            ))

        start += step_size

    period = f"{df['date'].min()} to {df['date'].max()}"
    return BacktestResult(
        model_name=model_name,
        pick_type=pick_type,
        period=period,
        bets=all_bets,
    )
