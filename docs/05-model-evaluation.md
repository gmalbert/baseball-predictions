# 05 – Model Evaluation & Backtesting

How to measure whether the models are actually profitable, well-calibrated, and trustworthy.

---

## Key Metrics Explained

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Win Rate** | % of picks that won | > 52.4% at -110 juice to break even |
| **ROI** | Return on investment per unit wagered | > 0% (positive = profitable) |
| **Units Profit** | Total profit in units ($100 = 1 unit) | Positive and growing |
| **Brier Score** | Calibration: how close probabilities match reality | Lower is better (0 = perfect) |
| **Log Loss** | Penalizes confident wrong predictions heavily | Lower is better |
| **ROC AUC** | Discrimination: can the model rank good bets above bad? | > 0.55 is useful for betting |
| **CLV (Closing Line Value)** | Did our line beat the closing line? | Positive CLV = long-term edge |
| **Kelly Edge** | Optimal bet sizing based on edge | Positive = bet, Negative = pass |

---

## 1. Backtesting Framework

```python
# src/evaluation/backtester.py
"""Walk-forward backtesting engine for betting models.

Critical: Uses time-series splits to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from datetime import date


@dataclass
class BetResult:
    """Single bet outcome."""
    game_id: int
    date: date
    pick_type: str          # 'underdog', 'spread', 'over_under'
    pick_value: str         # 'NYY +150', 'Over 8.5', etc.
    predicted_prob: float
    confidence_score: float
    confidence: str         # 'high', 'medium', 'low'
    edge: float
    american_odds: int
    result: str             # 'win', 'loss', 'push'
    profit_units: float     # +1.50, -1.00, 0.00


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
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(drawdown.max()) if len(drawdown) > 0 else 0.0
    
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
    
    Win: +profit based on odds
    Loss: -1.00
    Push: 0.00
    """
    if result == "push":
        return 0.0
    elif result == "loss":
        return -1.0
    elif result == "win":
        if american_odds > 0:
            return american_odds / 100
        else:
            return 100 / abs(american_odds)
    return 0.0


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
    
    1. Train on games [0 : train_end]
    2. Predict on games [train_end : train_end + test_window]
    3. Slide forward by step_size and repeat
    
    Args:
        features_df: Feature matrix sorted by date
        train_fn: Function(X_train, y_train) → model
        predict_fn: Function(model, X_test) → probabilities
        target_col: Column with actual outcomes (0/1)
        odds_col: Column with American odds for bet sizing
        pick_type: 'underdog', 'spread', or 'over_under'
        model_name: Identifier string
        min_edge: Minimum edge to place a bet
        min_confidence: Minimum confidence score to place a bet
        train_window_games: Number of games in training window
        test_window_games: Number of games in test window
        step_size: Slide forward by this many games
    """
    df = features_df.sort_values("date").reset_index(drop=True)
    all_bets = []
    
    start = 0
    while start + train_window_games + test_window_games <= len(df):
        train_end = start + train_window_games
        test_end = train_end + test_window_games
        
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Train
        X_train = train_df.drop(columns=[target_col, "game_id", "date"], errors="ignore")
        y_train = train_df[target_col]
        model = train_fn(X_train, y_train)
        
        # Predict
        X_test = test_df.drop(columns=[target_col, "game_id", "date"], errors="ignore")
        probs = predict_fn(model, X_test)
        
        # Evaluate each game
        for i, (idx, row) in enumerate(test_df.iterrows()):
            prob = probs[i]
            odds = int(row.get(odds_col, -110))
            
            # Calculate edge
            implied = abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
            edge = prob - implied
            
            # Confidence
            prob_strength = abs(prob - 0.5) * 2
            edge_strength = max(0, min(edge / 0.15, 1))
            conf_score = 0.45 * prob_strength + 0.55 * edge_strength
            
            # Filter: only bet if edge and confidence meet threshold
            if edge < min_edge or conf_score < min_confidence:
                continue
            
            # Determine result
            actual = row[target_col]
            predicted_pick = 1 if prob >= 0.5 else 0
            
            if actual == predicted_pick:
                result = "win"
            else:
                result = "loss"
            
            profit = calculate_profit(odds, result)
            
            all_bets.append(BetResult(
                game_id=row.get("game_id", idx),
                date=row.get("date", None),
                pick_type=pick_type,
                pick_value=f"Pred={prob:.3f}",
                predicted_prob=prob,
                confidence_score=conf_score,
                confidence="high" if conf_score >= 0.65 else ("medium" if conf_score >= 0.35 else "low"),
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
```

---

## 2. Calibration Analysis

```python
# src/evaluation/calibration.py
"""Calibration analysis: are predicted probabilities accurate?"""

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "model",
) -> pd.DataFrame:
    """Generate calibration report showing predicted vs. actual rates.
    
    A well-calibrated model: when it says 60%, events happen ~60% of the time.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    
    # Bin the predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
            "count": int(mask.sum()),
            "avg_predicted": float(y_prob[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
            "calibration_error": float(abs(y_prob[mask].mean() - y_true[mask].mean())),
        })
    
    report = pd.DataFrame(rows)
    
    # Overall metrics
    brier = brier_score_loss(y_true, y_prob)
    ece = report["calibration_error"].mean()  # Expected Calibration Error
    
    print(f"\n=== Calibration Report: {model_name} ===")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(report.to_string(index=False))
    
    return report


def calibration_plot_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Return data needed to render a calibration plot on the frontend.
    
    The frontend can plot:
    - X axis: mean predicted probability per bin
    - Y axis: actual fraction of positives per bin
    - A diagonal line represents perfect calibration
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    
    return {
        "mean_predicted": mean_predicted.tolist(),
        "fraction_positive": fraction_positive.tolist(),
        "perfect_line": [[0, 0], [1, 1]],
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
```

---

## 3. Profitability Analysis

```python
# src/evaluation/profitability.py
"""Profitability and ROI analysis for betting models."""

import pandas as pd
import numpy as np
from .backtester import BacktestResult, BetResult


def profitability_report(result: BacktestResult) -> pd.DataFrame:
    """Detailed profitability breakdown by confidence tier."""
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
    print(f"\nOverall: {result.total_bets} bets, {result.win_rate:.1%} win rate, "
          f"{result.total_units:+.2f} units, {result.roi:.1%} ROI")
    print(f"Max Drawdown: {result.max_drawdown:.2f} units\n")
    print(tiers.to_string(index=False))
    
    return tiers


def cumulative_profit_data(result: BacktestResult) -> dict:
    """Data for a cumulative profit line chart."""
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
    """Month-by-month performance breakdown."""
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
    
    This helps determine the optimal minimum edge to filter bets.
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
        
        rows.append({
            "min_edge": threshold,
            "total_bets": len(filtered),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
            "total_profit": round(profit, 2),
            "roi": round(profit / len(filtered), 4) if filtered else 0,
        })
    
    report = pd.DataFrame(rows)
    print("\n=== Edge Filter Analysis ===")
    print(report.to_string(index=False))
    return report
```

---

## 4. Closing Line Value (CLV) Tracker

```python
# src/evaluation/clv.py
"""Closing Line Value (CLV) analysis.

CLV is the #1 indicator of long-term betting success.
If you consistently beat the closing line, you will be profitable.
"""

import pandas as pd
import numpy as np
from ..models.features import implied_probability


def calculate_clv(
    picks_df: pd.DataFrame,
    opening_odds_col: str = "opening_odds",
    closing_odds_col: str = "closing_odds",
) -> pd.DataFrame:
    """Calculate Closing Line Value for each pick.
    
    CLV = implied_prob(closing_line) - implied_prob(opening_line)
    
    Positive CLV means the line moved in your direction after you picked it,
    indicating you identified value before the market corrected.
    """
    df = picks_df.copy()
    
    df["opening_implied"] = df[opening_odds_col].apply(implied_probability)
    df["closing_implied"] = df[closing_odds_col].apply(implied_probability)
    df["clv"] = df["closing_implied"] - df["opening_implied"]
    
    # CLV in cents (easier to interpret)
    df["clv_cents"] = (df["clv"] * 100).round(1)
    
    return df


def clv_report(df: pd.DataFrame) -> dict:
    """Summarize CLV performance."""
    avg_clv = df["clv"].mean()
    pct_positive = (df["clv"] > 0).mean()
    
    report = {
        "total_picks": len(df),
        "avg_clv": round(avg_clv, 4),
        "avg_clv_cents": round(avg_clv * 100, 1),
        "pct_positive_clv": round(pct_positive, 3),
        "median_clv_cents": round(df["clv"].median() * 100, 1),
    }
    
    print("\n=== Closing Line Value Report ===")
    print(f"Avg CLV: {report['avg_clv_cents']:+.1f} cents")
    print(f"Positive CLV: {report['pct_positive_clv']:.1%} of picks")
    print(f"Median CLV: {report['median_clv_cents']:+.1f} cents")
    
    return report
```

---

## 5. Model Comparison Dashboard Data

```python
# src/evaluation/dashboard.py
"""Generate data structures for the model performance dashboard."""

import pandas as pd
from .backtester import BacktestResult
from .profitability import cumulative_profit_data, monthly_breakdown
from .calibration import calibration_plot_data


def generate_dashboard_data(
    backtest_results: dict[str, BacktestResult],
    y_true_map: dict[str, list],
    y_prob_map: dict[str, list],
) -> dict:
    """Generate all data needed for the frontend performance dashboard.
    
    Args:
        backtest_results: Dict of model_name → BacktestResult
        y_true_map: Dict of model_name → actual outcomes
        y_prob_map: Dict of model_name → predicted probabilities
    
    Returns:
        Dict structure ready for JSON serialization to the frontend
    """
    dashboard = {
        "leaderboard": [],
        "cumulative_charts": {},
        "calibration_charts": {},
        "monthly_tables": {},
    }
    
    for name, result in backtest_results.items():
        # Leaderboard row
        dashboard["leaderboard"].append(result.summary())
        
        # Cumulative profit chart data
        dashboard["cumulative_charts"][name] = cumulative_profit_data(result)
        
        # Monthly breakdown
        monthly = monthly_breakdown(result)
        dashboard["monthly_tables"][name] = monthly.to_dict(orient="records")
        
        # Calibration chart
        if name in y_true_map and name in y_prob_map:
            import numpy as np
            dashboard["calibration_charts"][name] = calibration_plot_data(
                np.array(y_true_map[name]),
                np.array(y_prob_map[name]),
            )
    
    # Sort leaderboard by ROI descending
    dashboard["leaderboard"].sort(key=lambda x: x["roi"], reverse=True)
    
    return dashboard
```

---

## 6. Running the Full Evaluation

```python
# scripts/run_evaluation.py
"""Run full model evaluation pipeline."""

from src.models.underdog_model import train_underdog_model
from src.models.spread_model import train_spread_model
from src.models.totals_model import train_totals_model
from src.evaluation.backtester import walk_forward_backtest
from src.evaluation.profitability import profitability_report, edge_filter_analysis
from src.evaluation.calibration import calibration_report
from src.models.features import build_game_features
import pandas as pd

def main():
    # 1. Load data (from CSV or database)
    # features_df = build_game_features(...)  # See 04-betting-models.md
    
    # 2. Train each model and get metrics
    print("=" * 60)
    print("TRAINING UNDERDOG MODEL")
    print("=" * 60)
    underdog_result = train_underdog_model(features_df)
    
    print("\n" + "=" * 60)
    print("TRAINING SPREAD MODEL")
    print("=" * 60)
    spread_result = train_spread_model(features_df)
    
    print("\n" + "=" * 60)
    print("TRAINING TOTALS MODEL")
    print("=" * 60)
    totals_result = train_totals_model(features_df)
    
    # 3. Run backtests
    print("\n" + "=" * 60)
    print("BACKTESTING")
    print("=" * 60)
    
    # Example: underdog backtest with 3% minimum edge
    # backtest = walk_forward_backtest(
    #     features_df=features_df,
    #     train_fn=lambda X, y: XGBClassifier(...).fit(X, y),
    #     predict_fn=lambda model, X: model.predict_proba(X)[:, 1],
    #     target_col="underdog_won",
    #     odds_col="underdog_odds",
    #     pick_type="underdog",
    #     model_name="xgb_underdog_v1",
    #     min_edge=0.03,
    # )
    
    # 4. Profitability reports
    # profitability_report(backtest)
    # edge_filter_analysis(backtest)
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
```

---

## Metrics Cheat Sheet

| Situation | What to Check | Good Sign |
|-----------|--------------|-----------|
| "Is the model profitable?" | ROI, Units Profit | ROI > 2% over 500+ bets |
| "Are the probabilities accurate?" | Brier Score, Calibration plot | Brier < 0.24, close to diagonal |
| "Does the model find real edges?" | CLV, Edge Filter Analysis | Positive avg CLV |
| "Which confidence tier is best?" | Profitability by tier | High conf tier has best ROI |
| "Is it overfitting?" | Train vs. Test metrics gap | < 3% accuracy gap |
| "How bad can drawdowns get?" | Max Drawdown | < 15 units |

---

> **Next:** [06-daily-picks-engine.md](06-daily-picks-engine.md) – Turning model output into actionable daily picks.
