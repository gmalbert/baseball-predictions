# 10 – Bankroll & Risk Management

Features for tracking bets, managing bankroll, and responsible gambling.

---

## 1. Kelly Criterion Bet Sizing

The Kelly Criterion determines optimal bet size based on your edge.

```python
# src/bankroll/kelly.py
"""Kelly Criterion and fractional Kelly bet sizing."""

from dataclasses import dataclass


@dataclass
class BetSizing:
    """Recommended bet size for a pick."""
    full_kelly_pct: float     # Optimal Kelly % of bankroll
    half_kelly_pct: float     # Conservative: half Kelly
    quarter_kelly_pct: float  # Ultra-conservative: quarter Kelly
    recommended_pct: float    # Our recommendation
    recommended_units: float  # In units (1 unit = 1% of bankroll)
    reason: str


def kelly_criterion(
    predicted_prob: float,
    american_odds: int,
) -> float:
    """Calculate the Kelly Criterion fraction.
    
    Kelly% = (bp - q) / b
    Where:
        b = decimal odds - 1 (net payout per $1)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Returns: fraction of bankroll to wager (0 to 1)
    """
    # Convert American to decimal
    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1
    
    b = decimal_odds - 1  # net payout
    p = predicted_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    return max(kelly, 0)  # Never bet negative Kelly


def get_bet_sizing(
    predicted_prob: float,
    american_odds: int,
    confidence: str,
    bankroll: float = 10000,
    unit_size_pct: float = 0.01,  # 1 unit = 1% of bankroll
    max_bet_pct: float = 0.05,    # Never bet more than 5% of bankroll
) -> BetSizing:
    """Calculate recommended bet size for a pick.
    
    Uses fractional Kelly to reduce variance.
    
    Args:
        predicted_prob: Model's probability of the pick winning
        american_odds: Line odds (e.g., +150, -110)
        confidence: 'high', 'medium', 'low'
        bankroll: Current bankroll in dollars
        unit_size_pct: What % of bankroll = 1 unit
        max_bet_pct: Maximum bet as % of bankroll
    """
    full_kelly = kelly_criterion(predicted_prob, american_odds)
    half_kelly = full_kelly / 2
    quarter_kelly = full_kelly / 4
    
    # Choose fraction based on confidence
    if confidence == "high":
        recommended = min(half_kelly, max_bet_pct)
        reason = "Half Kelly — high confidence pick"
    elif confidence == "medium":
        recommended = min(quarter_kelly, max_bet_pct)
        reason = "Quarter Kelly — medium confidence pick"
    else:
        recommended = min(quarter_kelly / 2, max_bet_pct)
        reason = "Eighth Kelly — low confidence, small position"
    
    # Convert to units
    unit_value = bankroll * unit_size_pct
    recommended_units = (recommended * bankroll) / unit_value
    
    return BetSizing(
        full_kelly_pct=round(full_kelly * 100, 2),
        half_kelly_pct=round(half_kelly * 100, 2),
        quarter_kelly_pct=round(quarter_kelly * 100, 2),
        recommended_pct=round(recommended * 100, 2),
        recommended_units=round(recommended_units, 2),
        reason=reason,
    )


# Example usage
if __name__ == "__main__":
    # Model says 55% chance, line is -110
    sizing = get_bet_sizing(
        predicted_prob=0.55,
        american_odds=-110,
        confidence="high",
        bankroll=10000,
    )
    print(f"Full Kelly: {sizing.full_kelly_pct}%")
    print(f"Half Kelly: {sizing.half_kelly_pct}%")
    print(f"Recommended: {sizing.recommended_pct}% ({sizing.recommended_units} units)")
    print(f"Reason: {sizing.reason}")
    
    # Underdog at +200, model says 40%
    sizing2 = get_bet_sizing(
        predicted_prob=0.40,
        american_odds=200,
        confidence="medium",
    )
    print(f"\nUnderdog bet: {sizing2.recommended_units} units")
```

---

## 2. Bankroll Tracker

```python
# src/bankroll/tracker.py
"""Track user bankroll, bet history, and P/L over time."""

import pandas as pd
import numpy as np
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BetRecord:
    """Single tracked bet."""
    id: int
    date: date
    game: str               # "NYY @ BOS"
    pick_type: str
    pick_value: str
    odds: int
    units_wagered: float
    confidence: str
    result: Optional[str] = None     # win, loss, push
    profit_units: Optional[float] = None
    sportsbook: str = ""


@dataclass
class BankrollState:
    """Current bankroll snapshot."""
    starting_bankroll: float
    current_bankroll: float
    total_wagered: float
    total_profit: float
    total_bets: int
    wins: int
    losses: int
    pushes: int
    pending: int
    
    @property
    def roi(self) -> float:
        return self.total_profit / self.total_wagered if self.total_wagered > 0 else 0
    
    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0


class BankrollTracker:
    """Manage and track a user's betting bankroll."""
    
    def __init__(self, starting_bankroll: float = 10000, unit_size: float = 100):
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.unit_size = unit_size  # $100 per unit
        self.bets: list[BetRecord] = []
        self._next_id = 1
    
    def place_bet(
        self,
        game: str,
        pick_type: str,
        pick_value: str,
        odds: int,
        units: float,
        confidence: str,
        sportsbook: str = "",
    ) -> BetRecord:
        """Record a new bet."""
        bet = BetRecord(
            id=self._next_id,
            date=date.today(),
            game=game,
            pick_type=pick_type,
            pick_value=pick_value,
            odds=odds,
            units_wagered=units,
            confidence=confidence,
            sportsbook=sportsbook,
        )
        self.bets.append(bet)
        self.current_bankroll -= units * self.unit_size
        self._next_id += 1
        return bet
    
    def settle_bet(self, bet_id: int, result: str):
        """Settle a bet with its result."""
        bet = next((b for b in self.bets if b.id == bet_id), None)
        if not bet:
            raise ValueError(f"Bet {bet_id} not found")
        
        bet.result = result
        
        if result == "win":
            if bet.odds > 0:
                profit = bet.units_wagered * (bet.odds / 100)
            else:
                profit = bet.units_wagered * (100 / abs(bet.odds))
            bet.profit_units = profit
            self.current_bankroll += (bet.units_wagered + profit) * self.unit_size
        elif result == "loss":
            bet.profit_units = -bet.units_wagered
            # Bankroll already decreased when bet was placed
        elif result == "push":
            bet.profit_units = 0
            self.current_bankroll += bet.units_wagered * self.unit_size
    
    def get_state(self) -> BankrollState:
        """Get current bankroll state."""
        settled = [b for b in self.bets if b.result is not None]
        
        return BankrollState(
            starting_bankroll=self.starting_bankroll,
            current_bankroll=round(self.current_bankroll, 2),
            total_wagered=sum(b.units_wagered for b in self.bets) * self.unit_size,
            total_profit=sum(b.profit_units or 0 for b in settled) * self.unit_size,
            total_bets=len(self.bets),
            wins=sum(1 for b in settled if b.result == "win"),
            losses=sum(1 for b in settled if b.result == "loss"),
            pushes=sum(1 for b in settled if b.result == "push"),
            pending=sum(1 for b in self.bets if b.result is None),
        )
    
    def daily_pnl(self) -> pd.DataFrame:
        """Get daily profit/loss breakdown."""
        settled = [b for b in self.bets if b.result is not None]
        if not settled:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            "date": b.date,
            "profit_units": b.profit_units or 0,
        } for b in settled])
        
        daily = df.groupby("date").agg(
            bets=("profit_units", "count"),
            profit=("profit_units", "sum"),
        ).reset_index()
        
        daily["cumulative"] = daily["profit"].cumsum()
        return daily
    
    def streak(self) -> tuple[str, int]:
        """Get current win/loss streak."""
        settled = [b for b in self.bets if b.result in ("win", "loss")]
        if not settled:
            return ("", 0)
        
        # Sort by date descending
        settled.sort(key=lambda b: b.date, reverse=True)
        current = settled[0].result
        count = 0
        for b in settled:
            if b.result == current:
                count += 1
            else:
                break
        
        return (current[0].upper(), count)  # ("W", 5) or ("L", 3)
```

---

## 3. Risk Management Rules

```python
# src/bankroll/risk.py
"""Risk management rules to protect the bankroll."""

from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Configurable risk management parameters."""
    max_daily_loss_units: float = 5.0      # Stop after losing 5 units in a day
    max_daily_bets: int = 8                 # Max 8 bets per day
    max_single_bet_units: float = 3.0       # No single bet > 3 units
    max_exposure_pct: float = 0.15          # Max 15% of bankroll at risk
    min_bankroll_pct: float = 0.50          # Stop if bankroll drops below 50%
    cooldown_after_streak: int = 5          # Pause after 5 consecutive losses


def check_risk_limits(
    daily_profit: float,
    daily_bet_count: int,
    proposed_units: float,
    current_bankroll: float,
    starting_bankroll: float,
    pending_exposure: float,
    loss_streak: int,
    limits: RiskLimits = RiskLimits(),
) -> tuple[bool, str]:
    """Check if a proposed bet passes risk management rules.
    
    Returns:
        (allowed: bool, reason: str)
    """
    # Rule 1: Daily loss limit
    if daily_profit <= -limits.max_daily_loss_units:
        return False, f"Daily loss limit reached ({limits.max_daily_loss_units} units)"
    
    # Rule 2: Max daily bets
    if daily_bet_count >= limits.max_daily_bets:
        return False, f"Max daily bets reached ({limits.max_daily_bets})"
    
    # Rule 3: Single bet size
    if proposed_units > limits.max_single_bet_units:
        return False, f"Bet size {proposed_units}u exceeds max ({limits.max_single_bet_units}u)"
    
    # Rule 4: Total exposure
    total_exposure = (pending_exposure + proposed_units * 100) / current_bankroll
    if total_exposure > limits.max_exposure_pct:
        return False, f"Total exposure {total_exposure:.1%} exceeds max ({limits.max_exposure_pct:.0%})"
    
    # Rule 5: Bankroll floor
    bankroll_pct = current_bankroll / starting_bankroll
    if bankroll_pct < limits.min_bankroll_pct:
        return False, f"Bankroll at {bankroll_pct:.0%} of starting — below minimum ({limits.min_bankroll_pct:.0%})"
    
    # Rule 6: Loss streak cooldown
    if loss_streak >= limits.cooldown_after_streak:
        return False, f"On a {loss_streak}-game losing streak — cooldown period"
    
    return True, "Bet approved"


# Example usage
if __name__ == "__main__":
    allowed, reason = check_risk_limits(
        daily_profit=-3.5,
        daily_bet_count=4,
        proposed_units=2.0,
        current_bankroll=8500,
        starting_bankroll=10000,
        pending_exposure=300,
        loss_streak=2,
    )
    print(f"Allowed: {allowed} — {reason}")
```

---

## 4. Bankroll Data Access

Bankroll state is persisted as a CSV/Parquet file and loaded directly by Streamlit.

```python
# src/bankroll/storage.py
"""Persist and load bankroll state to/from Parquet."""

import pandas as pd
from pathlib import Path
from datetime import date

DATA_DIR = Path(__file__).resolve().parents[2] / "data_files" / "processed"
BETS_FILE = DATA_DIR / "bets_log.parquet"


def save_bet(bet_record: dict):
    """Append a bet record to the bets log."""
    df_new = pd.DataFrame([bet_record])
    if BETS_FILE.exists():
        df_existing = pd.read_parquet(BETS_FILE)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(BETS_FILE, index=False)


def load_bets() -> pd.DataFrame:
    """Load full bet history."""
    if not BETS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(BETS_FILE)


def get_bankroll_summary(starting_bankroll: float = 10000) -> dict:
    """Compute bankroll summary from bet history."""
    df = load_bets()
    if df.empty:
        return {
            "starting": starting_bankroll,
            "current": starting_bankroll,
            "profit": 0,
            "roi": 0,
            "total_bets": 0,
            "win_rate": 0,
            "streak": "",
        }

    settled = df[df["result"].notna()]
    wins = (settled["result"] == "win").sum()
    losses = (settled["result"] == "loss").sum()
    profit = settled["profit_units"].sum()

    return {
        "starting": starting_bankroll,
        "current": starting_bankroll + profit * (starting_bankroll * 0.01),
        "profit": float(profit),
        "roi": float(profit / len(settled)) if len(settled) > 0 else 0,
        "total_bets": len(df),
        "win_rate": float(wins / (wins + losses)) if (wins + losses) > 0 else 0,
        "streak": _current_streak(settled),
    }


def _current_streak(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    recent = df.sort_values("date", ascending=False)
    current = recent.iloc[0]["result"]
    count = 0
    for _, row in recent.iterrows():
        if row["result"] == current:
            count += 1
        else:
            break
    return f"{current[0].upper()}{count}"
```

---

## 5. Bankroll UI Component (Streamlit)

```python
# streamlit_app/components/bankroll_widget.py
"""Reusable bankroll display component for Streamlit."""

import streamlit as st


def render_bankroll_widget(data: dict):
    """Render bankroll summary as Streamlit metrics + progress bar.
    
    Args:
        data: dict with keys: starting, current, profit, roi, streak, win_rate, total_bets
    """
    st.subheader("Bankroll")

    # Progress bar
    bankroll_pct = data["current"] / data["starting"] if data["starting"] > 0 else 1.0
    st.progress(
        min(bankroll_pct, 1.0),
        text=f"${data['current']:,.0f} / ${data['starting']:,.0f} ({bankroll_pct:.0%})",
    )

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit", f"{data['profit']:+.1f} units")
    c2.metric("ROI", f"{data['roi']:.1%}")
    c3.metric("Win Rate", f"{data['win_rate']:.1%}")
    c4.metric("Streak", data.get("streak", "—"))
```

---

## 6. Responsible Gambling

### Required Disclaimers (Streamlit)

```python
# streamlit_app/components/disclaimer.py
"""Responsible gambling notice for Streamlit."""

import streamlit as st


def render_disclaimer():
    """Show responsible gambling warning banner."""
    st.warning(
        "**Responsible Gambling Notice**\n\n"
        "This site provides statistical predictions for entertainment and "
        "informational purposes only. Past performance does not guarantee future "
        "results. Never bet more than you can afford to lose.\n\n"
        "If you or someone you know has a gambling problem, call "
        "**1-800-GAMBLER** or visit [ncpgambling.org](https://www.ncpgambling.org)."
    )
```

### Self-Exclusion / Limit Features

```python
# src/bankroll/responsible.py
"""Responsible gambling features."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class UserLimits:
    """User-configurable responsible gambling limits."""
    daily_deposit_limit: float = 500
    weekly_deposit_limit: float = 2000
    monthly_loss_limit: float = 1000
    session_time_limit_minutes: int = 120
    self_exclusion_until: date | None = None
    
    def is_excluded(self) -> bool:
        if self.self_exclusion_until is None:
            return False
        return date.today() < self.self_exclusion_until
    
    def set_self_exclusion(self, days: int):
        """User can self-exclude for a period."""
        self.self_exclusion_until = date.today() + timedelta(days=days)
```

---

## Summary

This bankroll module provides:

1. **Kelly Criterion sizing** — mathematically optimal bet sizing based on model edge
2. **Bankroll tracking** — log bets, track P/L, monitor progress
3. **Risk management rules** — automatic stops for daily losses, exposure limits, streak cooldowns
4. **Streamlit components** — bankroll widget, disclaimer, Kelly calculator
5. **Responsible gambling** — disclaimers, self-exclusion, configurable limits

---

> **Back to:** [00-roadmap-overview.md](00-roadmap-overview.md)
