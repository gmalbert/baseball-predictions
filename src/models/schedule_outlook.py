"""Clearly speculative schedule/rotation/roster Monte Carlo for F60 research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutlookRow:
    team_id: str
    expected_wins: float
    win_low: float
    win_high: float
    simulations: int
    speculative: bool = True


def simulate_schedule_outlook(
    schedule: pd.DataFrame,
    *,
    simulations: int = 10_000,
    seed: int = 42,
) -> list[OutlookRow]:
    required = {"home_team_id", "away_team_id", "home_win_probability"}
    if missing := required - set(schedule):
        raise KeyError(f"Schedule outlook missing: {sorted(missing)}")
    generator = np.random.default_rng(seed)
    teams = sorted(set(schedule["home_team_id"]) | set(schedule["away_team_id"]))
    totals = {team: np.zeros(simulations) for team in teams}
    for row in schedule.to_dict("records"):
        home_wins = generator.random(simulations) < float(row["home_win_probability"])
        totals[row["home_team_id"]] += home_wins
        totals[row["away_team_id"]] += ~home_wins
    return [
        OutlookRow(
            team_id=team,
            expected_wins=float(values.mean()),
            win_low=float(np.quantile(values, 0.05)),
            win_high=float(np.quantile(values, 0.95)),
            simulations=simulations,
        )
        for team, values in totals.items()
    ]
