"""Build honest game snapshots from strictly prior, observed team-game facts."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.contracts.domain import stable_id


def _prior_summary(
    history: pd.DataFrame,
    *,
    value_columns: list[str],
    prior: pd.Series,
    prior_strength: float,
    half_life_games: float,
) -> dict[str, float | int | pd.Timestamp]:
    result: dict[str, float | int | pd.Timestamp] = {"games_played": len(history)}
    weights = (
        np.power(0.5, np.arange(len(history) - 1, -1, -1) / half_life_games)
        if len(history)
        else np.array([])
    )
    for column in value_columns:
        values = pd.to_numeric(history[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        weighted_sum = float(np.dot(values[valid], weights[valid])) if valid.any() else 0.0
        weight = float(weights[valid].sum()) if valid.any() else 0.0
        result[f"{column}_ewm"] = (weighted_sum + prior_strength * float(prior[column])) / (
            weight + prior_strength
        )
        result[f"{column}_missing"] = int(not valid.any())
        result[f"{column}_uncertainty"] = (
            float(np.nanstd(values) / np.sqrt(max(valid.sum(), 1))) if valid.any() else float("nan")
        )
    result["source_max_observed_at"] = history["observed_at"].max() if len(history) else pd.NaT
    return result


def build_game_snapshots(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    value_columns: list[str],
    decision_offset: timedelta = timedelta(hours=4),
    prior_strength: float = 10.0,
    half_life_games: float = 10.0,
    feature_set_version: str = "mlb_game_v2",
) -> pd.DataFrame:
    required_games = {"game_id", "scheduled_start_utc", "home_team_id", "away_team_id"}
    required_history = {"team_id", "game_id", "event_time", "observed_at", *value_columns}
    if missing := required_games - set(games):
        raise KeyError(f"Games missing columns: {sorted(missing)}")
    if missing := required_history - set(team_games):
        raise KeyError(f"Team games missing columns: {sorted(missing)}")
    targets = games.copy()
    history = team_games.copy()
    targets["scheduled_start_utc"] = pd.to_datetime(targets["scheduled_start_utc"], utc=True)
    history["event_time"] = pd.to_datetime(history["event_time"], utc=True)
    history["observed_at"] = pd.to_datetime(history["observed_at"], utc=True)
    rows = []
    for game in targets.to_dict("records"):
        as_of = game["scheduled_start_utc"] - decision_offset
        eligible_league = history[
            (history["event_time"] < game["scheduled_start_utc"])
            & (history["observed_at"] <= as_of)
            & (history["game_id"] != game["game_id"])
        ]
        if eligible_league.empty:
            raise ValueError(f"No prior league observations available for {game['game_id']}")
        league_prior = eligible_league[value_columns].apply(pd.to_numeric, errors="coerce").mean()
        row: dict[str, object] = {
            "snapshot_id": stable_id("snapshot", game["game_id"], as_of, feature_set_version),
            "game_id": game["game_id"],
            "as_of_time": as_of,
            "scheduled_start_utc": game["scheduled_start_utc"],
            "feature_set_version": feature_set_version,
        }
        watermarks = []
        for side in ("home", "away"):
            team_id = game[f"{side}_team_id"]
            eligible = history[
                (history["team_id"] == team_id)
                & (history["event_time"] < game["scheduled_start_utc"])
                & (history["observed_at"] <= as_of)
                & (history["game_id"] != game["game_id"])
            ].sort_values("event_time")
            summary = _prior_summary(
                eligible,
                value_columns=value_columns,
                prior=league_prior,
                prior_strength=prior_strength,
                half_life_games=half_life_games,
            )
            for key, value in summary.items():
                if key == "source_max_observed_at":
                    if pd.notna(value):
                        watermarks.append(value)
                else:
                    row[f"{side}_{key}"] = value
            row[f"{side}_team_id"] = team_id
        row["source_max_observed_at"] = max(watermarks) if watermarks else pd.NaT
        if pd.notna(row["source_max_observed_at"]) and row["source_max_observed_at"] > as_of:
            raise AssertionError("Future observation entered model matrix")
        rows.append(row)
    return pd.DataFrame(rows)
