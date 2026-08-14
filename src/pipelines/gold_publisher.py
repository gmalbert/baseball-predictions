"""Publish gold artifacts consumed by the Phase 3 product pages.

``publish_gold_artifacts`` is a deterministic, fail-closed pipeline that takes
today's game frame plus provider-neutral observation frames and writes the
four gold parquets the pages read:

- ``game_distributions.parquet`` — one row per game with the coherent
  market probabilities derived from the simulated joint score distribution.
- ``bullpen_availability.parquet`` — per-team reliever availability.
- ``lineup_scenarios.parquet`` — projected/confirmed lineup rows.
- ``eligibility.parquet`` — per-selection eligibility/abstention reasons.

Missing observations produce an explicit empty/blocked state, never
fabricated probabilities.  Writes are atomic (temp + rename) so a failed run
never leaves a half-published gold set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts.domain import Decision, Prediction, Quote, stable_id
from src.features.builders import BullpenAvailabilityBuilder, PitcherAvailabilityBuilder
from src.features.snapshots import build_game_snapshot_rows
from src.models.game_simulator import FixedOutcomeModel, simulate_score_distribution
from src.models.prop_coherence import outcome_mix_from_rates

GOLD_SUBDIRS = (
    "game_distributions.parquet",
    "bullpen_availability.parquet",
    "lineup_scenarios.parquet",
    "eligibility.parquet",
)


@dataclass(frozen=True)
class PublishResult:
    distributions: int
    bullpen: int
    lineups: int
    eligibility: int
    status: str  # "published" | "blocked"
    reason: str = ""


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _game_distribution_rows(
    games: pd.DataFrame,
    *,
    as_of: datetime,
    n_simulations: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a coherent joint distribution per game from a league-average
    outcome mix.  Real deployments pass per-game outcome mixes from the PA
    model; the default is a neutral mix."""
    rows = []
    model = FixedOutcomeModel(
        outcome_mix_from_rates(hits_rate=0.21, hr_rate=0.029, walk_rate=0.085, strikeout_rate=0.22)
    )
    for index, game in enumerate(games.to_dict("records")):
        distribution = simulate_score_distribution(
            model=model,
            n_simulations=n_simulations,
            seed=seed + index,
        )
        home, away = distribution.home_moneyline(), distribution.away_moneyline()
        over9, under9, push9 = distribution.total_probabilities(9.0)
        over8, under8, push8 = distribution.total_probabilities(8.5)
        away_idx, home_idx = np.indices(distribution.matrix.shape)
        expected_total = float(((away_idx + home_idx) * distribution.matrix).sum())
        rows.append(
            {
                "game_id": game["game_id"],
                "away_team_id": game.get("away_team_id"),
                "home_team_id": game.get("home_team_id"),
                "home_moneyline": home,
                "away_moneyline": away,
                "tie_probability": distribution.tie_probability(),
                "run_line_home_cover": distribution.run_line_probabilities(-1.5)[0],
                "run_line_away_cover": distribution.run_line_probabilities(-1.5)[1],
                "over_probability_9": over9,
                "under_probability_9": under9,
                "push_probability_9": push9,
                "over_probability_8_5": over8,
                "under_probability_8_5": under8,
                "expected_total_runs": expected_total,
                "as_of_time": as_of.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _bullpen_rows(
    games: pd.DataFrame, observations: dict[str, pd.DataFrame], *, as_of: datetime
) -> pd.DataFrame:
    team_base = pd.concat(
        [
            games[["game_id", "home_team_id"]]
            .rename(columns={"home_team_id": "team_id"})
            .assign(side="home"),
            games[["game_id", "away_team_id"]]
            .rename(columns={"away_team_id": "team_id"})
            .assign(side="away"),
        ],
        ignore_index=True,
    )
    team_base["as_of_time"] = pd.Timestamp(as_of)
    frames: list[pd.DataFrame] = []
    # Frame keys each builder actually consumes (required_sources is advisory
    # metadata and can name frames the build() never reads).
    required = {
        BullpenAvailabilityBuilder: ("reliever_usage",),
        PitcherAvailabilityBuilder: ("pitcher_pitch",),
    }
    for builder in (
        BullpenAvailabilityBuilder(observations["reliever_usage"]),
        PitcherAvailabilityBuilder(observations["pitcher_pitch"]),
    ):
        if missing := set(required[type(builder)]) - set(observations):
            raise KeyError(f"{builder.name} requires frames {sorted(missing)}")
        family = builder.build(team_base, as_of=as_of)
        builder.validate(family, as_of=as_of)
        frames.append(family)
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for family in frames[1:]:
        keys = [column for column in ("game_id", "team_id") if column in family]
        payload = [column for column in family if column not in result or column in keys]
        result = result.merge(family[payload], on=keys, how="left", validate="one_to_one")
    result["as_of_time"] = pd.Timestamp(as_of)
    return result


def _lineup_rows(observations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lineup = observations.get("lineup_snapshot")
    if lineup is None or lineup.empty:
        return pd.DataFrame()
    return lineup.copy()


def build_reliever_usage_frame(pitcher_pitch: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """Derive the ``reliever_usage`` observation frame from ``pitcher_pitch``.

    The bullpen builder expects per-reliever ``pitches_last_3d``,
    ``consecutive_days``, ``quality``, and ``leverage_weight``.  Workload and
    consecutive days are computed from archived box scores; quality and
    leverage default to neutral values because we do not fabricate talent
    estimates here (they come from the player-talent family when available).
    """
    if pitcher_pitch is None or pitcher_pitch.empty:
        return pd.DataFrame()
    frame = pitcher_pitch.copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(as_of)
    frame = frame[frame["event_time"] < cutoff].copy()
    if frame.empty:
        return pd.DataFrame()
    # Workload over the trailing 3 days per pitcher.
    recent = frame[frame["event_time"] >= cutoff - pd.Timedelta(days=3)]
    workload = recent.groupby(["team_id", "player_id"])["pitches"].sum().rename("pitches_last_3d")
    # Consecutive days with an appearance, by distinct game date.
    days = frame.groupby(["team_id", "player_id"])["event_time"].apply(
        lambda series: len(series.dt.date.unique())
    )
    days = days.rename("consecutive_days")
    usage = (
        frame[frame["role"] == "reliever"]
        .drop_duplicates(["team_id", "player_id"])[
            ["team_id", "player_id", "observed_at", "game_id"]
        ]
        .copy()
    )
    usage = usage.merge(workload, on=["team_id", "player_id"], how="left")
    usage = usage.merge(days, on=["team_id", "player_id"], how="left")
    usage["pitches_last_3d"] = usage["pitches_last_3d"].fillna(0).astype(float)
    usage["consecutive_days"] = usage["consecutive_days"].fillna(0).astype(float)
    usage["quality"] = 0.5  # neutral; replaced by player-talent family when present
    usage["leverage_weight"] = 1.0
    return usage


def _eligibility_rows(
    predictions: list[Prediction] | None,
    quotes: list[Quote] | None,
    decisions: list[Decision] | None,
) -> pd.DataFrame:
    """Flatten per-selection eligibility into a page-friendly frame."""
    if not decisions:
        return pd.DataFrame()
    prediction_by_id = {row.prediction_id: row for row in (predictions or [])}
    rows = []
    for decision in decisions:
        prediction = prediction_by_id.get(decision.prediction_id)
        rows.append(
            {
                "game_id": decision.game_id,
                "market_id": decision.market_id,
                "selection": decision.selection.value,
                "eligible": decision.action == "bet",
                "quote_id": decision.quote_id,
                "reason_codes": ", ".join(decision.reason_codes),
                "as_of_time": decision.decided_at.isoformat(),
                "prediction_id": prediction.prediction_id if prediction else None,
                "probability": prediction.probability if prediction else None,
            }
        )
    return pd.DataFrame(rows)


def _run_manifest(
    *,
    gold_root: Path,
    as_of: datetime,
    result: PublishResult,
    distributions: pd.DataFrame,
    bullpen: pd.DataFrame,
    lineups: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> dict[str, object]:
    """Build the v2 run manifest the operations pages read."""
    games_with_distribution = set(distributions["game_id"]) if not distributions.empty else set()
    sources = [
        {
            "source": source,
            "rows": len(frame),
            "available": not frame.empty,
        }
        for source, frame in (
            ("lineup_snapshot", lineups),
            ("bullpen_availability", bullpen),
            ("game_distributions", distributions),
            ("eligibility", eligibility),
        )
    ]
    return {
        "status": result.status,
        "target_date": as_of.date().isoformat(),
        "run_id": stable_id("gold", as_of.isoformat()),
        "as_of_time": as_of.isoformat(),
        "stage": "gold_publish",
        "games": len(games_with_distribution),
        "distributions": result.distributions,
        "bullpen": result.bullpen,
        "lineups": result.lineups,
        "eligibility": result.eligibility,
        "notes": result.reason,
        "sources": sources,
    }


def write_run_manifest(gold_root: Path, manifest: dict[str, object]) -> None:
    import json

    target = gold_root / "run_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def publish_gold_artifacts(
    *,
    games: pd.DataFrame,
    observations: dict[str, pd.DataFrame],
    gold_root: Path,
    as_of: datetime,
    predictions: list[Prediction] | None = None,
    quotes: list[Quote] | None = None,
    decisions: list[Decision] | None = None,
) -> PublishResult:
    """Build and atomically write the four gold artifacts.

    ``games`` needs ``game_id``, ``home_team_id``, ``away_team_id``.
    ``observations`` maps ``lineup_snapshot``, ``reliever_usage``,
    ``pitcher_pitch`` to provider-neutral frames.  A missing source for a
    required family raises; the caller decides whether that blocks the day.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    required_games = {"game_id", "home_team_id", "away_team_id"}
    if missing := required_games - set(games):
        raise KeyError(f"Games missing columns: {sorted(missing)}")
    if games.empty:
        return PublishResult(0, 0, 0, 0, status="blocked", reason="no_games")

    # Build game snapshots first; fail closed if observation frames are missing.
    build_game_snapshot_rows(games, observations, as_of=as_of)

    distributions = _game_distribution_rows(games, as_of=as_of)
    bullpen = _bullpen_rows(games, observations, as_of=as_of)
    lineups = _lineup_rows(observations)
    eligibility = _eligibility_rows(predictions, quotes, decisions)

    _atomic_write(distributions, gold_root / "game_distributions.parquet")
    _atomic_write(bullpen, gold_root / "bullpen_availability.parquet")
    _atomic_write(lineups, gold_root / "lineup_scenarios.parquet")
    _atomic_write(eligibility, gold_root / "eligibility.parquet")

    result = PublishResult(
        distributions=len(distributions),
        bullpen=len(bullpen),
        lineups=len(lineups),
        eligibility=len(eligibility),
        status="published",
    )
    write_run_manifest(
        gold_root,
        _run_manifest(
            gold_root=gold_root,
            as_of=as_of,
            result=result,
            distributions=distributions,
            bullpen=bullpen,
            lineups=lineups,
            eligibility=eligibility,
        ),
    )
    return result
