"""Pitcher availability facts derived from archived box scores.

``fetch_pitcher_usage_for_date`` finds the trailing final games for teams
playing on ``target_date``, archives each box score through the content-
addressed raw store, and normalizes per-pitcher workload facts (pitches
thrown, innings, role) with an explicit ``observed_at`` watermark.
Results are the provider-neutral ``pitcher_pitch`` observation frame consumed
by the point-in-time feature builders; replay never calls a live API.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import statsapi

from src.contracts.domain import RawObservation, stable_id
from src.ingestion.base import RetrievedPayload
from src.ingestion.raw_store import RawStore

from .config import config


def _archive_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    observed_at: datetime,
) -> Path | None:
    """Publish an immutable normalized observation to data/silver.

    The filename is content-addressed from the frame itself, so a corrected
    normalization produces a new file and never overwrites a prior version.
    """
    if frame.empty:
        return None
    import hashlib

    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()
    target = (
        config.project_root
        / "data"
        / "silver"
        / source
        / f"observed_date={observed_at.date().isoformat()}"
        / f"{source}_{content_hash[:16]}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    return target


def _role_of(pitcher: dict[str, object], innings: float) -> str:
    """Infer role from the boxscore note suffix and innings pitched.

    ``(W``/``(L`` decisions mark starters (or winning/losing pitchers); ``(H``
    and ``(S`` marks are holds/saves, i.e. relievers.  A pitcher with 5+ IP is
    treated as a starter regardless.
    """
    note = str(pitcher.get("note", "") or "")
    if "(H" in note or "(S" in note or "(BS" in note:
        return "reliever"
    if innings >= 5.0:
        return "starter"
    if "(W" in note or "(L" in note:
        return "starter"
    return "reliever"


def _boxscore_payload(game_id: int, *, observed_at: datetime) -> RetrievedPayload:
    box = statsapi.boxscore_data(game_id)
    body = json.dumps(box, default=str, separators=(",", ":"), sort_keys=True).encode()
    return RetrievedPayload(
        source="mlb_stats_boxscore",
        body=body,
        observed_at=observed_at,
        request_params={"game_id": game_id},
        http_metadata={"client": "MLB-StatsAPI"},
    )


def _game_team_ids(game: dict[str, object]) -> tuple[int, int]:
    away = int(str(game.get("away_id", 0) or 0))
    home = int(str(game.get("home_id", 0) or 0))
    return away, home


def fetch_pitcher_usage_for_date(
    target_date: date,
    *,
    lookback_days: int = 7,
    as_of: datetime | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch trailing pitcher usage for teams playing on ``target_date``.

    Returns one row per (team_id, player_id) with ``pitches``, ``outs_recorded``,
    ``role``, ``event_time``, and ``observed_at``.  Only final games strictly
    before the cutoff contribute; an incomplete box score is excluded rather
    than assumed complete.
    """
    observed_at = as_of or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    requested = target_date.isoformat()
    run_id = run_id or stable_id("ingestion", "mlb_pitcher_usage", observed_at.isoformat())

    schedule = statsapi.schedule(date=requested)
    team_ids = sorted(
        {int(game.get("away_id", 0)) for game in schedule}
        | {int(game.get("home_id", 0)) for game in schedule}
    )
    team_ids = [team for team in team_ids if team]

    start = (observed_at.date() - timedelta(days=lookback_days)).isoformat()
    end = observed_at.date().isoformat()
    raw_store = RawStore(config.project_root / "data" / "bronze")

    observations: list[tuple[int, int, RawObservation]] = []
    seen_games: set[int] = set()
    for team_id in team_ids:
        games = statsapi.schedule(team=team_id, start_date=start, end_date=end)
        for game in games:
            if game.get("status") != "Final":
                continue
            game_date = pd.to_datetime(game.get("game_date"), utc=True, errors="coerce")
            if pd.isna(game_date) or game_date >= pd.Timestamp(observed_at):
                continue
            game_id = int(game["game_id"])
            if game_id in seen_games:
                continue  # avoid persisting the same boxscore twice
            seen_games.add(game_id)
            observation = raw_store.persist(
                _boxscore_payload(game_id, observed_at=observed_at),
                ingestion_run_id=run_id,
            )
            away_team, home_team = _game_team_ids(game)
            observations.append((game_id, away_team, observation))
            observations.append((game_id, home_team, observation))

    rows: list[dict[str, object]] = []
    for game_id, team_id, observation in observations:
        payload = raw_store.load(observation)
        boxscore = json.loads(payload.decode())
        game_date = pd.to_datetime(
            boxscore.get("gameDate") or boxscore.get("gameId", "").split("/")[-1],
            utc=True,
            errors="coerce",
        )
        for side in ("away", "home"):
            pitchers = boxscore.get(f"{side}Pitchers", []) or []
            for pitcher in pitchers:
                if not isinstance(pitcher, dict):
                    continue
                person_id = int(pitcher.get("personId", 0) or 0)
                if person_id == 0:
                    continue  # summary row
                innings = float(pitcher.get("ip", 0) or 0)
                rows.append(
                    {
                        "game_id": game_id,
                        "team_id": str(team_id),
                        "player_id": str(person_id),
                        "event_time": (
                            game_date.isoformat()
                            if pd.notna(game_date)
                            else observed_at.date().isoformat()
                        ),
                        "pitches": int(pitcher.get("p", 0) or 0),
                        "outs_recorded": int(round(innings * 3)),
                        "role": _role_of(pitcher, innings),
                        "observed_at": observed_at.isoformat(),
                        "raw_payload_hash": observation.payload_sha256,
                        "ingestion_run_id": run_id,
                    }
                )
    frame = pd.DataFrame(rows)
    _archive_frame(frame, source="pitcher_pitch", observed_at=observed_at)
    return frame
