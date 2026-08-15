"""Timestamped lineup and probable-starter observations from the MLB Stats API.

Every call archives the raw provider response through the content-addressed raw
store before normalization, so lineup facts carry retrieval metadata and can be
replayed deterministically.  Normalized rows are the provider-neutral
``lineup_snapshot`` / ``pitcher_pitch`` observation frames consumed by the
point-in-time feature builders.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import statsapi

from src.contracts.domain import stable_id
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


def fetch_lineups_for_date(
    target_date: date,
    *,
    as_of: datetime | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch and archive one lineups observation before returning normalized rows.

    Returns one row per (game_id, team_id, player_id) with batting order,
    defensive position, and lineup status, plus an ``observed_at`` watermark.
    """
    requested = target_date.isoformat()
    observed_at = as_of or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    # statsapi.schedule() cannot hydrate lineups; use the raw endpoint, which
    # returns ``lineups.homePlayers``/``awayPlayers`` in batting order.
    data = statsapi.get(
        "schedule",
        {"sportId": 1, "date": requested, "hydrate": "lineups,probablePitcher"},
    )
    games = []
    for date_group in data.get("dates", []):
        games.extend(date_group.get("games", []))
    body = json.dumps(games, default=str, separators=(",", ":"), sort_keys=True).encode()
    run_id = run_id or stable_id("ingestion", "mlb_lineups", observed_at.isoformat())
    observation = RawStore(config.project_root / "data" / "bronze").persist(
        RetrievedPayload(
            source="mlb_stats_lineups",
            body=body,
            observed_at=observed_at,
            request_params={"date": requested, "hydrate": "lineups,probablePitcher"},
            http_metadata={"client": "MLB-StatsAPI"},
        ),
        ingestion_run_id=run_id,
    )
    rows: list[dict[str, object]] = []
    for game in games:
        provider_game_id = int(game["gamePk"])
        teams = game.get("teams", {})
        for side in ("away", "home"):
            team_id = str(teams.get(side, {}).get("team", {}).get("id", "?"))
            players = game.get("lineups", {}).get(f"{side}Players", []) or []
            for index, player in enumerate(players):
                rows.append(
                    {
                        "game_id": provider_game_id,
                        "team_id": team_id,
                        "player_id": str(player.get("id", "?")),
                        "batting_order": index + 1,
                        "defensive_position": player.get("primaryPosition", {}).get(
                            "abbreviation", "?"
                        ),
                        "lineup_status": "confirmed",
                        "observed_at": observed_at.isoformat(),
                        "raw_payload_hash": observation.payload_sha256,
                        "ingestion_run_id": run_id,
                    }
                )
    frame = pd.DataFrame(rows)
    _archive_frame(frame, source="lineup_snapshot", observed_at=observed_at)
    return frame
