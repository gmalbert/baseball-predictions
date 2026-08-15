# src/ingestion/mlb_stats.py
"""Pull schedules, game results, and probable pitchers from the MLB Stats API."""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from time import sleep

import pandas as pd
import statsapi

from src.contracts.domain import stable_game_id, stable_id
from src.ingestion.base import RetrievedPayload
from src.ingestion.raw_store import RawStore

from .config import config


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """Fetch every game for a given season.

    Returns DataFrame with: game_id, date, away_team, home_team,
    away_score, home_score, status, venue, away_pitcher, home_pitcher.
    """
    start = f"{year}-02-20"  # Spring training start
    end = f"{year}-11-05"  # Include postseason

    print(f"Fetching {year} schedule...")
    games = statsapi.schedule(start_date=start, end_date=end)

    rows = []
    for g in games:
        rows.append(
            {
                "game_id": g["game_id"],
                "date": g["game_date"],
                "away_team": g["away_name"],
                "home_team": g["home_name"],
                "away_score": g.get("away_score"),
                "home_score": g.get("home_score"),
                "status": g["status"],
                "venue": g.get("venue_name", ""),
                "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
                "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
                "series_description": g.get("series_description", ""),
                "game_type": g.get("game_type", "R"),  # R=regular, P=postseason
            }
        )

    df = pd.DataFrame(rows)
    return df


def fetch_all_schedules() -> pd.DataFrame:
    """Fetch schedules for all configured years and save to CSV."""
    all_dfs: list[pd.DataFrame] = []
    for year in range(config.start_year, config.end_year + 1):
        df = fetch_season_schedule(year)
        # Filter to regular season + postseason only
        df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])]
        outpath = config.raw_dir / "gamelogs" / f"schedule_{year}.csv"
        df.to_csv(outpath, index=False)
        print(f"  Saved {len(df)} games → {outpath}")
        all_dfs.append(df)
        sleep(config.request_delay_sec)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(config.raw_dir / "gamelogs" / "schedule_all.csv", index=False)
    return combined


def fetch_schedule_for_date(
    target_date: date,
    *,
    as_of: datetime | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch and archive one schedule observation before returning normalized rows."""
    requested = (target_date or date.today()).isoformat()
    observed_at = as_of or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    games = statsapi.schedule(date=requested)
    body = json.dumps(games, default=str, separators=(",", ":"), sort_keys=True).encode()
    run_id = run_id or stable_id("ingestion", "mlb_schedule", observed_at.isoformat())
    observation = RawStore(config.project_root / "data" / "bronze").persist(
        RetrievedPayload(
            source="mlb_stats_schedule",
            body=body,
            observed_at=observed_at,
            request_params={"date": requested},
            http_metadata={"client": "MLB-StatsAPI"},
        ),
        ingestion_run_id=run_id,
    )
    rows = []
    for g in games:
        provider_game_id = int(g["game_id"])
        scheduled_start = pd.to_datetime(g.get("game_datetime"), utc=True, errors="coerce")
        rows.append(
            {
                "game_id": stable_game_id(
                    season=target_date.year,
                    scheduled_start_utc=(
                        scheduled_start.to_pydatetime()
                        if pd.notna(scheduled_start)
                        else datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
                    ),
                    away_team_id=str(g.get("away_id", g["away_name"])),
                    home_team_id=str(g.get("home_id", g["home_name"])),
                    doubleheader_number=g.get("game_num"),
                    mlb_game_pk=provider_game_id,
                ),
                "provider_game_id": provider_game_id,
                "date": requested,
                "away_team": g["away_name"],
                "home_team": g["home_name"],
                "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
                "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
                "venue": g.get("venue_name", ""),
                "game_time": g.get("game_datetime", ""),
                "scheduled_start_utc": (
                    scheduled_start.isoformat() if pd.notna(scheduled_start) else None
                ),
                "status": g.get("status", ""),
                "away_score": g.get("away_score"),
                "home_score": g.get("home_score"),
                "game_type": g.get("game_type", "R"),
                "observed_at": observed_at.isoformat(),
                "raw_payload_hash": observation.payload_sha256,
                "ingestion_run_id": run_id,
            }
        )
    frame = pd.DataFrame(rows)
    # MLB StatsAPI returns scores as strings for in-progress games and ints
    # for completed ones; normalize so pyarrow can write a single column type.
    for col in ("away_score", "home_score"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    _archive_schedule_observation(
        frame,
        observed_at=observed_at,
        raw_payload_hash=observation.payload_sha256,
    )
    return frame


def _archive_schedule_observation(
    frame: pd.DataFrame,
    *,
    observed_at: datetime,
    raw_payload_hash: str,
) -> Path | None:
    """Publish a content-addressed, immutable normalized schedule observation."""
    if frame.empty:
        return None
    target = (
        config.project_root
        / "data"
        / "silver"
        / "game_schedule_observation"
        / f"observed_date={observed_at.date().isoformat()}"
        / f"schedule_{raw_payload_hash[:16]}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    return target


def fetch_todays_probable_pitchers(target_date: date | None = None) -> pd.DataFrame:
    """Fetch a requested schedule date with probable pitchers."""
    return fetch_schedule_for_date(target_date or date.today())


def fetch_game_pace(year: int) -> pd.DataFrame:
    """Fetch game-pace metrics for a season from the MLB Stats API.

    Includes average innings pitched, game duration, pitches per plate appearance,
    and total runs per game — useful context for totals model calibration.

    Args:
        year: The MLB season.

    Returns:
        DataFrame with columns: season, league, games, avg_game_duration_min,
        avg_innings, runs_per_game, pitches_per_pa.
        Returns an empty DataFrame if the endpoint is unavailable.
    """
    try:
        data = statsapi.get(
            "schedule_games_pace",
            {"season": year, "sportId": 1},
        )
        items = data.get("gamesPaced", [])
        rows = []
        for item in items:
            rows.append(
                {
                    "season": year,
                    "league": item.get("leagueAbbreviation", "MLB"),
                    "games": item.get("gamesPlayed"),
                    "avg_game_duration_min": item.get("avgGameDurationMinutes"),
                    "avg_innings": item.get("avgInningsPlayed"),
                    "runs_per_game": item.get("runsPerGame"),
                    "pitches_per_pa": item.get("pitchesPerPlateAppearance"),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("fetch_game_pace failed for %d: %s", year, exc)
        return pd.DataFrame()


def fetch_streaks(year: int, streak_type: str = "wins", threshold: int = 4) -> pd.DataFrame:
    """Fetch current hot/cold streaks for teams via the MLB Stats API.

    Uses the ``/stats/streaks`` endpoint to identify teams on notable
    win or loss streaks — a signal for short-term momentum in moneyline models.

    Args:
        year:        The MLB season.
        streak_type: "wins" or "losses".
        threshold:   Minimum streak length to include.

    Returns:
        DataFrame with columns: team, streak_type, streak_length, season.
        Returns an empty DataFrame if the endpoint is unavailable.
    """
    try:
        stat_type = "wins" if streak_type.lower() == "wins" else "losses"
        data = statsapi.get(
            "stats_streaks",
            {
                "season": year,
                "sportId": 1,
                "streakType": stat_type,
                "streakSpan": "career",
                "gameType": "R",
                "limit": 50,
            },
        )
        rows = []
        for entry in data.get("streaks", []):
            length = entry.get("streakLength", 0)
            if length >= threshold:
                team_info = entry.get("team", {}) or entry.get("player", {})
                rows.append(
                    {
                        "team": team_info.get("name", ""),
                        "streak_type": stat_type,
                        "streak_length": length,
                        "season": year,
                    }
                )
        return pd.DataFrame(rows)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "fetch_streaks failed for %d/%s: %s", year, streak_type, exc
        )
        return pd.DataFrame()


if __name__ == "__main__":
    fetch_all_schedules()
