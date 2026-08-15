"""Publish today's gold artifacts from the latest silver observation frames.

Usage:
    python scripts/run_gold_publish.py --target-date 2026-08-10 [--as-of 2026-08-10T16:00:00Z]

Loads the latest ``lineup_snapshot``, ``pitcher_pitch``, and ``reliever_usage``
frames under ``data/silver`` for the target date, builds game snapshots, and
publishes the four gold parquets the product pages read.  Fails closed with a
non-zero exit when required observations are missing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.ingestion.config import config
from src.pipelines.gold_publisher import build_reliever_usage_frame, publish_gold_artifacts

SILVER_SOURCES = {
    "lineup_snapshot": "lineup_snapshot",
    "pitcher_pitch": "pitcher_pitch",
    "reliever_usage": "reliever_usage",
}


def _load_latest_silver(source: str, target_date: date) -> pd.DataFrame:
    """Load the most recent observation parquet for ``source`` on target date."""
    root = config.project_root / "data" / "silver" / source
    if not root.is_dir():
        return pd.DataFrame()
    partition = root / f"observed_date={target_date.isoformat()}"
    candidates = (
        sorted(partition.glob("*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
        if partition.is_dir()
        else []
    )
    if not candidates:
        return pd.DataFrame()
    return pd.read_parquet(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    parser.add_argument("--gold-root", type=Path, default=None)
    args = parser.parse_args()

    as_of = args.as_of or datetime.now(UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    gold_root = args.gold_root or config.project_root / "data" / "gold"

    # Games come from the schedule observation for the target date, falling
    # back to a live fetch so the publisher can run after ingestion.
    schedule = _load_latest_silver("game_schedule_observation", args.target_date)
    if schedule.empty:
        from src.ingestion.mlb_stats import fetch_schedule_for_date

        schedule = fetch_schedule_for_date(args.target_date, as_of=as_of)
    if schedule.empty:
        raise SystemExit(
            f"No schedule observation for {args.target_date.isoformat()}; "
            "run the ingestion job first."
        )
    observations = {
        name: _load_latest_silver(source, args.target_date)
        for name, source in SILVER_SOURCES.items()
    }
    # Derive the reliever-usage frame the bullpen builder needs from the
    # archived pitcher-pitch boxscore facts.
    if observations.get("reliever_usage") is None or observations["reliever_usage"].empty:
        pitcher_pitch = observations.get("pitcher_pitch")
        if pitcher_pitch is not None and not pitcher_pitch.empty:
            observations["reliever_usage"] = build_reliever_usage_frame(pitcher_pitch, as_of=as_of)

    games = (
        schedule[["provider_game_id", "away_team", "home_team"]]
        .drop_duplicates()
        .rename(columns={"provider_game_id": "game_id"})
        .reset_index(drop=True)
    )
    # Canonical game_id is a string; the provider ints must be normalized to
    # match the lineup/pitcher observation frames.
    games["game_id"] = games["game_id"].astype(str)
    for frame in observations.values():
        if frame is not None and not frame.empty and "game_id" in frame:
            frame["game_id"] = frame["game_id"].astype(str)
    # The schedule observation carries team names but not IDs; derive team IDs
    # from the lineup frame, which keys on the same MLB gamePk.  The raw
    # schedule endpoint lists homePlayers before awayPlayers, so the first
    # team per game is home.
    lineup = observations.get("lineup_snapshot")
    if lineup is not None and not lineup.empty and "team_id" in lineup:
        team_ids = (
            lineup[["game_id", "team_id"]]
            .drop_duplicates()
            .groupby("game_id")["team_id"]
            .apply(list)
            .to_dict()
        )
        games["home_team_id"] = games["game_id"].map(
            lambda game_id: (team_ids.get(str(game_id)) or [None])[0]
        )
        games["away_team_id"] = games["game_id"].map(
            lambda game_id: (team_ids.get(str(game_id)) or [None, None])[1]
        )
        games = games[games["home_team_id"].notna() & games["away_team_id"].notna()]

    result = publish_gold_artifacts(
        games=games,
        observations=observations,
        gold_root=gold_root,
        as_of=as_of,
    )
    print(
        f"gold publish {result.status}: distributions={result.distributions} "
        f"bullpen={result.bullpen} lineups={result.lineups} "
        f"eligibility={result.eligibility}"
    )
    if result.status != "published":
        raise SystemExit(f"gold publish blocked: {result.reason}")


if __name__ == "__main__":
    main()
