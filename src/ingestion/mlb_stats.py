# src/ingestion/mlb_stats.py
"""Pull schedules, game results, and probable pitchers from the MLB Stats API."""

import statsapi
import pandas as pd
from datetime import datetime
from time import sleep

from .config import config


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """Fetch every game for a given season.

    Returns DataFrame with: game_id, date, away_team, home_team,
    away_score, home_score, status, venue, away_pitcher, home_pitcher.
    """
    start = f"{year}-02-20"  # Spring training start
    end = f"{year}-11-05"    # Include postseason

    print(f"Fetching {year} schedule...")
    games = statsapi.schedule(start_date=start, end_date=end)

    rows = []
    for g in games:
        rows.append({
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
        })

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


def fetch_todays_probable_pitchers() -> pd.DataFrame:
    """Fetch today's games with probable pitchers (for daily picks)."""
    today = datetime.now().strftime("%Y-%m-%d")
    games = statsapi.schedule(date=today)
    rows = []
    for g in games:
        rows.append({
            "game_id": g["game_id"],
            "date": today,
            "away_team": g["away_name"],
            "home_team": g["home_name"],
            "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
            "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
            "venue": g.get("venue_name", ""),
            "game_time": g.get("game_datetime", ""),
        })
    return pd.DataFrame(rows)


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
            rows.append({
                "season": year,
                "league": item.get("leagueAbbreviation", "MLB"),
                "games": item.get("gamesPlayed"),
                "avg_game_duration_min": item.get("avgGameDurationMinutes"),
                "avg_innings": item.get("avgInningsPlayed"),
                "runs_per_game": item.get("runsPerGame"),
                "pitches_per_pa": item.get("pitchesPerPlateAppearance"),
            })
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "fetch_game_pace failed for %d: %s", year, exc
        )
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
                rows.append({
                    "team": team_info.get("name", ""),
                    "streak_type": stat_type,
                    "streak_length": length,
                    "season": year,
                })
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "fetch_streaks failed for %d/%s: %s", year, streak_type, exc
        )
        return pd.DataFrame()


if __name__ == "__main__":
    fetch_all_schedules()
