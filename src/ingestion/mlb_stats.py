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


if __name__ == "__main__":
    fetch_all_schedules()
