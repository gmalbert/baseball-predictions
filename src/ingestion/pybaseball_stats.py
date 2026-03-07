# src/ingestion/pybaseball_stats.py
"""Pull batting and pitching stats via pybaseball (FanGraphs + Statcast)."""

import calendar
import pandas as pd
from pybaseball import (
    batting_stats,
    pitching_stats,
    team_batting,
    team_pitching,
    statcast,
)
from time import sleep

from .config import config


def fetch_batting_stats(year: int, qual: int = 50) -> pd.DataFrame:
    """Fetch FanGraphs batting leaderboard for a season.

    Args:
        year: Season year.
        qual: Minimum plate appearances to qualify.
    """
    print(f"Fetching batting stats for {year} (qual={qual} PA)...")
    df = batting_stats(year, qual=qual)
    outpath = config.raw_dir / "batting" / f"batting_{year}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} batters → {outpath}")
    return df


def fetch_pitching_stats(year: int, qual: int = 20) -> pd.DataFrame:
    """Fetch FanGraphs pitching leaderboard for a season."""
    print(f"Fetching pitching stats for {year} (qual={qual} IP)...")
    df = pitching_stats(year, qual=qual)
    outpath = config.raw_dir / "pitching" / f"pitching_{year}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} pitchers → {outpath}")
    return df


def fetch_team_stats(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch team-level batting and pitching aggregates."""
    print(f"Fetching team stats for {year}...")
    tb = team_batting(year)
    tp = team_pitching(year)

    tb.to_csv(config.raw_dir / "batting" / f"team_batting_{year}.csv", index=False)
    tp.to_csv(config.raw_dir / "pitching" / f"team_pitching_{year}.csv", index=False)
    return tb, tp


def fetch_statcast_monthly(year: int, month: int) -> pd.DataFrame:
    """Fetch Statcast pitch-level data for one month.

    Statcast queries are LARGE — always chunk by month or smaller.
    """
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{last_day}"

    print(f"Fetching Statcast data: {start} → {end}...")
    df = statcast(start_dt=start, end_dt=end)
    if df is not None and len(df) > 0:
        outpath = config.raw_dir / "batting" / f"statcast_{year}_{month:02d}.csv"
        df.to_csv(outpath, index=False)
        print(f"  {len(df)} pitches → {outpath}")
    return df


def fetch_all_stats() -> None:
    """Master function: pull all batting and pitching stats for configured years."""
    for year in range(config.start_year, config.end_year + 1):
        fetch_batting_stats(year)
        sleep(2)
        fetch_pitching_stats(year)
        sleep(2)
        fetch_team_stats(year)
        sleep(2)

    print("\nAll batting/pitching stats ingested.")


if __name__ == "__main__":
    fetch_all_stats()
