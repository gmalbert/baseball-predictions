# src/ingestion/retrosheet.py
"""Download and parse Retrosheet game logs for deep historical data."""

import io
import zipfile
from time import sleep

import pandas as pd
import requests

from .config import config

# Key column indices in Retrosheet game logs.
# Full spec: https://www.retrosheet.org/gamelogs/glfields.txt
GAMELOG_COLS: dict[int, str] = {
    0:   "date",
    1:   "game_number",
    2:   "day_of_week",
    3:   "away_team",
    4:   "away_league",
    6:   "home_team",
    7:   "home_league",
    9:   "away_score",
    10:  "home_score",
    12:  "day_night",
    16:  "park_id",
    17:  "attendance",
    18:  "duration_minutes",
    # Starting pitchers
    101: "away_starting_pitcher_id",
    102: "away_starting_pitcher_name",
    103: "home_starting_pitcher_id",
    104: "home_starting_pitcher_name",
}


def download_gamelog(year: int) -> pd.DataFrame:
    """Download and parse a single year's game log from Retrosheet.

    Args:
        year: Season year (e.g. 2023).

    Returns:
        DataFrame with renamed columns and derived fields.
    """
    url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
    print(f"Downloading Retrosheet game log for {year}...")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = z.namelist()[0]

    df = pd.read_csv(z.open(csv_name), header=None, low_memory=False)

    # Rename known columns
    rename_map = {k: v for k, v in GAMELOG_COLS.items() if k < len(df.columns)}
    df.rename(columns=rename_map, inplace=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Derived fields
    df["total_runs"] = df["away_score"] + df["home_score"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["run_diff"] = df["home_score"] - df["away_score"]

    outpath = config.raw_dir / "gamelogs" / f"retrosheet_{year}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} games → {outpath}")

    return df


def download_all_gamelogs() -> pd.DataFrame:
    """Download game logs for all configured years and combine."""
    all_dfs: list[pd.DataFrame] = []
    for year in range(config.start_year, config.end_year + 1):
        try:
            df = download_gamelog(year)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error for {year}: {e}")
        sleep(2)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(
            config.raw_dir / "gamelogs" / "retrosheet_all.csv", index=False
        )
        print(f"\nCombined: {len(combined)} total games")
        return combined
    return pd.DataFrame()


if __name__ == "__main__":
    download_all_gamelogs()
