# 02 – Data Ingestion Pipelines

This document covers retrieving data from every source in [01-data-sources.md](01-data-sources.md) and landing it into local storage (CSV → PostgreSQL).

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  MLB Stats   │────▶│              │────▶│  CSV / Parquet  │
│  pybaseball  │     │  Ingestion   │     │  data_files/    │
│  Retrosheet  │     │  Scripts     │     │  (primary)      │
│  Odds API    │     │  (Python)    │     └────────────────┘
│  Weather     │     │              │            │
└─────────────┘     └──────────────┘     ┌────────────────┐
                           │               │ PostgreSQL     │
                     APScheduler / cron    │ (optional)     │
                     (daily + hourly)      └────────────────┘
```

---

## Project Structure

```
baseball-predictions/
├── data_files/              # Primary data store (CSV + Parquet)
│   ├── raw/
│   │   ├── gamelogs/
│   │   ├── batting/
│   │   ├── pitching/
│   │   ├── odds/
│   │   └── weather/
│   └── processed/           # Parquet files for model consumption
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── mlb_stats.py
│   │   ├── pybaseball_stats.py
│   │   ├── retrosheet.py
│   │   ├── odds.py
│   │   ├── weather.py
│   │   └── loader.py        # CSV → Parquet consolidator
│   └── ...
├── requirements.txt
└── ...
```

---

## 1. Configuration

```python
# src/ingestion/config.py
from pathlib import Path
from dataclasses import dataclass
import os

@dataclass
class IngestionConfig:
    """Central config for all ingestion jobs."""
    
    # Date range
    start_year: int = 2021
    end_year: int = 2025
    
    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_dir: Path = project_root / "data_files" / "raw"
    processed_dir: Path = project_root / "data_files" / "processed"
    
    # API keys (from environment variables)
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    
    # Rate limiting
    request_delay_sec: float = 1.0   # polite delay between API calls
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for subdir in ["gamelogs", "batting", "pitching", "odds", "weather"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

config = IngestionConfig()
```

---

## 2. MLB Stats API Ingestion

```python
# src/ingestion/mlb_stats.py
"""Pull schedules, game results, and probable pitchers from the MLB Stats API."""

import statsapi
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
from .config import config

def fetch_season_schedule(year: int) -> pd.DataFrame:
    """Fetch every game for a given season.
    
    Returns DataFrame with: game_id, date, away_team, home_team,
    away_score, home_score, status, venue, away_pitcher, home_pitcher.
    """
    start = f"{year}-02-20"   # Spring training start (adjust as needed)
    end = f"{year}-11-05"     # Include postseason
    
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
    """Fetch schedules for all configured years and save."""
    all_dfs = []
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
```

---

## 3. pybaseball Stats Ingestion

```python
# src/ingestion/pybaseball_stats.py
"""Pull batting and pitching stats via pybaseball (FanGraphs + Statcast)."""

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
        year: Season year
        qual: Minimum plate appearances to qualify
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
    import calendar
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


def fetch_all_stats():
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
```

---

## 4. Odds Ingestion

```python
# src/ingestion/odds.py
"""Fetch current and historical odds from The Odds API."""

import requests
import pandas as pd
from datetime import datetime
from .config import config


def fetch_current_odds(
    markets: str = "h2h,spreads,totals",
    bookmakers: str = "draftkings,fanduel,betmgm,caesars,pointsbet",
) -> pd.DataFrame:
    """Fetch live MLB odds for today's games.
    
    Markets:
        h2h      = moneyline (underdog picks)
        spreads  = run line (+/- 1.5 typically)
        totals   = over/under
    """
    if not config.odds_api_key:
        raise ValueError("Set ODDS_API_KEY environment variable")
    
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": config.odds_api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": bookmakers,
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Odds API requests remaining: {remaining}")
    
    games = resp.json()
    rows = []
    
    for game in games:
        game_id = game["id"]
        away = game["away_team"]
        home = game["home_team"]
        commence = game["commence_time"]
        
        for book in game.get("bookmakers", []):
            book_name = book["key"]
            for market in book.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "away_team": away,
                        "home_team": home,
                        "bookmaker": book_name,
                        "market": market_key,
                        "outcome_name": outcome["name"],
                        "outcome_price": outcome["price"],
                        "outcome_point": outcome.get("point"),
                        "fetched_at": datetime.utcnow().isoformat(),
                    })
    
    df = pd.DataFrame(rows)
    
    # Save with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = config.raw_dir / "odds" / f"odds_{ts}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} odds rows → {outpath}")
    
    return df


def get_consensus_line(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the consensus (median) line across bookmakers for each game/market."""
    consensus = (
        df.groupby(["game_id", "away_team", "home_team", "market", "outcome_name"])
        .agg(
            median_price=("outcome_price", "median"),
            mean_price=("outcome_price", "mean"),
            median_point=("outcome_point", "median"),
            num_books=("bookmaker", "nunique"),
        )
        .reset_index()
    )
    return consensus


if __name__ == "__main__":
    odds_df = fetch_current_odds()
    consensus = get_consensus_line(odds_df)
    print(consensus.to_string())
```

---

## 5. Weather Ingestion

```python
# src/ingestion/weather.py
"""Fetch weather data for upcoming games using Open-Meteo (free, no key)."""

import requests
import pandas as pd
from datetime import datetime
from .config import config

# Ballpark coordinates (subset; full list in 01-data-sources.md)
BALLPARK_COORDS = {
    "Wrigley Field":            (41.9484, -87.6553),
    "Yankee Stadium":           (40.8296, -73.9262),
    "Fenway Park":              (42.3467, -71.0972),
    "Dodger Stadium":           (34.0739, -118.2400),
    "Coors Field":              (39.7559, -104.9942),
    "Oracle Park":              (37.7786, -122.3893),
    "Globe Life Field":         (32.7473, -97.0845),
    "Minute Maid Park":         (29.7572, -95.3555),
    "Truist Park":              (33.8908, -84.4678),
    "Citizens Bank Park":       (39.9061, -75.1665),
    "T-Mobile Park":            (47.5914, -122.3325),
    "Petco Park":               (32.7076, -117.1570),
    "Busch Stadium":            (38.6226, -90.1928),
    "Target Field":             (44.9818, -93.2775),
    "PNC Park":                 (40.4469, -80.0058),
    "Camden Yards":             (39.2838, -76.6216),
    "Rogers Centre":            (43.6414, -79.3894),
    "Tropicana Field":          (27.7682, -82.6534),
    "Kauffman Stadium":         (39.0517, -94.4803),
    "Comerica Park":            (42.3390, -83.0485),
    "Progressive Field":        (41.4962, -81.6852),
    "Great American Ball Park": (39.0975, -84.5069),
    "American Family Field":    (43.0280, -87.9712),
    "Chase Field":              (33.4455, -112.0667),
    "loanDepot Park":           (25.7781, -80.2197),
    "Citi Field":               (40.7571, -73.8458),
    "Angel Stadium":            (33.8003, -117.8827),
    "Nationals Park":           (38.8730, -77.0074),
    "Guaranteed Rate Field":    (41.8299, -87.6338),
    "Oakland Coliseum":         (37.7516, -122.2005),
}

# Retractable/domed stadiums where weather matters less
DOMED_STADIUMS = {
    "Globe Life Field", "Tropicana Field", "Minute Maid Park",
    "loanDepot Park", "Rogers Centre", "Chase Field",
    "American Family Field", "T-Mobile Park",
}


def fetch_weather_for_venue(
    venue_name: str, game_date: str, game_hour: int = 19
) -> dict | None:
    """Get weather at a venue for a specific date/hour.
    
    Returns None for domed stadiums.
    """
    if venue_name in DOMED_STADIUMS:
        return {"venue": venue_name, "dome": True}
    
    coords = BALLPARK_COORDS.get(venue_name)
    if not coords:
        print(f"  Warning: no coords for '{venue_name}'")
        return None
    
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability",
        "start_date": game_date,
        "end_date": game_date,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "auto",
    }
    
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    
    data = resp.json().get("hourly", {})
    idx = min(game_hour, len(data.get("temperature_2m", [])) - 1)
    
    return {
        "venue": venue_name,
        "dome": False,
        "temp_f": data["temperature_2m"][idx],
        "wind_mph": data["windspeed_10m"][idx],
        "wind_dir_deg": data["winddirection_10m"][idx],
        "precip_prob_pct": data["precipitation_probability"][idx],
    }


def fetch_weather_for_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Given a schedule DataFrame (must have 'venue' and 'date' columns),
    enrich with weather data."""
    weather_rows = []
    
    for _, row in schedule_df.iterrows():
        venue = row.get("venue", "")
        date = row.get("date", "")[:10]  # 'YYYY-MM-DD'
        w = fetch_weather_for_venue(venue, date)
        if w:
            w["game_id"] = row.get("game_id")
            weather_rows.append(w)
    
    wdf = pd.DataFrame(weather_rows)
    ts = datetime.now().strftime("%Y%m%d")
    outpath = config.raw_dir / "weather" / f"weather_{ts}.csv"
    wdf.to_csv(outpath, index=False)
    print(f"Weather for {len(wdf)} games → {outpath}")
    return wdf


if __name__ == "__main__":
    # Quick test: fetch weather for one venue
    result = fetch_weather_for_venue("Wrigley Field", "2026-03-04", 19)
    print(result)
```

---

## 6. Retrosheet Historical Loader

```python
# src/ingestion/retrosheet.py
"""Download and parse Retrosheet game logs for deep historical data."""

import pandas as pd
import requests
import zipfile
import io
from time import sleep
from .config import config

# Key column indices in Retrosheet game logs
# Full spec: https://www.retrosheet.org/gamelogs/glfields.txt
GAMELOG_COLS = {
    0: "date",
    1: "game_number",
    2: "day_of_week",
    3: "away_team",
    4: "away_league",
    6: "home_team",
    7: "home_league",
    9: "away_score",
    10: "home_score",
    12: "day_night",
    16: "park_id",
    17: "attendance",
    18: "duration_minutes",
    # Pitchers
    101: "away_starting_pitcher_id",
    102: "away_starting_pitcher_name",
    103: "home_starting_pitcher_id",
    104: "home_starting_pitcher_name",
}


def download_gamelog(year: int) -> pd.DataFrame:
    """Download and parse a single year's game log from Retrosheet."""
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
    
    # Compute derived fields
    df["total_runs"] = df["away_score"] + df["home_score"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["run_diff"] = df["home_score"] - df["away_score"]
    
    outpath = config.raw_dir / "gamelogs" / f"retrosheet_{year}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} games → {outpath}")
    
    return df


def download_all_gamelogs() -> pd.DataFrame:
    """Download game logs for all configured years."""
    all_dfs = []
    for year in range(config.start_year, config.end_year + 1):
        try:
            df = download_gamelog(year)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error for {year}: {e}")
        sleep(2)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(config.raw_dir / "gamelogs" / "retrosheet_all.csv", index=False)
        print(f"\nCombined: {len(combined)} total games")
        return combined
    return pd.DataFrame()


if __name__ == "__main__":
    download_all_gamelogs()
```

---

## 7. CSV → Parquet Consolidator

```python
# src/ingestion/loader.py
"""Consolidate raw CSV files into optimized Parquet files.

Parquet is the preferred storage format:
- Columnar compression (5-10x smaller than CSV)
- Fast reads with pandas/pyarrow
- Schema enforcement
- Works well with Git LFS for version control

PostgreSQL is available as an optional secondary store.
"""

import pandas as pd
from pathlib import Path
from .config import config
import os


def csv_to_parquet(csv_path: Path, parquet_path: Path, **kwargs):
    """Convert a single CSV to Parquet."""
    df = pd.read_csv(csv_path, **kwargs)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    csv_size = csv_path.stat().st_size / 1024
    pq_size = parquet_path.stat().st_size / 1024
    print(f"  {csv_path.name} ({csv_size:.0f} KB) → {parquet_path.name} ({pq_size:.0f} KB)")
    return df


def consolidate_all():
    """Consolidate all raw CSVs into processed Parquet files."""
    processed = config.processed_dir
    
    # Schedules
    sched_csv = config.raw_dir / "gamelogs" / "schedule_all.csv"
    if sched_csv.exists():
        csv_to_parquet(sched_csv, processed / "schedules.parquet")
    
    # Retrosheet game logs
    retro_csv = config.raw_dir / "gamelogs" / "retrosheet_all.csv"
    if retro_csv.exists():
        csv_to_parquet(retro_csv, processed / "gamelogs.parquet")
    
    # Batting stats (combine yearly files)
    bat_files = sorted(config.raw_dir.glob("batting/batting_*.csv"))
    bat_files = [f for f in bat_files if "team_" not in f.name and "statcast" not in f.name]
    if bat_files:
        dfs = [pd.read_csv(f) for f in bat_files]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "batting_stats.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(bat_files)} batting files → {out.name} ({len(combined)} rows)")
    
    # Pitching stats
    pitch_files = sorted(config.raw_dir.glob("pitching/pitching_*.csv"))
    pitch_files = [f for f in pitch_files if "team_" not in f.name]
    if pitch_files:
        dfs = [pd.read_csv(f) for f in pitch_files]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "pitching_stats.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(pitch_files)} pitching files → {out.name} ({len(combined)} rows)")
    
    # Team stats
    for stat_type in ["batting", "pitching"]:
        team_files = sorted(config.raw_dir.glob(f"{stat_type}/team_{stat_type}_*.csv"))
        if team_files:
            dfs = [pd.read_csv(f) for f in team_files]
            combined = pd.concat(dfs, ignore_index=True)
            out = processed / f"team_{stat_type}.parquet"
            combined.to_parquet(out, index=False, engine="pyarrow")
            print(f"  {len(team_files)} team {stat_type} files → {out.name}")
    
    # Odds (append-friendly — keep CSVs, create daily Parquet)
    odds_csvs = sorted(config.raw_dir.glob("odds/odds_*.csv"))
    if odds_csvs:
        dfs = [pd.read_csv(f) for f in odds_csvs]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "odds_history.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(odds_csvs)} odds snapshots → {out.name} ({len(combined)} rows)")
    
    print("\nConsolidation complete.")
    _report_sizes(processed)


def _report_sizes(directory: Path):
    """Print file sizes for all Parquet files."""
    print("\nProcessed data files:")
    for f in sorted(directory.glob("*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        df = pd.read_parquet(f)
        print(f"  {f.name:30s}  {size_mb:6.1f} MB  {len(df):>8,} rows")


# ---- Optional: Load Parquet into PostgreSQL ----

def load_parquet_to_postgres(parquet_path: Path, table_name: str, if_exists: str = "replace"):
    """Optionally load a Parquet file into PostgreSQL.
    
    Only needed if you want SQL query access or are scaling beyond flat files.
    """
    from sqlalchemy import create_engine
    
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/baseball_predictions"
    )
    engine = create_engine(DATABASE_URL, echo=False)
    
    df = pd.read_parquet(parquet_path)
    print(f"Loading {len(df)} rows from {parquet_path.name} → {table_name}...")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"  Done.")


if __name__ == "__main__":
    consolidate_all()
```

---

## 8. Daily Scheduler

```python
# src/ingestion/scheduler.py
"""APScheduler-based daily data refresh."""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from .mlb_stats import fetch_todays_probable_pitchers
from .odds import fetch_current_odds
from .weather import fetch_weather_for_games
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()


@scheduler.scheduled_job(CronTrigger(hour=8, minute=0, timezone="America/New_York"))
def morning_data_pull():
    """8 AM ET – Pull today's schedule and probable pitchers."""
    logger.info("Running morning data pull...")
    schedule = fetch_todays_probable_pitchers()
    logger.info(f"Found {len(schedule)} games today")


@scheduler.scheduled_job(CronTrigger(hour=11, minute=0, timezone="America/New_York"))
def midday_odds_pull():
    """11 AM ET – Pull opening odds for today's slate."""
    logger.info("Running midday odds pull...")
    fetch_current_odds()


@scheduler.scheduled_job(CronTrigger(hour=16, minute=0, timezone="America/New_York"))
def afternoon_update():
    """4 PM ET – Pull updated odds + weather (closer to game time)."""
    logger.info("Running afternoon update...")
    odds = fetch_current_odds()
    schedule = fetch_todays_probable_pitchers()
    fetch_weather_for_games(schedule)


@scheduler.scheduled_job(CronTrigger(hour=1, minute=0, timezone="America/New_York"))
def overnight_results():
    """1 AM ET – Pull final scores and update model results."""
    logger.info("Running overnight results collection...")
    # TODO: fetch final scores, update pick results, recalculate metrics


if __name__ == "__main__":
    logger.info("Starting ingestion scheduler...")
    scheduler.start()
```

---

## 9. Requirements

```text
# requirements.txt additions for ingestion
MLB-StatsAPI>=1.7.0
pybaseball>=2.3.0
requests>=2.31.0
requests-cache>=1.1.0
pandas>=2.1.0
pyarrow>=14.0.0
apsched>=3.10.0
python-dotenv>=1.0.0
```

---

## Running the Full Historical Backfill

```bash
# 1. Set up environment
cd baseball-predictions
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Set env vars
export ODDS_API_KEY="your-key-here"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/baseball_predictions"

# 3. Run historical ingestion
python -m src.ingestion.mlb_stats          # Schedules & games
python -m src.ingestion.pybaseball_stats   # Batting & pitching stats
python -m src.ingestion.retrosheet         # Deep historical game logs

# 4. Load into Parquet
python -m src.ingestion.loader

# 5. (Optional) Load into PostgreSQL
# python -c "from src.ingestion.loader import load_parquet_to_postgres; ..."
```

---

> **Next:** [03-database-schema.md](03-database-schema.md) – Designing the tables to hold all of this data.
