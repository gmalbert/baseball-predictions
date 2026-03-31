"""Bulk-fetch Open-Meteo archive weather for all MLB games [min_year → today].

Run once after updating Retrosheet CSV/Parquet data, or at the start of each
season to extend coverage:

    python scripts/fetch_weather_history.py

Makes ~30 API calls (one per stadium), each covering the full date range.
Saves to:  data_files/processed/weather_historical.parquet

Columns:
    gid              Retrosheet game ID
    temp_f           Average temperature (°F) during game-time hours
    wind_mph         Average wind speed (mph) during game-time hours
    wind_dir_deg     Average wind direction (degrees) during game-time hours
    precip_mm        Total precipitation (mm) during game-time hours
    humidity_pct     Average relative humidity (%) during game-time hours
    cloud_cover_pct  Average cloud cover (%) during game-time hours
    is_dome          True if the stadium has a retractable / fixed roof
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.ingestion.weather import build_historical_weather

RETRO_DIR  = ROOT / "data_files" / "retrosheet"
PROCESSED  = ROOT / "data_files" / "processed"
MIN_YEAR   = 2020
OUT_PATH   = PROCESSED / "weather_historical.parquet"


def main() -> None:
    parquet = RETRO_DIR / "gameinfo.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found.\n"
            "Run  python scripts/build_parquet_data.py  first."
        )

    print(f"Loading gameinfo parquet ({parquet}) …")
    gi = pd.read_parquet(parquet)
    gi["season"] = pd.to_numeric(gi["season"], errors="coerce")
    gi = gi[gi["season"] >= MIN_YEAR].copy()
    print(f"  {len(gi):,} games with season ≥ {MIN_YEAR}\n")

    # Skip gids already written so a rate-limited run can be resumed
    existing_df: pd.DataFrame | None = None
    already_done: set[str] = set()
    if OUT_PATH.exists():
        existing_df = pd.read_parquet(OUT_PATH)   # load BEFORE build call overwrites file
        already_done = set(existing_df["gid"].dropna())
        if already_done:
            print(f"  Resuming — {len(already_done):,} gids already in parquet, skipping them.\n")
            gi = gi[~gi["gid"].isin(already_done)]
            if gi.empty:
                print("All games already fetched. Nothing to do.")
                return

    print("Fetching historical weather (one API call per stadium) …\n")
    df = build_historical_weather(gi, min_year=MIN_YEAR, verbose=True)

    if existing_df is not None and not df.empty:
        # build_historical_weather already wrote only the new rows; merge with originals
        merged = pd.concat([existing_df, df], ignore_index=True)
        merged.to_parquet(OUT_PATH, index=False)
        print(f"\nDone. {len(merged):,} total game-weather rows in parquet ({len(df):,} new).")
    else:
        print(f"\nDone. {len(df):,} game-weather rows written.")


if __name__ == "__main__":
    main()
