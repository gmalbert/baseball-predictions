# src/ingestion/weather.py
"""MLB weather from Open-Meteo (free, no API key required).

Two modes:
  forecast  — fetch_forecast(venue_name, "YYYY-MM-DD")
              Uses archive API for past dates, forecast API for upcoming/today.
  archive   — build_historical_weather(gameinfo_df, min_year=2020)
              Batch-fetches one API call per stadium, saves to
              data_files/processed/weather_historical.parquet.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# data_files/processed/ — two levels up from src/ingestion/
_PROCESSED = Path(__file__).resolve().parents[2] / "data_files" / "processed"

_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HOURLY_VARS  = (
    "temperature_2m,wind_speed_10m,wind_direction_10m,"
    "precipitation,relative_humidity_2m,cloud_cover"
)
_GAME_HOUR_START = 13   # 1 PM local  } game-time window used for
_GAME_HOUR_END   = 21   # 9 PM local  } averaging hourly data

# ---------------------------------------------------------------------------
# Stadium coordinates: (lat, lon, is_dome)
# Keys = exact venue_name values returned by statsapi.schedule()
# ---------------------------------------------------------------------------
BALLPARK_COORDS: dict[str, tuple[float, float, bool]] = {
    # AL East
    "Fenway Park":                  (42.3467, -71.0972, False),
    "Yankee Stadium":               (40.8296, -73.9262, False),
    "Citi Field":                   (40.7571, -73.8458, False),
    "Camden Yards":                 (39.2838, -76.6216, False),
    "Oriole Park at Camden Yards":  (39.2838, -76.6216, False),
    "Nationals Park":               (38.8730, -77.0074, False),
    "Tropicana Field":              (27.7682, -82.6534, True),
    "Rogers Centre":                (43.6414, -79.3894, True),
    # AL Central
    "Guaranteed Rate Field":        (41.8299, -87.6338, False),
    "Progressive Field":            (41.4962, -81.6852, False),
    "Comerica Park":                (42.3390, -83.0485, False),
    "American Family Field":        (43.0280, -87.9712, True),
    "Target Field":                 (44.9818, -93.2775, False),
    "Kauffman Stadium":             (39.0517, -94.4803, False),
    # AL West
    "Globe Life Field":             (32.7473, -97.0845, True),   # Rangers — retractable roof 2020+
    "Minute Maid Park":             (29.7572, -95.3555, True),
    "Angel Stadium":                (33.8003, -117.8827, False),
    "Oakland Coliseum":             (37.7516, -122.2005, False),
    "Sutter Health Park":           (38.5802, -121.5003, False),  # A's Sacramento 2025+
    "T-Mobile Park":                (47.5914, -122.3325, True),
    # NL East
    "Citizens Bank Park":           (39.9061, -75.1665, False),
    "PNC Park":                     (40.4469, -80.0058, False),
    "Truist Park":                  (33.8908, -84.4678, False),
    "loanDepot Park":               (25.7781, -80.2197, True),
    # NL Central
    "Wrigley Field":                (41.9484, -87.6553, False),
    "Great American Ball Park":     (39.0975, -84.5069, False),
    "Busch Stadium":                (38.6226, -90.1928, False),
    # NL West
    "Dodger Stadium":               (34.0739, -118.2400, False),
    "Petco Park":                   (32.7076, -117.1570, False),
    "Oracle Park":                  (37.7786, -122.3893, False),
    "Coors Field":                  (39.7559, -104.9942, False),
    "Chase Field":                  (33.4455, -112.0667, True),
}

# Retrosheet team code → (lat, lon, is_dome)
# Keys = Retrosheet 3-letter codes as stored in gameinfo.parquet's hometeam column.
HOME_TEAM_COORDS: dict[str, tuple[float, float, bool]] = {
    "BOS": (42.3467, -71.0972, False),   # Red Sox — Fenway Park
    "NYA": (40.8296, -73.9262, False),   # Yankees — Yankee Stadium
    "NYN": (40.7571, -73.8458, False),   # Mets — Citi Field
    "BAL": (39.2838, -76.6216, False),   # Orioles — Camden Yards
    "WAS": (38.8730, -77.0074, False),   # Nationals — Nationals Park
    "WSN": (38.8730, -77.0074, False),   # Nationals (alt code)
    "TBA": (27.7682, -82.6534, True),    # Rays — Tropicana Field
    "TBD": (27.7682, -82.6534, True),    # Rays (alt code)
    "TOR": (43.6414, -79.3894, True),    # Blue Jays — Rogers Centre
    "CHA": (41.8299, -87.6338, False),   # White Sox — Guaranteed Rate Field
    "CLE": (41.4962, -81.6852, False),   # Guardians — Progressive Field
    "DET": (42.3390, -83.0485, False),   # Tigers — Comerica Park
    "MIL": (43.0280, -87.9712, True),    # Brewers — American Family Field
    "MIN": (44.9818, -93.2775, False),   # Twins — Target Field
    "KCA": (39.0517, -94.4803, False),   # Royals — Kauffman Stadium
    "TEX": (32.7473, -97.0845, True),    # Rangers — Globe Life Field (retractable 2020+)
    "HOU": (29.7572, -95.3555, True),    # Astros — Minute Maid Park
    "ANA": (33.8003, -117.8827, False),  # Angels — Angel Stadium
    "CAL": (33.8003, -117.8827, False),  # Angels (alt code)
    "OAK": (37.7516, -122.2005, False),  # Athletics — Oakland Coliseum (thru 2024)
    "ATH": (38.5802, -121.5003, False),  # Athletics — Sutter Health Park (Sacramento 2025+)
    "SEA": (47.5914, -122.3325, True),   # Mariners — T-Mobile Park
    "PHI": (39.9061, -75.1665, False),   # Phillies — Citizens Bank Park
    "PIT": (40.4469, -80.0058, False),   # Pirates — PNC Park
    "ATL": (33.8908, -84.4678, False),   # Braves — Truist Park
    "MIA": (25.7781, -80.2197, True),    # Marlins — loanDepot Park
    "FLO": (25.7781, -80.2197, True),    # Marlins (alt code)
    "CHN": (41.9484, -87.6553, False),   # Cubs — Wrigley Field
    "CIN": (39.0975, -84.5069, False),   # Reds — Great American Ball Park
    "SLN": (38.6226, -90.1928, False),   # Cardinals — Busch Stadium
    "LAN": (34.0739, -118.2400, False),  # Dodgers — Dodger Stadium
    "SDN": (32.7076, -117.1570, False),  # Padres — Petco Park
    "SFN": (37.7786, -122.3893, False),  # Giants — Oracle Park
    "COL": (39.7559, -104.9942, False),  # Rockies — Coors Field
    "ARI": (33.4455, -112.0667, True),   # Diamondbacks — Chase Field
}

_DOME_PRESET: dict = {
    "is_dome":         True,
    "temp_f":          72.0,
    "wind_mph":        0.0,
    "wind_dir_deg":    0.0,
    "precip_mm":       0.0,
    "humidity_pct":    50.0,
    "cloud_cover_pct": 0.0,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_venue(name: str) -> tuple[float, float, bool] | None:
    """Exact-match then longest-substring fallback against BALLPARK_COORDS."""
    if name in BALLPARK_COORDS:
        return BALLPARK_COORDS[name]
    nl = name.lower()
    for k, v in BALLPARK_COORDS.items():
        if k.lower() in nl or nl in k.lower():
            return v
    return None


def _get_json(url: str, params: dict, retries: int = 4) -> dict:
    """GET with exponential-backoff retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (2 ** attempt))  # 5s, 10s, 20s
    return {}


def _extract_game_hours(data: dict) -> dict:
    """Aggregate hourly Open-Meteo response into one game-time summary."""
    h   = data.get("hourly", {})
    sl  = slice(_GAME_HOUR_START, _GAME_HOUR_END + 1)
    def _arr(key: str, fill: float = 0.0) -> np.ndarray:
        vals = h.get(key, [fill] * 24)
        return np.array(vals, dtype=float)[sl]
    return {
        "is_dome":         False,
        "temp_f":          float(np.nanmean(_arr("temperature_2m",    np.nan))),
        "wind_mph":        float(np.nanmean(_arr("wind_speed_10m"))),
        "wind_dir_deg":    float(np.nanmean(_arr("wind_direction_10m"))),
        "precip_mm":       float(np.nansum(_arr("precipitation"))),
        "humidity_pct":    float(np.nanmean(_arr("relative_humidity_2m", 50.0))),
        "cloud_cover_pct": float(np.nanmean(_arr("cloud_cover"))),
    }


# ---------------------------------------------------------------------------
# Public API — single-game look-up
# ---------------------------------------------------------------------------

def fetch_forecast(venue_name: str, game_date_str: str) -> dict | None:
    """Weather for a stadium on a specific date.

    Automatically routes to the archive API for past dates and the forecast
    API for today / upcoming games (up to 16 days out).

    Args:
        venue_name:     Stadium name matching a key in BALLPARK_COORDS.
        game_date_str:  ``'YYYY-MM-DD'``

    Returns:
        Dict with keys: ``is_dome, temp_f, wind_mph, wind_dir_deg,
        precip_mm, humidity_pct, cloud_cover_pct``
        or ``None`` if the venue is unknown or the call fails.
    """
    coords = _resolve_venue(venue_name)
    if coords is None:
        return None
    lat, lon, is_dome = coords
    if is_dome:
        return dict(_DOME_PRESET)

    game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
    url = _ARCHIVE_URL if game_date < date.today() else _FORECAST_URL
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           _HOURLY_VARS,
        "start_date":       game_date_str,
        "end_date":         game_date_str,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone":         "auto",
    }
    try:
        return _extract_game_hours(_get_json(url, params))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API — bulk historical fetch (one API call per stadium)
# ---------------------------------------------------------------------------

def build_historical_weather(
    gameinfo_df: pd.DataFrame,
    min_year: int = 2020,
    verbose: bool = True,
) -> pd.DataFrame:
    """Batch-fetch Open-Meteo archive weather for all historical MLB games.

    Groups games by home team → makes one archive API call per stadium
    covering the full date range (``min_year-01-01`` to today).
    ~30 API calls instead of ~14,000 individual per-game calls.

    Saves results to ``data_files/processed/weather_historical.parquet``.

    Args:
        gameinfo_df:  DataFrame with ``gid``, ``hometeam``, ``date`` columns.
                      ``date`` must be parseable by ``pd.to_datetime``.
        min_year:     Earliest season to include (default 2020).
        verbose:      Print per-stadium progress.

    Returns:
        DataFrame: gid, temp_f, wind_mph, wind_dir_deg,
                   precip_mm, humidity_pct, cloud_cover_pct, is_dome
    """
    today_str = date.today().isoformat()
    start_str = f"{min_year}-01-01"

    gdf = gameinfo_df.copy()
    # date column may be int YYYYMMDD (Retrosheet) or a date string — handle both
    date_col = gdf["date"]
    if pd.api.types.is_integer_dtype(date_col) or pd.api.types.is_float_dtype(date_col):
        gdf["_date_ts"] = pd.to_datetime(date_col.astype(str), format="%Y%m%d", errors="coerce")
    else:
        gdf["_date_ts"] = pd.to_datetime(date_col, errors="coerce")
    gdf = gdf.dropna(subset=["_date_ts", "gid", "hometeam"])
    gdf = gdf[gdf["_date_ts"].dt.year >= min_year]

    rows: list[dict] = []
    teams = sorted(gdf["hometeam"].dropna().unique())

    for team in teams:
        coords = HOME_TEAM_COORDS.get(team)
        if coords is None:
            if verbose:
                print(f"  [skip] no coords for '{team}'")
            continue
        lat, lon, is_dome = coords
        team_games = gdf[gdf["hometeam"] == team].copy()
        if verbose:
            print(f"  {team} ({len(team_games)} games)", end=" … ")

        if is_dome:
            for _, g in team_games.iterrows():
                rows.append({"gid": g["gid"], **_DOME_PRESET})
            if verbose:
                print("dome — no API call needed")
            continue

        params = {
            "latitude":         lat,
            "longitude":        lon,
            "hourly":           _HOURLY_VARS,
            "start_date":       start_str,
            "end_date":         today_str,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":         "auto",
        }
        try:
            data = _get_json(_ARCHIVE_URL, params)
            time.sleep(2)  # avoid Open-Meteo rate limit (~10 req/s free tier)
            h    = data.get("hourly", {})
            times = pd.DatetimeIndex(h.get("time", []))

            hourly_df = pd.DataFrame({
                "time":   times,
                "temp_f": pd.to_numeric(pd.Series(h.get("temperature_2m",     []), dtype=object), errors="coerce"),
                "wind":   pd.to_numeric(pd.Series(h.get("wind_speed_10m",     []), dtype=object), errors="coerce"),
                "wdir":   pd.to_numeric(pd.Series(h.get("wind_direction_10m", []), dtype=object), errors="coerce"),
                "precip": pd.to_numeric(pd.Series(h.get("precipitation",      []), dtype=object), errors="coerce"),
                "humid":  pd.to_numeric(pd.Series(h.get("relative_humidity_2m",[]), dtype=object), errors="coerce"),
                "cloud":  pd.to_numeric(pd.Series(h.get("cloud_cover",        []), dtype=object), errors="coerce"),
            })
            hourly_df["date_day"] = hourly_df["time"].dt.floor("D")
            hourly_df["hour"]     = hourly_df["time"].dt.hour
            game_hours = hourly_df[
                (hourly_df["hour"] >= _GAME_HOUR_START) &
                (hourly_df["hour"] <= _GAME_HOUR_END)
            ]
            daily = game_hours.groupby("date_day").agg(
                temp_f          =("temp_f", "mean"),
                wind_mph        =("wind",   "mean"),
                wind_dir_deg    =("wdir",   "mean"),
                precip_mm       =("precip", "sum"),
                humidity_pct    =("humid",  "mean"),
                cloud_cover_pct =("cloud",  "mean"),
            ).reset_index()

            team_games["_day"] = team_games["_date_ts"].dt.floor("D")
            merged = team_games[["gid", "_day"]].merge(
                daily, left_on="_day", right_on="date_day", how="left"
            )
            merged["is_dome"] = False
            rows.extend(
                merged[["gid", "temp_f", "wind_mph", "wind_dir_deg",
                         "precip_mm", "humidity_pct", "cloud_cover_pct", "is_dome"]]
                .to_dict("records")
            )
            if verbose:
                print(f"fetched {len(team_games)} records")
        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")

    _cols = ["gid", "temp_f", "wind_mph", "wind_dir_deg",
             "precip_mm", "humidity_pct", "cloud_cover_pct", "is_dome"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_cols)

    if not df.empty:
        _PROCESSED.mkdir(parents=True, exist_ok=True)
        out = _PROCESSED / "weather_historical.parquet"
        df.to_parquet(out, index=False)
        if verbose:
            print(f"\n  Saved {len(df):,} rows → {out}")
    return df


if __name__ == "__main__":
    result = fetch_forecast("Wrigley Field", date.today().isoformat())
    print(result)
