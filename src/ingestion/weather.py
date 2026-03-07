# src/ingestion/weather.py
"""Fetch weather data for upcoming games using Open-Meteo (free, no key required)."""

import requests
import pandas as pd
from datetime import datetime

from .config import config

# Ballpark coordinates
BALLPARK_COORDS: dict[str, tuple[float, float]] = {
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
DOMED_STADIUMS: set[str] = {
    "Globe Life Field",
    "Tropicana Field",
    "Minute Maid Park",
    "loanDepot Park",
    "Rogers Centre",
    "Chase Field",
    "American Family Field",
    "T-Mobile Park",
}


def fetch_weather_for_venue(
    venue_name: str, game_date: str, game_hour: int = 19
) -> dict | None:
    """Get weather at a venue for a specific date/hour.

    Args:
        venue_name: Stadium name (must match a key in BALLPARK_COORDS).
        game_date:  Date string in 'YYYY-MM-DD' format.
        game_hour:  Local hour for which to return weather (0-23).

    Returns:
        Dict with weather fields, or None if venue is unknown.
        Domed stadiums return ``{"venue": ..., "dome": True}``.
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

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return None

    data = resp.json().get("hourly", {})
    temps = data.get("temperature_2m", [])
    if not temps:
        return None
    idx = min(game_hour, len(temps) - 1)

    return {
        "venue": venue_name,
        "dome": False,
        "temp_f": data["temperature_2m"][idx],
        "wind_mph": data["windspeed_10m"][idx],
        "wind_dir_deg": data["winddirection_10m"][idx],
        "precip_prob_pct": data["precipitation_probability"][idx],
    }


def fetch_weather_for_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a schedule DataFrame with weather data.

    Args:
        schedule_df: Must contain 'venue' and 'date' columns.

    Returns:
        DataFrame of weather rows joined by game_id.
    """
    weather_rows = []

    for _, row in schedule_df.iterrows():
        venue = row.get("venue", "")
        date = str(row.get("date", ""))[:10]  # 'YYYY-MM-DD'
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
    result = fetch_weather_for_venue("Wrigley Field", datetime.now().strftime("%Y-%m-%d"), 19)
    print(result)
