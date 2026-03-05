# 01 – Data Sources

This document catalogs every data source we will use (or can optionally add), what each provides, and how to access it.

---

## 1. MLB Stats API (free, official)

The **official MLB Stats API** (`statsapi.mlb.com`) is the single richest free source. Two excellent Python wrappers exist:

| Library | Repo | Notes |
|---------|------|-------|
| `MLB-StatsAPI` | <https://github.com/toddrob99/MLB-StatsAPI> | Lightweight, sync |
| `python-mlb-statsapi` | <https://github.com/zero-sum-seattle/python-mlb-statsapi> | Pydantic models, async-ready |

### What you get
- Game-by-game results (scores, innings, venue, attendance)
- Box scores, play-by-play, pitch-by-pitch
- Player season & career stats (batting, pitching, fielding)
- Roster / lineup data
- Standings, schedules, probable pitchers

### Quick-start code

```python
# pip install MLB-StatsAPI
import statsapi

# Get full 2025 schedule
schedule = statsapi.schedule(start_date="2025-03-27", end_date="2025-09-28")
for game in schedule[:3]:
    print(f"{game['away_name']} @ {game['home_name']}  "
          f"{game['away_score']}-{game['home_score']}  "
          f"({game['status']})")

# Probable pitchers for today
today = statsapi.schedule(date="2026-03-04")
for g in today:
    print(g.get("away_probable_pitcher", "TBD"),
          "vs",
          g.get("home_probable_pitcher", "TBD"))
```

```python
# Get a player's season stats
player = statsapi.lookup_player("Shohei Ohtani")[0]
stats = statsapi.player_stat_data(player["id"], group="hitting", type="season")
print(stats["stats"][0]["stats"])  # dict with AVG, HR, OPS, etc.
```

---

## 2. Baseball Savant / Statcast (free)

<https://baseballsavant.mlb.com>

Statcast is the gold standard for advanced metrics: exit velocity, launch angle, spin rate, sprint speed, expected stats (xBA, xSLG, xwOBA).

### Access methods
| Method | URL / Tool | Notes |
|--------|-----------|-------|
| CSV export | `https://baseballsavant.mlb.com/statcast_search` | Manual or scripted via query params |
| `pybaseball` | `pip install pybaseball` | Wraps Savant + FanGraphs + more |

### pybaseball examples

```python
# pip install pybaseball
from pybaseball import (
    statcast,
    statcast_pitcher,
    statcast_batter,
    batting_stats,
    pitching_stats,
    team_batting,
    team_pitching,
    schedule_and_record,
)

# ------- Statcast pitch-level data (date range) -------
# WARNING: large date ranges return millions of rows; chunk by month
data = statcast(start_dt="2025-03-27", end_dt="2025-04-30")
print(data.shape)           # e.g. (180000, 92)
print(data.columns.tolist())  # launch_speed, launch_angle, etc.

# ------- Season-level batting leaders -------
batters_2025 = batting_stats(2025, qual=100)  # min 100 PA
print(batters_2025[["Name", "Team", "AVG", "OBP", "SLG", "wOBA", "WAR"]].head(10))

# ------- Season-level pitching leaders -------
pitchers_2025 = pitching_stats(2025, qual=50)  # min 50 IP
print(pitchers_2025[["Name", "Team", "ERA", "FIP", "xFIP", "K/9", "WAR"]].head(10))

# ------- Team-level stats (great for model features) -------
team_bat = team_batting(2025)
team_pit = team_pitching(2025)
```

---

## 3. Retrosheet (free, historical)

<https://retrosheet.org/downloads/csvdownloads.html>

Play-by-play event files going back to 1871. Essential for deep historical analysis.

### What you get
- Game logs (every game ever played)
- Event files (every play, pitch by pitch for modern era)
- Roster files, park data, umpire assignments

### Download & parse

```python
import pandas as pd
import requests, zipfile, io

# Download game logs for a specific year
def download_retrosheet_gamelogs(year: int) -> pd.DataFrame:
    url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    filename = z.namelist()[0]
    
    # Retrosheet game log columns (161 columns)
    # Full column spec: https://www.retrosheet.org/gamelogs/glfields.txt
    col_names = [
        "date", "game_num", "day_of_week", "visiting_team", "visiting_league",
        "visiting_game_num", "home_team", "home_league", "home_game_num",
        "visiting_score", "home_score", "num_outs", "day_night", "completion_info",
        "forfeit_info", "protest_info", "park_id", "attendance", "time_of_game",
        "visiting_line_score", "home_line_score",
        # ... many more; load all as generic cols
    ]
    
    df = pd.read_csv(
        z.open(filename),
        header=None,
        low_memory=False,
    )
    # Rename the first few important columns
    important_cols = {
        0: "date", 3: "away_team", 6: "home_team",
        9: "away_score", 10: "home_score", 12: "day_night",
        16: "park_id", 17: "attendance",
    }
    df.rename(columns=important_cols, inplace=True)
    return df

# Pull last 5 years
for year in range(2021, 2026):
    try:
        gl = download_retrosheet_gamelogs(year)
        print(f"{year}: {len(gl)} games loaded")
    except Exception as e:
        print(f"{year}: {e}")
```

---

## 4. Odds & Lines Data

### 4a. The Odds API (freemium)
<https://the-odds-api.com/>

Real-time & historical odds from 30+ sportsbooks. Free tier: 500 requests/month.

```python
import requests

API_KEY = "YOUR_KEY"  # sign up at the-odds-api.com

def get_mlb_odds(api_key: str, markets: str = "h2h,spreads,totals") -> list:
    """Fetch current MLB odds from multiple books."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,          # h2h=moneyline, spreads=runline, totals=O/U
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    print(f"Remaining requests: {resp.headers.get('x-requests-remaining')}")
    return resp.json()

games = get_mlb_odds(API_KEY)
for g in games[:2]:
    print(f"\n{g['away_team']} @ {g['home_team']}")
    for book in g["bookmakers"]:
        for market in book["markets"]:
            print(f"  {book['key']} – {market['key']}: {market['outcomes']}")
```

### 4b. Playbook API
<https://www.playbook-api.com/>

From your README – provides structured betting data. Check their docs for MLB coverage and pricing.

---

## 5. Optional Scraping Targets (unofficial)

Many baseball data sites lack formal APIs but are rich enough to
warrant scraping. Below are a few commonly used sources:

### Baseball Reference 🟩
- **Massive historical database**: player stats, team stats, splits, game
  logs going back decades.
- **No official API**, so most users scrape HTML pages directly.
- Widely used for fantasy and analytics projects; the site structure is
  fairly consistent, making BeautifulSoup or `pandas.read_html` viable.

### FanGraphs 🟩
- Deep sabermetrics: WAR, wRC+, pitch values, projections (ZiPS,
  Steamer).
- Also **no easy API**; scraping is common and FanGraphs occasionally
  throttles scrapers. Some community wrappers (e.g. `fg-scrape`) exist.

### MLB Stats API (unofficial/undocumented) 🟥
- A structured JSON API used internally by MLB.com and the various
  league sites. The official `statsapi.mlb.com` wrapper above is just a
  thin client around it, but many people hit the API directly for
  endpoints not exposed by the wrappers.
- Provides schedules, standings, player stats, play‑by‑play, and more.
- GitHub projects like `mlb_scraper` wrap or mirror this API for easier
  use.

### Baseball Savant / Statcast 🟧
- The **gold standard for pitch‑level data**: exit velocity, spin rate,
  launch angle, pitch movement, etc.
- Scraping is common (csv exports via query params), but Python
  packages such as `pybaseball` or `baseball-scraper` provide programmatic
  access.

### MLB.com 🟨
- Source for game schedules, box scores, standings, and player pages.
- Often scraped with `BeautifulSoup` tutorials; the mobile site JSON
  endpoints are also a handy alternative.

> **Note:** these sources should be treated as secondary; prefer the
> official/structured APIs where available. Scraping may break if site
> layouts change and could violate terms of service, so cache results
> aggressively and respect robots.txt.


### 4c. mlb-odds-scraper
<https://github.com/ArnavSaraogi/mlb-odds-scraper>

Open-source scraper targeting specific sportsbook sites. Use as a reference or supplement.

### 4d. Action Network / Covers.com (scraping)
Historical opening & closing lines, public betting percentages. Scraping these is fragile; prefer APIs when possible.

---

## 5. FanGraphs (free)

<https://www.fangraphs.com/>

`pybaseball` wraps most FanGraphs data. Key extras beyond Savant:
- **Park Factors** – run-scoring environment per ballpark
- **Projection Systems** – ZiPS, Steamer, ATC
- **Pitch-level grades** (Stuff+, Location+, Pitching+)
- **WAR / wRC+ / FIP** – advanced rate stats

```python
from pybaseball import (
    batting_stats,
    pitching_stats,
    team_batting,
    fg_batting_data,
)

# FanGraphs leaderboard with advanced stats
leaders = batting_stats(2025, qual=200)
cols = ["Name", "Team", "PA", "wRC+", "Barrel%", "HardHit%", "WAR"]
print(leaders[cols].sort_values("wRC+", ascending=False).head(15))
```

---

## 6. Weather Data

Wind and temperature dramatically affect totals (especially at Wrigley, Coors, etc.).

| Source | URL | Free Tier |
|--------|-----|-----------|
| Open-Meteo | <https://open-meteo.com/> | Unlimited, no key needed |
| OpenWeatherMap | <https://openweathermap.org/api> | 1,000 calls/day free |
| Visual Crossing | <https://www.visualcrossing.com/> | 1,000 records/day free |

```python
import requests
from datetime import datetime

# Open-Meteo: free, no API key, historical + forecast
def get_weather_for_game(lat: float, lon: float, game_date: str, game_hour: int = 19):
    """Get weather for a specific ballpark at game time.
    
    Args:
        lat, lon: Ballpark coordinates
        game_date: 'YYYY-MM-DD'
        game_hour: Local hour (24h) of first pitch
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability",
        "start_date": game_date,
        "end_date": game_date,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params)
    data = resp.json()["hourly"]
    
    # Extract the hour closest to game time
    idx = game_hour  # index into hourly arrays
    return {
        "temp_f": data["temperature_2m"][idx],
        "wind_mph": data["windspeed_10m"][idx],
        "wind_dir": data["winddirection_10m"][idx],
        "precip_prob": data["precipitation_probability"][idx],
    }

# Example: Wrigley Field, 7 PM CT game
weather = get_weather_for_game(41.9484, -87.6553, "2026-03-04", game_hour=19)
print(weather)
```

### Ballpark Coordinates Reference

```python
BALLPARKS = {
    "Wrigley Field":           (41.9484, -87.6553),
    "Guaranteed Rate Field":   (41.8299, -87.6338),
    "Yankee Stadium":          (40.8296, -73.9262),
    "Citi Field":              (40.7571, -73.8458),
    "Fenway Park":             (42.3467, -71.0972),
    "Dodger Stadium":          (34.0739, -118.2400),
    "Oracle Park":             (37.7786, -122.3893),
    "Coors Field":             (39.7559, -104.9942),
    "Globe Life Field":        (32.7473, -97.0845),
    "Minute Maid Park":        (29.7572, -95.3555),
    "Truist Park":             (33.8908, -84.4678),
    "Nationals Park":          (38.8730, -77.0074),
    "Citizens Bank Park":      (39.9061, -75.1665),
    "Camden Yards":            (39.2838, -76.6216),
    "PNC Park":                (40.4469, -80.0058),
    "Great American Ball Park":(39.0975, -84.5069),
    "American Family Field":   (43.0280, -87.9712),
    "Target Field":            (44.9818, -93.2775),
    "Busch Stadium":           (38.6226, -90.1928),
    "Kauffman Stadium":        (39.0517, -94.4803),
    "T-Mobile Park":           (47.5914, -122.3325),
    "Oakland Coliseum":        (37.7516, -122.2005),
    "Angel Stadium":           (33.8003, -117.8827),
    "Petco Park":              (32.7076, -117.1570),
    "Chase Field":             (33.4455, -112.0667),
    "Rogers Centre":           (43.6414, -79.3894),
    "Tropicana Field":         (27.7682, -82.6534),
    "loanDepot Park":          (25.7781, -80.2197),
    "Comerica Park":           (42.3390, -83.0485),
    "Progressive Field":       (41.4962, -81.6852),
}
```

---

## 7. Additional / Optional Sources

| Source | What It Provides | URL |
|--------|-----------------|-----|
| **Baseball Reference** | Historical stats, game logs, splits | <https://www.baseball-reference.com/> |
| **Lahman Database** | Comprehensive historical database (1871+) | `pip install lahman` or <https://www.seanlahman.com/baseball-archive/statistics/> |
| **Bovada / DraftKings APIs** | Real-time lines (unofficial) | Scraping required |
| **ESPN API** | Scores, standings, news | `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard` |
| **Rotowire / RotoGrinders** | Lineups, DFS projections | <https://www.rotowire.com/baseball/daily-lineups.php> |
| **Umpire Scorecards** | Umpire zone accuracy & tendencies | <https://umpscorecards.com/> |
| **Brooks Baseball** | Pitch movement & velocity trends | <http://www.brooksbaseball.net/> |
| **Swish Analytics** | Predictive models, player props | <https://swishanalytics.com/> |

---

## Source Priority Matrix

For the MVP, focus on these (in order):

| Priority | Source | Purpose |
|----------|--------|---------|
| **P0** | MLB Stats API (`MLB-StatsAPI`) | Schedules, scores, rosters, probable pitchers |
| **P0** | pybaseball (Savant + FanGraphs) | Advanced batting/pitching stats, Statcast |
| **P0** | The Odds API | Live & historical odds for all 3 bet types |
| **P1** | Open-Meteo | Weather at game time |
| **P1** | Retrosheet | Deep historical game logs (5+ years) |
| **P2** | Umpire Scorecards | Umpire zone tendencies for totals model |
| **P2** | Rotowire | Confirmed lineups (pre-game) |
| **P3** | Lahman Database | Extended historical context |

---

## Rate Limits & Caching Strategy

```python
# Use a simple disk cache to avoid hammering APIs
# pip install requests-cache
import requests_cache

# Cache MLB Stats API responses for 6 hours
requests_cache.install_cache(
    "mlb_cache",
    backend="sqlite",
    expire_after=21600,  # 6 hours in seconds
    allowable_methods=["GET"],
)
```

> **Next:** [02-data-ingestion.md](02-data-ingestion.md) – Building the pipelines to pull and store this data.
