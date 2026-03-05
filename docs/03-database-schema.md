# 03 – Data Schema & Storage

Primary storage uses **CSV / Parquet** files in `data_files/`. PostgreSQL is documented as an optional upgrade path.

---

## Storage Strategy

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **CSV (raw)** | Ingestion landing zone | Human-readable, Git-friendly | Slow for large reads, no types |
| **Parquet (processed)** | Model training, dashboards | Columnar compression, fast reads, typed | Binary format, needs pyarrow |
| **PostgreSQL (optional)** | Multi-user, complex queries, scaling | SQL joins, concurrent access | Hosting cost, ops overhead |

**Recommended flow:** Ingest → CSV (raw) → Parquet (processed) → Models & Streamlit read Parquet directly.

---

## File Layout

```
data_files/
├── raw/                          # Landing zone (CSV)
│   ├── gamelogs/
│   │   ├── schedule_2021.csv
│   │   ├── schedule_2022.csv
│   │   ├── ...
│   │   ├── schedule_all.csv
│   │   └── retrosheet_all.csv
│   ├── batting/
│   │   ├── batting_2021.csv
│   │   ├── team_batting_2021.csv
│   │   └── ...
│   ├── pitching/
│   │   ├── pitching_2021.csv
│   │   ├── team_pitching_2021.csv
│   │   └── ...
│   ├── odds/
│   │   ├── odds_20260304_1100.csv
│   │   └── ...
│   └── weather/
│       └── weather_20260304.csv
│
├── processed/                    # Optimized for reads (Parquet)
│   ├── schedules.parquet
│   ├── gamelogs.parquet
│   ├── batting_stats.parquet
│   ├── pitching_stats.parquet
│   ├── team_batting.parquet
│   ├── team_pitching.parquet
│   ├── odds_history.parquet
│   ├── weather_history.parquet
│   ├── picks_history.parquet     # All daily picks + results
│   └── model_performance.parquet
│
└── models/                       # Trained model artifacts
    ├── underdog_xgb_v1.joblib
    ├── spread_xgb_v1.joblib
    └── totals_xgb_v1.joblib
```

---

## Parquet Schema Definitions

Define schemas with pandas + pyarrow to enforce types on write.

```python
# src/data/schemas.py
"""Parquet schema definitions for all data files."""

import pandas as pd
import pyarrow as pa
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data_files"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# TEAMS reference data (small CSV, loaded into memory)
# ============================================================
TEAMS_DATA = [
    {"abbrev": "ARI", "name": "Arizona Diamondbacks",     "league": "NL", "division": "West",    "ballpark": "Chase Field",              "is_dome": True},
    {"abbrev": "ATL", "name": "Atlanta Braves",           "league": "NL", "division": "East",    "ballpark": "Truist Park",              "is_dome": False},
    {"abbrev": "BAL", "name": "Baltimore Orioles",        "league": "AL", "division": "East",    "ballpark": "Camden Yards",             "is_dome": False},
    {"abbrev": "BOS", "name": "Boston Red Sox",           "league": "AL", "division": "East",    "ballpark": "Fenway Park",              "is_dome": False},
    {"abbrev": "CHC", "name": "Chicago Cubs",             "league": "NL", "division": "Central", "ballpark": "Wrigley Field",            "is_dome": False},
    {"abbrev": "CWS", "name": "Chicago White Sox",        "league": "AL", "division": "Central", "ballpark": "Guaranteed Rate Field",    "is_dome": False},
    {"abbrev": "CIN", "name": "Cincinnati Reds",          "league": "NL", "division": "Central", "ballpark": "Great American Ball Park", "is_dome": False},
    {"abbrev": "CLE", "name": "Cleveland Guardians",      "league": "AL", "division": "Central", "ballpark": "Progressive Field",        "is_dome": False},
    {"abbrev": "COL", "name": "Colorado Rockies",         "league": "NL", "division": "West",    "ballpark": "Coors Field",              "is_dome": False},
    {"abbrev": "DET", "name": "Detroit Tigers",           "league": "AL", "division": "Central", "ballpark": "Comerica Park",            "is_dome": False},
    {"abbrev": "HOU", "name": "Houston Astros",           "league": "AL", "division": "West",    "ballpark": "Minute Maid Park",         "is_dome": True},
    {"abbrev": "KC",  "name": "Kansas City Royals",       "league": "AL", "division": "Central", "ballpark": "Kauffman Stadium",         "is_dome": False},
    {"abbrev": "LAA", "name": "Los Angeles Angels",       "league": "AL", "division": "West",    "ballpark": "Angel Stadium",            "is_dome": False},
    {"abbrev": "LAD", "name": "Los Angeles Dodgers",      "league": "NL", "division": "West",    "ballpark": "Dodger Stadium",           "is_dome": False},
    {"abbrev": "MIA", "name": "Miami Marlins",            "league": "NL", "division": "East",    "ballpark": "loanDepot Park",           "is_dome": True},
    {"abbrev": "MIL", "name": "Milwaukee Brewers",        "league": "NL", "division": "Central", "ballpark": "American Family Field",    "is_dome": True},
    {"abbrev": "MIN", "name": "Minnesota Twins",          "league": "AL", "division": "Central", "ballpark": "Target Field",             "is_dome": False},
    {"abbrev": "NYM", "name": "New York Mets",            "league": "NL", "division": "East",    "ballpark": "Citi Field",               "is_dome": False},
    {"abbrev": "NYY", "name": "New York Yankees",         "league": "AL", "division": "East",    "ballpark": "Yankee Stadium",           "is_dome": False},
    {"abbrev": "OAK", "name": "Oakland Athletics",        "league": "AL", "division": "West",    "ballpark": "Oakland Coliseum",         "is_dome": False},
    {"abbrev": "PHI", "name": "Philadelphia Phillies",    "league": "NL", "division": "East",    "ballpark": "Citizens Bank Park",       "is_dome": False},
    {"abbrev": "PIT", "name": "Pittsburgh Pirates",       "league": "NL", "division": "Central", "ballpark": "PNC Park",                 "is_dome": False},
    {"abbrev": "SD",  "name": "San Diego Padres",         "league": "NL", "division": "West",    "ballpark": "Petco Park",               "is_dome": False},
    {"abbrev": "SF",  "name": "San Francisco Giants",     "league": "NL", "division": "West",    "ballpark": "Oracle Park",              "is_dome": False},
    {"abbrev": "SEA", "name": "Seattle Mariners",         "league": "AL", "division": "West",    "ballpark": "T-Mobile Park",            "is_dome": True},
    {"abbrev": "STL", "name": "St. Louis Cardinals",      "league": "NL", "division": "Central", "ballpark": "Busch Stadium",            "is_dome": False},
    {"abbrev": "TB",  "name": "Tampa Bay Rays",           "league": "AL", "division": "East",    "ballpark": "Tropicana Field",          "is_dome": True},
    {"abbrev": "TEX", "name": "Texas Rangers",            "league": "AL", "division": "West",    "ballpark": "Globe Life Field",         "is_dome": True},
    {"abbrev": "TOR", "name": "Toronto Blue Jays",        "league": "AL", "division": "East",    "ballpark": "Rogers Centre",            "is_dome": True},
    {"abbrev": "WSH", "name": "Washington Nationals",     "league": "NL", "division": "East",    "ballpark": "Nationals Park",           "is_dome": False},
]


def get_teams_df() -> pd.DataFrame:
    """Get the 30-team reference DataFrame."""
    return pd.DataFrame(TEAMS_DATA)


# ============================================================
# GAMES schema
# ============================================================
GAMES_SCHEMA = pa.schema([
    ("game_id",          pa.int32()),
    ("date",             pa.date32()),
    ("season",           pa.int16()),
    ("game_type",        pa.string()),     # R=regular, P=postseason
    ("away_team",        pa.string()),     # team abbreviation
    ("home_team",        pa.string()),
    ("venue",            pa.string()),
    ("away_score",       pa.int16()),
    ("home_score",       pa.int16()),
    ("total_runs",       pa.int16()),
    ("run_diff",         pa.int16()),      # home_score - away_score
    ("home_win",         pa.bool_()),
    ("status",           pa.string()),
    ("away_starter",     pa.string()),
    ("home_starter",     pa.string()),
    ("attendance",       pa.int32()),
])


# ============================================================
# BATTING STATS schema (per player per season)
# ============================================================
BATTING_SCHEMA = pa.schema([
    ("player_name",  pa.string()),
    ("player_id",    pa.int32()),
    ("season",       pa.int16()),
    ("team",         pa.string()),
    ("g",            pa.int16()),
    ("pa",           pa.int16()),
    ("ab",           pa.int16()),
    ("h",            pa.int16()),
    ("doubles",      pa.int16()),
    ("triples",      pa.int16()),
    ("hr",           pa.int16()),
    ("rbi",          pa.int16()),
    ("sb",           pa.int16()),
    ("bb",           pa.int16()),
    ("so",           pa.int16()),
    ("avg",          pa.float32()),
    ("obp",          pa.float32()),
    ("slg",          pa.float32()),
    ("ops",          pa.float32()),
    ("woba",         pa.float32()),
    ("wrc_plus",     pa.int16()),
    ("barrel_pct",   pa.float32()),
    ("hard_hit_pct", pa.float32()),
    ("exit_velo",    pa.float32()),
    ("launch_angle", pa.float32()),
    ("xba",          pa.float32()),
    ("xslg",         pa.float32()),
    ("war",          pa.float32()),
])


# ============================================================
# PITCHING STATS schema (per player per season)
# ============================================================
PITCHING_SCHEMA = pa.schema([
    ("player_name",  pa.string()),
    ("player_id",    pa.int32()),
    ("season",       pa.int16()),
    ("team",         pa.string()),
    ("g",            pa.int16()),
    ("gs",           pa.int16()),
    ("ip",           pa.float32()),
    ("w",            pa.int16()),
    ("l",            pa.int16()),
    ("sv",           pa.int16()),
    ("h",            pa.int16()),
    ("er",           pa.int16()),
    ("hr",           pa.int16()),
    ("bb",           pa.int16()),
    ("so",           pa.int16()),
    ("era",          pa.float32()),
    ("whip",         pa.float32()),
    ("fip",          pa.float32()),
    ("xfip",         pa.float32()),
    ("k_per_9",      pa.float32()),
    ("bb_per_9",     pa.float32()),
    ("hr_per_9",     pa.float32()),
    ("k_pct",        pa.float32()),
    ("bb_pct",       pa.float32()),
    ("avg_velo",     pa.float32()),
    ("spin_rate",    pa.float32()),
    ("barrel_pct",   pa.float32()),
    ("hard_hit_pct", pa.float32()),
    ("xera",         pa.float32()),
    ("war",          pa.float32()),
])


# ============================================================
# GAME ODDS schema
# ============================================================
ODDS_SCHEMA = pa.schema([
    ("game_id",       pa.string()),
    ("commence_time", pa.string()),
    ("away_team",     pa.string()),
    ("home_team",     pa.string()),
    ("bookmaker",     pa.string()),
    ("market",        pa.string()),     # h2h, spreads, totals
    ("outcome_name",  pa.string()),
    ("outcome_price", pa.int16()),      # American odds
    ("outcome_point", pa.float32()),    # spread/total value
    ("fetched_at",    pa.string()),
])


# ============================================================
# DAILY PICKS schema
# ============================================================
PICKS_SCHEMA = pa.schema([
    ("date",             pa.date32()),
    ("game_id",          pa.int32()),
    ("away_team",        pa.string()),
    ("home_team",        pa.string()),
    ("model_name",       pa.string()),
    ("pick_type",        pa.string()),     # underdog, spread, over_under
    ("pick_value",       pa.string()),     # "NYY +150", "Over 8.5"
    ("predicted_prob",   pa.float32()),
    ("confidence",       pa.string()),     # high, medium, low
    ("confidence_score", pa.float32()),
    ("edge",             pa.float32()),
    ("result",           pa.string()),     # win, loss, push, None
    ("actual_profit",    pa.float32()),    # in units
    ("settled_at",       pa.string()),
])


# ============================================================
# MODEL PERFORMANCE schema
# ============================================================
PERFORMANCE_SCHEMA = pa.schema([
    ("model_name",      pa.string()),
    ("pick_type",       pa.string()),
    ("period_start",    pa.date32()),
    ("period_end",      pa.date32()),
    ("total_picks",     pa.int32()),
    ("wins",            pa.int32()),
    ("losses",          pa.int32()),
    ("pushes",          pa.int32()),
    ("win_rate",        pa.float32()),
    ("roi",             pa.float32()),
    ("units_profit",    pa.float32()),
    ("avg_confidence",  pa.float32()),
    ("avg_edge",        pa.float32()),
    ("brier_score",     pa.float32()),
    ("log_loss",        pa.float32()),
])


# ============================================================
# Helper functions
# ============================================================

def load_parquet(name: str) -> pd.DataFrame:
    """Load a processed Parquet file by name.
    
    Usage:
        games = load_parquet("schedules")
        batting = load_parquet("batting_stats")
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, name: str, schema: pa.Schema = None):
    """Save a DataFrame as a Parquet file.
    
    Usage:
        save_parquet(picks_df, "picks_history", schema=PICKS_SCHEMA)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    if schema:
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, path)
    else:
        df.to_parquet(path, index=False, engine="pyarrow")
    print(f"Saved {len(df)} rows → {path}")


def append_parquet(df: pd.DataFrame, name: str):
    """Append rows to an existing Parquet file (read-merge-write).
    
    For daily pick appends, settled results, etc.
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(path, index=False, engine="pyarrow")
    print(f"Appended {len(df)} rows → {path} (total: {len(combined)})")
```

---

## Optional: PostgreSQL Schema

If you outgrow flat files or need multi-user concurrent access, here is a simplified SQL schema. Use `src/ingestion/loader.load_parquet_to_postgres()` to hydrate these tables from your Parquet files.

```sql
-- 000_create_schema.sql (optional — only if using PostgreSQL)

CREATE TABLE IF NOT EXISTS teams (
    id       SERIAL PRIMARY KEY,
    abbrev   VARCHAR(3) UNIQUE NOT NULL,
    name     VARCHAR(100) NOT NULL,
    league   VARCHAR(2) NOT NULL,
    division VARCHAR(10) NOT NULL,
    ballpark VARCHAR(100),
    is_dome  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS games (
    id           SERIAL PRIMARY KEY,
    mlb_game_id  INTEGER UNIQUE,
    date         DATE NOT NULL,
    season       SMALLINT NOT NULL,
    game_type    CHAR(1) DEFAULT 'R',
    away_team    VARCHAR(3),
    home_team    VARCHAR(3),
    venue        VARCHAR(100),
    away_score   SMALLINT,
    home_score   SMALLINT,
    total_runs   SMALLINT GENERATED ALWAYS AS (COALESCE(away_score,0) + COALESCE(home_score,0)) STORED,
    home_win     BOOLEAN GENERATED ALWAYS AS (home_score > away_score) STORED,
    status       VARCHAR(20) DEFAULT 'Scheduled'
);

CREATE TABLE IF NOT EXISTS daily_picks (
    id               SERIAL PRIMARY KEY,
    game_id          INTEGER,
    date             DATE NOT NULL,
    model_name       VARCHAR(50) NOT NULL,
    pick_type        VARCHAR(20) NOT NULL,
    pick_value       VARCHAR(50) NOT NULL,
    predicted_prob   DECIMAL(4, 3),
    confidence       VARCHAR(10) NOT NULL,
    confidence_score DECIMAL(4, 3),
    edge             DECIMAL(5, 3),
    result           VARCHAR(10),
    actual_profit    DECIMAL(6, 2),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_performance (
    id           SERIAL PRIMARY KEY,
    model_name   VARCHAR(50) NOT NULL,
    pick_type    VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    period_end   DATE NOT NULL,
    total_picks  INTEGER,
    wins         INTEGER,
    losses       INTEGER,
    win_rate     DECIMAL(4, 3),
    roi          DECIMAL(6, 3),
    units_profit DECIMAL(8, 2)
);
```

---

## Data Access Patterns

```python
# Examples of reading Parquet data throughout the app

import pandas as pd
from src.data.schemas import load_parquet

# Load all games
games = load_parquet("schedules")
games_2025 = games[games["season"] == 2025]

# Load batting stats and filter
batting = load_parquet("batting_stats")
qualified = batting[batting["pa"] >= 200]

# Load today's picks
picks = load_parquet("picks_history")
today = picks[picks["date"] == pd.Timestamp.today().date()]

# Join games + picks for Streamlit display
merged = today.merge(
    games[["game_id", "venue", "away_starter", "home_starter"]],
    on="game_id",
    how="left",
)
```

---

> **Next:** [04-betting-models.md](04-betting-models.md) – Building the prediction models that consume this data.
