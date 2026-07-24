# The Rundown API Integration Guide

This document contains all code needed to integrate The Rundown API as the primary odds source in any repo. Copy-paste these files directly, adjusting import paths as needed.

---

## Prerequisites

1. Sign up at <https://therundown.io/> and get an API key
2. Add `THERUNDOWN_API_KEY` to your `.env` file
3. Add `THERUNDOWN_API_KEY` to your CI/CD secrets (GitHub Actions, etc.)
4. Install dependencies: `pip install pandas requests python-dotenv`

---

## 1. Configuration (`src/ingestion/config.py`)

Add the `therundown_api_key` field and `therundown/` subdirectory to your existing config:

```python
# src/ingestion/config.py
from pathlib import Path
from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class IngestionConfig:
    """Central config for all ingestion jobs."""

    # Date range
    start_year: int = 2020
    end_year: int = 2026

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # API keys (from environment variables)
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    therundown_api_key: str = field(default_factory=lambda: os.getenv("THERUNDOWN_API_KEY", ""))

    # Rate limiting
    request_delay_sec: float = 1.0  # polite delay between API calls

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data_files" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data_files" / "processed"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for subdir in ["gamelogs", "batting", "pitching", "odds", "therundown", "weather"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


config = IngestionConfig()
```

**Key changes:**
- Added `therundown_api_key` field with `field(default_factory=...)` pattern
- Added `"therundown"` to the subdirectory list in `__post_init__`

---

## 2. API Client (`src/ingestion/therundown.py`)

Create this file with the full API client:

```python
# src/ingestion/therundown.py
"""Fetch MLB odds from TheRundown API (V2).

TheRundown free tier:
  - 20,000 data points/day
  - 3 sportsbooks: DraftKings (19), FanDuel (23), BetMGM (22)
  - Pre-match odds only, 5-minute delay
  - 1 req/sec rate limit

Reference data (/affiliates, /markets) is free and doesn't count against quota.

Data-point cost per fetch: ~15 games × 3 markets × 3 books = ~135 points.
"""

from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests

from .config import config

BASE_URL = "https://therundown.io/api/v2"
MLB_SPORT_ID = 3

# Free-tier affiliate IDs
AFFILIATE_IDS = {
    19: "draftkings",
    22: "betmgm",
    23: "fanduel",
}

# Market ID → our canonical market name
MARKET_MAP = {
    1: "h2h",       # moneyline
    2: "spreads",   # run line
    3: "totals",    # over/under
}

# Sentinel value meaning "off the board"
SENTINEL_OFF_BOARD = 0.0001


def _headers() -> dict:
    return {"X-TheRundown-Key": config.therundown_api_key}


def fetch_events_for_date(
    target_date: Optional[date] = None,
    affiliate_ids: Optional[list[int]] = None,
    market_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Fetch raw event data from TheRundown for a given date.

    Args:
        target_date:    Date to fetch (defaults to today).
        affiliate_ids:  Sportsbook IDs to include (defaults to free-tier books).
        market_ids:     Market type IDs to include (defaults to ML/spread/total).

    Returns:
        List of event dicts from the API response.
    """
    if not config.therundown_api_key:
        raise ValueError("Set THERUNDOWN_API_KEY environment variable")

    target_date = target_date or date.today()
    affiliate_ids = affiliate_ids or list(AFFILIATE_IDS.keys())
    market_ids = market_ids or list(MARKET_MAP.keys())

    date_str = target_date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/sports/{MLB_SPORT_ID}/events/{date_str}"

    # Try main_line=true first to minimize data-point usage.
    # Falls back to unfiltered if the API returns no markets (can happen
    # for completed games where main-line metadata has been cleared).
    for main_line_param in ("true", None):
        params = {
            "affiliate_ids": ",".join(str(a) for a in affiliate_ids),
            "market_ids": ",".join(str(m) for m in market_ids),
        }
        if main_line_param:
            params["main_line"] = main_line_param

        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])

        # Check if we got any markets back
        has_markets = any(ev.get("markets") for ev in events)
        if has_markets or main_line_param is None:
            break
        print("  main_line=true returned no markets, retrying without filter")

    # Log usage from response headers
    used = resp.headers.get("X-Datapoints-Used", "?")
    remaining = resp.headers.get("X-Datapoints-Remaining", "?")
    print(f"TheRundown data points used: {used}, remaining: {remaining}")

    return events


def fetch_current_odds(
    target_date: Optional[date] = None,
    affiliate_ids: Optional[list[int]] = None,
    market_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Fetch today's MLB odds and return a DataFrame matching odds.py format.

    Output columns: game_id, commence_time, away_team, home_team, bookmaker,
    market, outcome_name, outcome_price, outcome_point, fetched_at.

    For spreads and totals, alternate lines are filtered out — only the main
    (consensus) line is kept.  Main line is identified by ``is_main_line=True``
    on any price; if no prices carry that flag, the line with the most
    sportsbook coverage is selected.

    Raises:
        ValueError: If THERUNDOWN_API_KEY is not set.
        requests.HTTPError: If the API request fails.
    """
    events = fetch_events_for_date(target_date, affiliate_ids, market_ids)
    fetched_at = datetime.utcnow().isoformat()
    rows = []

    for event in events:
        event_id = event["event_id"]
        event_date = event["event_date"]
        teams = event.get("teams", [])
        away_team = next((t["name"] for t in teams if t.get("is_away")), "")
        home_team = next((t["name"] for t in teams if t.get("is_home")), "")

        for market in event.get("markets", []):
            market_id = market["market_id"]
            market_name = MARKET_MAP.get(market_id)
            if not market_name:
                continue

            for participant in market.get("participants", []):
                outcome_name = participant["name"]

                for line in participant.get("lines", []):
                    line_value = line.get("value", "")
                    prices = line.get("prices", {})

                    # Track whether any book marks this as the main line
                    is_main = any(
                        p.get("is_main_line") for p in prices.values()
                    )

                    for aff_id_str, price_obj in prices.items():
                        aff_id = int(aff_id_str)
                        bookmaker = AFFILIATE_IDS.get(aff_id)
                        if not bookmaker:
                            continue

                        price = price_obj["price"]
                        if price == SENTINEL_OFF_BOARD:
                            continue

                        outcome_point = None
                        if market_name in ("spreads", "totals") and line_value:
                            try:
                                outcome_point = float(line_value)
                            except (ValueError, TypeError):
                                pass

                        rows.append({
                            "game_id": event_id,
                            "commence_time": event_date,
                            "away_team": away_team,
                            "home_team": home_team,
                            "bookmaker": bookmaker,
                            "market": market_name,
                            "outcome_name": outcome_name,
                            "outcome_price": price,
                            "outcome_point": outcome_point,
                            "is_main_line": is_main,
                            "line_value": line_value,
                            "fetched_at": fetched_at,
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # For spreads/totals, keep only the main line per (game, participant).
    # Moneyline has only one line per participant so no filtering needed.
    alt_mask = df["market"].isin(["spreads", "totals"])
    if alt_mask.any():
        alts = df[alt_mask].copy()
        mains = df[~alt_mask].copy()

        # Prefer is_main_line=True; fall back to the line with most books
        alts["_line_group"] = (
            alts["game_id"] + "|" + alts["outcome_name"] + "|" + alts["line_value"].astype(str)
        )
        # Rank: main lines first, then by number of books
        book_counts = alts.groupby("_line_group")["bookmaker"].nunique()
        alts["_book_count"] = alts["_line_group"].map(book_counts)
        alts["_rank"] = (~alts["is_main_line"]).astype(int) * 1000 - alts["_book_count"]

        best_groups = (
            alts.groupby(["game_id", "outcome_name"])["_rank"]
            .idxmin()
        )
        best_line_groups = alts.loc[best_groups, "_line_group"].unique()
        alts_filtered = alts[alts["_line_group"].isin(best_line_groups)]

        df = pd.concat([mains, alts_filtered.drop(columns=["_line_group", "_book_count", "_rank"])], ignore_index=True)

    # Drop helper columns
    df = df.drop(columns=["is_main_line", "line_value"], errors="ignore")

    # Save with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = config.raw_dir / "therundown" / f"therundown_{ts}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} therundown odds rows → {outpath}")

    return df


def get_consensus_line(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the consensus (median) line across bookmakers for each game/market.

    Same logic as odds.get_consensus_line — produces the same output schema
    so downstream code works unchanged.
    """
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

## 3. Integration Pattern (Fallback to The Odds API)

Add this pattern to any file that fetches odds. It tries TheRundown first, falls back to The Odds API on failure.

### 3a. Scheduler (`src/ingestion/scheduler.py`)

Add import and helper function:

```python
from . import therundown
from .odds import fetch_current_odds  # existing fallback

def _fetch_odds_primary() -> pd.DataFrame:
    """Try TheRundown first, fall back to The Odds API."""
    try:
        return therundown.fetch_current_odds()
    except Exception as exc:
        logger.warning("TheRundown fetch failed (%s), falling back to Odds API", exc)
        return fetch_current_odds()
```

Then replace all calls to `fetch_current_odds()` with `_fetch_odds_primary()`.

### 3b. Daily Pipeline (`src/picks/daily_pipeline.py`)

Same pattern:

```python
from src.ingestion import therundown
from src.ingestion.odds import fetch_current_odds, get_consensus_line

def _fetch_odds_primary() -> pd.DataFrame:
    """Try TheRundown first, fall back to The Odds API."""
    try:
        return therundown.fetch_current_odds()
    except Exception as exc:
        logger.warning("TheRundown fetch failed (%s), falling back to Odds API", exc)
        return fetch_current_odds()
```

### 3c. Afternoon Refresh (`src/picks/afternoon_refresh.py`)

Same pattern:

```python
from src.ingestion import therundown
from src.ingestion.odds import fetch_current_odds, get_consensus_line

def _fetch_odds_primary() -> pd.DataFrame:
    """Try TheRundown first, fall back to The Odds API."""
    try:
        return therundown.fetch_current_odds()
    except Exception as exc:
        logger.warning("TheRundown fetch failed (%s), falling back to Odds API", exc)
        return fetch_current_odds()
```

---

## 4. Environment Variables (`.env`)

```bash
# TheRundown API (primary odds source)
THERUNDOWN_API_KEY=your_api_key_here

# The Odds API (fallback)
ODDS_API_KEY=your_existing_key
```

---

## 5. CI/CD Secrets (GitHub Actions)

Add to `.github/workflows/*.yml` in every step that fetches odds:

```yaml
env:
  ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
  THERUNDOWN_API_KEY: ${{ secrets.THERUNDOWN_API_KEY }}
```

Then add the secret in GitHub:
1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `THERUNDOWN_API_KEY`
4. Value: your API key

---

## 6. Tests (`tests/test_rundown.py`)

Create this test file with mocked API responses:

```python
"""Tests for src/ingestion/therundown.py."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.ingestion.therundown import (
    fetch_events_for_date,
    fetch_current_odds,
    get_consensus_line,
    SENTINEL_OFF_BOARD,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

SAMPLE_EVENT = {
    "event_id": "abc123",
    "event_date": "2026-07-10T23:10:00Z",
    "teams": [
        {"name": "New York Yankees", "is_away": True, "is_home": False},
        {"name": "Boston Red Sox", "is_away": False, "is_home": True},
    ],
    "markets": [
        {
            "market_id": 1,
            "participants": [
                {
                    "name": "New York Yankees",
                    "lines": [
                        {
                            "value": "",
                            "prices": {
                                "19": {"price": -150, "is_main_line": True},
                                "22": {"price": -145, "is_main_line": True},
                                "23": {"price": -155, "is_main_line": True},
                            },
                        }
                    ],
                },
                {
                    "name": "Boston Red Sox",
                    "lines": [
                        {
                            "value": "",
                            "prices": {
                                "19": {"price": 130, "is_main_line": True},
                                "22": {"price": 125, "is_main_line": True},
                                "23": {"price": 135, "is_main_line": True},
                            },
                        }
                    ],
                },
            ],
        },
        {
            "market_id": 2,
            "participants": [
                {
                    "name": "New York Yankees",
                    "lines": [
                        {
                            "value": "-1.5",
                            "prices": {
                                "19": {"price": 120, "is_main_line": True},
                                "22": {"price": 115, "is_main_line": True},
                            },
                        },
                        {
                            "value": "-2.5",
                            "prices": {
                                "19": {"price": 180, "is_main_line": False},
                            },
                        },
                    ],
                },
                {
                    "name": "Boston Red Sox",
                    "lines": [
                        {
                            "value": "1.5",
                            "prices": {
                                "19": {"price": -140, "is_main_line": True},
                                "22": {"price": -135, "is_main_line": True},
                            },
                        },
                        {
                            "value": "2.5",
                            "prices": {
                                "19": {"price": -180, "is_main_line": False},
                            },
                        },
                    ],
                },
            ],
        },
        {
            "market_id": 3,
            "participants": [
                {
                    "name": "Over",
                    "lines": [
                        {
                            "value": "8.5",
                            "prices": {
                                "19": {"price": -110, "is_main_line": True},
                            },
                        }
                    ],
                },
                {
                    "name": "Under",
                    "lines": [
                        {
                            "value": "8.5",
                            "prices": {
                                "19": {"price": -110, "is_main_line": True},
                            },
                        }
                    ],
                },
            ],
        },
    ],
}


def _mock_response(events, headers=None):
    """Create a mock requests.Response with JSON body."""
    resp = MagicMock()
    resp.json.return_value = {"events": events}
    resp.raise_for_status = MagicMock()
    resp.headers = headers or {}
    return resp


# ── Tests: fetch_events_for_date ─────────────────────────────────────────

@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_returns_events(mock_config, mock_get):
    mock_config.therundown_api_key = "test-key"
    mock_get.return_value = _mock_response([SAMPLE_EVENT])

    events = fetch_events_for_date(date(2026, 7, 10))
    assert len(events) == 1
    assert events[0]["event_id"] == "abc123"


@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_raises_without_api_key(mock_config, mock_get):
    mock_config.therundown_api_key = ""
    with pytest.raises(ValueError, match="THERUNDOWN_API_KEY"):
        fetch_events_for_date()


@patch("src.ingestion.therundown.requests.get")
@patch("src.ingestion.therundown.config")
def test_fetch_events_for_date_retries_without_main_line(mock_config, mock_get):
    """When main_line=true returns no markets, retries without filter."""
    mock_config.therundown_api_key = "test-key"
    empty_resp = _mock_response([{"event_id": "x", "markets": []}])
    full_resp = _mock_response([SAMPLE_EVENT])
    mock_get.side_effect = [empty_resp, full_resp]

    events = fetch_events_for_date(date(2026, 7, 10))
    assert len(events) == 1
    assert mock_get.call_count == 2


# ── Tests: fetch_current_odds ────────────────────────────────────────────

@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_schema(mock_config, mock_fetch):
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))

    expected_cols = {
        "game_id", "commence_time", "away_team", "home_team",
        "bookmaker", "market", "outcome_name", "outcome_price",
        "outcome_point", "fetched_at",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_filters_sentinel(mock_config, mock_fetch):
    """Off-the-board sentinel (0.0001) should be excluded."""
    event = {
        "event_id": "xyz",
        "event_date": "2026-07-10T23:10:00Z",
        "teams": [
            {"name": "Team A", "is_away": True},
            {"name": "Team B", "is_home": True},
        ],
        "markets": [
            {
                "market_id": 1,
                "participants": [
                    {
                        "name": "Team A",
                        "lines": [
                            {
                                "value": "",
                                "prices": {
                                    "19": {"price": SENTINEL_OFF_BOARD, "is_main_line": True},
                                    "22": {"price": -150, "is_main_line": True},
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [event]

    df = fetch_current_odds(date(2026, 7, 10))
    assert SENTINEL_OFF_BOARD not in df["outcome_price"].values
    assert -150 in df["outcome_price"].values


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_filters_alternate_lines(mock_config, mock_fetch):
    """Spreads/totals should keep only the main line, not alternates."""
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))

    spreads = df[df["market"] == "spreads"]
    # Yankees should only have -1.5, not -2.5
    yankees_spreads = spreads[spreads["outcome_name"] == "New York Yankees"]
    assert set(yankees_spreads["outcome_point"].unique()) == {-1.5}

    # Red Sox should only have +1.5, not +2.5
    sox_spreads = spreads[spreads["outcome_name"] == "Boston Red Sox"]
    assert set(sox_spreads["outcome_point"].unique()) == {1.5}


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_moneyline_no_filtering(mock_config, mock_fetch):
    """Moneyline should not be filtered (only one line per participant)."""
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = [SAMPLE_EVENT]

    df = fetch_current_odds(date(2026, 7, 10))
    h2h = df[df["market"] == "h2h"]
    assert len(h2h) == 6  # 2 teams × 3 books


@patch("src.ingestion.therundown.fetch_events_for_date")
@patch("src.ingestion.therundown.config")
def test_fetch_current_odds_empty_events(mock_config, mock_fetch):
    mock_config.raw_dir = MagicMock()
    mock_fetch.return_value = []

    df = fetch_current_odds(date(2026, 7, 10))
    assert df.empty


# ── Tests: get_consensus_line ────────────────────────────────────────────

def test_get_consensus_line_medians():
    df = pd.DataFrame([
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -150, "outcome_point": None, "bookmaker": "dk"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -140, "outcome_point": None, "bookmaker": "fd"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "h2h",
         "outcome_name": "A", "outcome_price": -160, "outcome_point": None, "bookmaker": "mgm"},
    ])

    consensus = get_consensus_line(df)
    row = consensus.iloc[0]
    assert row["median_price"] == -150.0
    assert row["num_books"] == 3
    assert row["mean_price"] == pytest.approx(-150.0)


def test_get_consensus_line_with_points():
    df = pd.DataFrame([
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "spreads",
         "outcome_name": "A", "outcome_price": -110, "outcome_point": -1.5, "bookmaker": "dk"},
        {"game_id": "g1", "away_team": "A", "home_team": "B", "market": "spreads",
         "outcome_name": "A", "outcome_price": -115, "outcome_point": -1.5, "bookmaker": "fd"},
    ])

    consensus = get_consensus_line(df)
    assert consensus.iloc[0]["median_point"] == -1.5


# ── Tests: fallback behavior ─────────────────────────────────────────────

@patch("src.ingestion.therundown.fetch_current_odds")
def test_fallback_on_rundown_failure(mock_rundown):
    """Simulate the fallback pattern used in scheduler/pipeline."""
    from src.ingestion.odds import fetch_current_odds as fallback_fetch

    mock_rundown.side_effect = Exception("API down")

    with patch("src.ingestion.odds.fetch_current_odds") as mock_fallback:
        mock_fallback.return_value = pd.DataFrame({"col": [1]})

        # Replicate _fetch_odds_primary logic
        try:
            result = mock_rundown()
        except Exception:
            result = mock_fallback()

        assert len(result) == 1
        mock_fallback.assert_called_once()
```

---

## 7. Output Schema

The `fetch_current_odds()` function returns a DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | TheRundown event_id |
| `commence_time` | str | Event date/time |
| `away_team` | str | Away team name |
| `home_team` | str | Home team name |
| `bookmaker` | str | `draftkings`, `fanduel`, or `betmgm` |
| `market` | str | `h2h`, `spreads`, or `totals` |
| `outcome_name` | str | Team name or "Over"/"Under" |
| `outcome_price` | float | American odds price |
| `outcome_point` | float/None | Spread/total point value (None for moneyline) |
| `fetched_at` | str | ISO timestamp |

The `get_consensus_line()` function adds:
- `median_price` — Median price across books
- `mean_price` — Mean price across books
- `median_point` — Median point value for spreads/totals
- `num_books` — Number of bookmakers with odds

---

## 8. Key Implementation Details

### Why TheRundown First?
- **Higher quota**: 20,000 data points/day vs The Odds API's 500 requests/month
- **Consistent schema**: Same output format as The Odds API
- **More granular control**: Can filter by `main_line=true` to reduce data usage

### Main Line Filtering
TheRundown returns alternate lines (e.g., -1.5, -2.5, -3.5 for spreads). The client filters these to keep only the main line:
1. Prefers lines where `is_main_line=True` in the API response
2. Falls back to the line with the most sportsbook coverage
3. Moneyline has only one line per participant, so no filtering needed

### Sentinel Value Filtering
The value `0.0001` means "off the board" — the market is temporarily unavailable. The client filters these out.

### Data-Point Cost
Each fetch costs ~135 data points (15 games × 3 markets × 3 books). The client logs usage from response headers:
- `X-Datapoints-Used` — Points consumed by this request
- `X-Datapoints-Remaining` — Points left in daily quota

---

## 9. Checklist for New Repos

- [ ] Add `THERUNDOWN_API_KEY` to `.env`
- [ ] Add `therundown_api_key` to `IngestionConfig` in `config.py`
- [ ] Add `"therundown"` to subdirectory list in `__post_init__`
- [ ] Create `src/ingestion/therundown.py` with full client code
- [ ] Add `_fetch_odds_primary()` pattern to all odds consumers
- [ ] Add `THERUNDOWN_API_KEY` secret to CI/CD
- [ ] Pass `THERUNDOWN_API_KEY` in workflow env blocks
- [ ] Create `tests/test_rundown.py`
- [ ] Run tests: `python -m pytest tests/test_rundown.py -v`
