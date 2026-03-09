# 12 – Yahoo Fantasy Sports API: Untapped Data for Betting Models

## Overview

The [Yahoo Fantasy Sports API](https://developer.yahoo.com/fantasysports/guide/) exposes a rich set of MLB player and team data through a RESTful interface. While our platform already pulls stats from MLB Stats API, Retrosheet, pybaseball/FanGraphs, the Odds API, and Open-Meteo, the Yahoo Fantasy API offers **crowd-sourced signals** that no other source provides — specifically, ownership rates, roster transaction velocity, and consensus projections derived from the behavior of millions of fantasy managers.

These signals are valuable as **sentiment / wisdom-of-crowds features** that complement our statistical models.

---

## What This API Provides (MLB-Relevant)

| Resource | Sub-Resources | Key Fields | Betting Relevance |
|----------|---------------|------------|-------------------|
| **Game** | metadata, leagues, players | `game_key`, `season`, `game_code=mlb` | Entry point; gives current MLB game_id |
| **Player** | metadata, stats, ownership, draft_analysis | Name, team, position, `status`, `percent_owned`, fantasy points, player_notes | **Ownership % = crowd signal** of perceived value |
| **Player Stats** | by season, week, date | All standard MLB stat categories (H, R, HR, RBI, SB, ERA, WHIP, K, W, SV…) | Duplicates what we have — but with fantasy scoring context |
| **Player Ownership** | (sub-resource of Player) | `percent_owned`, `percent_started`, `percent_owned_delta` | **Novel signal** — not available from any other source |
| **Roster** | players (by date) | `selected_position`, `is_starting`, `starting_status` | Confirms real-world lineup status via fantasy rosters |
| **Transactions** | add/drop/trade/waiver | `type`, `timestamp`, `player_key`, `source_type`, `destination_type`, `faab_bid` | **Transaction velocity = injury/breakout early signal** |
| **League Settings** | stat_categories, stat_modifiers, roster_positions | Scoring weights per stat category | Understand which stats the crowd values |

---

## What We Are NOT Getting Today

### 1. Player Ownership & Start Rates ⭐⭐⭐⭐⭐ (Highest Value)

**Not available from any of our 6 current data sources.**

Yahoo tracks what percentage of all fantasy leagues have a player rostered (`percent_owned`) and what percentage of rosters are starting them (`percent_started`). They also surface `percent_owned_delta` — the rate-of-change over the past week.

| Field | What It Tells Us | Model Use |
|-------|------------------|-----------|
| `percent_owned` | How valuable the crowd thinks a player is right now | Weight SP quality by crowd consensus |
| `percent_started` | Whether managers trust the player in a given role | Detect platoon situations, phantom injuries |
| `percent_owned_delta` | Rate of adds/drops over past week | **Early injury signal, breakout detection** |

**Why this matters for betting:** A starting pitcher whose ownership % is dropping sharply may have unreported injury concerns or poor recent peripherals that Vegas hasn't fully priced in. Conversely, a reliever whose ownership is surging may be taking over a closer role.

**Fetch frequency:** Daily (once at 8 AM ET alongside schedule pull).

### 2. Fantasy Transaction Velocity ⭐⭐⭐⭐

**Not available from current sources.**

The Transactions collection lets us monitor adds/drops/trades across public leagues. A spike in drops for a specific player is a strong leading indicator of:
- Injury announcements (fantasy managers react faster than odds markets)
- Role changes (starter → bullpen, lineup demotion)
- Upcoming IL stints

| Signal | Detection Method | Betting Edge |
|--------|-----------------|--------------|
| Mass drops of a SP | Count drops in 24h window | Fade team with injured/demoted SP |
| Mass adds of a RP | Count adds in 24h window | Closer role change → impacts totals |
| FAAB bid amounts | Median bid for a waiver claim | Quantifies crowd's perceived value |

**Fetch frequency:** Every 6 hours during season.

### 3. Player Starting Status (Real-Time Lineups) ⭐⭐⭐⭐

We get probable pitchers from the MLB Stats API at 8 AM, but the Yahoo Roster resource provides `starting_status` per player per date — including position players. This lets us detect:
- Late lineup scratches (player owned but not started by most managers)
- Platoon matchups (LHP/RHP splits hidden in fantasy start/sit decisions)
- Rest days for key hitters (if a top-3 hitter isn't started in most leagues, he's likely sitting)

**Fetch frequency:** Twice daily (11 AM + 4 PM ET, aligned with existing odds pulls).

### 4. Player Stat Categories with Fantasy Scoring Weights ⭐⭐⭐

The League Settings resource exposes which statistical categories are enabled and their point modifiers. While we already have raw stats, the **fantasy scoring weights** represent a crowd-consensus valuation of which stats matter most. This can serve as a sanity check on our feature importance rankings.

**Fetch frequency:** Once per season (static per league).

### 5. Draft Analysis (ADP / Average Auction Value) ⭐⭐

**`/game/mlb/players;sort=AR` (Actual Rank) or sort by ADP**

Pre-season Average Draft Position reflects the crowd's baseline expectations. Comparing a player's ADP-implied performance to actual mid-season stats reveals who is over/underperforming expectations — a useful context signal.

**Fetch frequency:** Once pre-season, then monthly as a reference.

---

## API Authentication

Yahoo Fantasy API uses **OAuth 2.0** (3-legged for private league data, 2-legged for public data). Player ownership percentages and public league transactions are accessible via public endpoints.

```python
# src/ingestion/yahoo_fantasy.py
"""Yahoo Fantasy Sports API client for MLB crowd-signal features.

Fetches player ownership rates, transaction velocity, and lineup data
from the Yahoo Fantasy API. These provide wisdom-of-crowds signals
not available from traditional stats sources.

Setup:
    1. Register app at https://developer.yahoo.com/apps/create/
    2. Select "Fantasy Sports" with Read access
    3. Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET env vars
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests_oauthlib import OAuth2Session

from .config import config

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"  # noqa: S105

# MLB game codes — update each season
# Use `mlb` as game_code for current season
MLB_GAME_CODE = "mlb"


class YahooFantasyClient:
    """Authenticated client for the Yahoo Fantasy Sports API."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("YAHOO_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get(
            "YAHOO_CLIENT_SECRET", ""
        )
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET env vars"
            )
        self._session: Optional[OAuth2Session] = None
        self._token: Optional[dict] = None

    def _get_session(self) -> OAuth2Session:
        """Create or refresh an OAuth2 session (client credentials flow)."""
        if self._session and self._token:
            return self._session

        session = OAuth2Session(self.client_id)
        token = session.fetch_token(
            TOKEN_URL,
            client_id=self.client_id,
            client_secret=self.client_secret,
            grant_type="client_credentials",
        )
        self._session = session
        self._token = token
        return session

    def _get(self, uri: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated GET request, returning parsed JSON."""
        session = self._get_session()
        url = f"{BASE_URL}/{uri}"
        headers = {"Accept": "application/json"}
        resp = session.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
```

---

## Code Samples

### 1. Fetch Player Ownership Percentages

```python
def fetch_player_ownership(
    self,
    position: str = "SP",
    count: int = 50,
) -> pd.DataFrame:
    """Fetch ownership % for top players at a position.

    Args:
        position: Filter by position (SP, RP, C, 1B, OF, etc.)
        count: Number of players to return.

    Returns:
        DataFrame with player_key, name, team, percent_owned,
        percent_started, ownership_delta.
    """
    uri = (
        f"game/{MLB_GAME_CODE}/players;"
        f"position={position};sort=OR;count={count}"
        f"/ownership"
    )
    data = self._get(uri)

    rows = []
    players = (
        data.get("fantasy_content", {})
        .get("game", [{}])[1]
        .get("players", {})
    )

    # Yahoo nests players as numbered keys
    count_val = players.get("count", 0)
    for i in range(count_val):
        player_data = players.get(str(i), {}).get("player", [])
        if len(player_data) < 2:
            continue

        info = _parse_player_info(player_data[0])
        ownership = player_data[1].get("ownership", {})

        rows.append({
            "player_key": info.get("player_key"),
            "name": info.get("name"),
            "team": info.get("editorial_team_abbr"),
            "position": info.get("display_position"),
            "percent_owned": float(
                ownership.get("percent_owned", {}).get("value", 0)
            ),
            "percent_started": float(
                ownership.get("percent_started", {}).get("value", 0)
            ),
            "ownership_delta": float(
                ownership.get("percent_owned", {}).get("delta", 0)
            ),
            "fetched_at": datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(rows)
    logger.info("Fetched ownership for %d %s players", len(df), position)
    return df


def fetch_all_mlb_ownership(self) -> pd.DataFrame:
    """Fetch ownership for all MLB-relevant positions.

    Combines SP, RP, C, 1B, 2B, 3B, SS, OF into one DataFrame.
    """
    positions = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "OF"]
    frames = []
    for pos in positions:
        df = self.fetch_player_ownership(position=pos, count=50)
        frames.append(df)
        time.sleep(config.request_delay_sec)  # Rate-limit

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_key"])

    # Save raw
    outpath = (
        config.raw_dir / "yahoo" / f"ownership_{date.today().isoformat()}.csv"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outpath, index=False)
    logger.info("Ownership data saved → %s", outpath)

    return combined
```

### 2. Detect Transaction Velocity (Add/Drop Spikes)

```python
def fetch_recent_transactions(
    self,
    transaction_types: str = "add,drop",
    count: int = 100,
) -> pd.DataFrame:
    """Fetch recent public-league transactions to detect add/drop spikes.

    Args:
        transaction_types: Comma-separated types (add, drop, trade).
        count: Max transactions to return.

    Returns:
        DataFrame with player_key, name, type (add/drop),
        timestamp, source_type, destination_type.
    """
    # Use a well-known public league or aggregate across public leagues
    uri = (
        f"game/{MLB_GAME_CODE}/transactions;"
        f"types={transaction_types};count={count}"
    )
    data = self._get(uri)

    rows = []
    transactions = (
        data.get("fantasy_content", {})
        .get("game", [{}])[1]
        .get("transactions", {})
    )

    for i in range(transactions.get("count", 0)):
        txn = transactions.get(str(i), {}).get("transaction", [])
        if len(txn) < 2:
            continue

        meta = txn[0]
        players = txn[1].get("players", {})

        for j in range(players.get("count", 0)):
            player_data = players.get(str(j), {}).get("player", [])
            if len(player_data) < 2:
                continue

            info = _parse_player_info(player_data[0])
            txn_data = player_data[1].get("transaction_data", [{}])
            if isinstance(txn_data, list):
                txn_data = txn_data[0] if txn_data else {}

            rows.append({
                "player_key": info.get("player_key"),
                "name": info.get("name"),
                "team": info.get("editorial_team_abbr"),
                "transaction_type": txn_data.get("type"),
                "source_type": txn_data.get("source_type"),
                "destination_type": txn_data.get("destination_type"),
                "timestamp": meta.get("timestamp"),
                "fetched_at": datetime.utcnow().isoformat(),
            })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d recent transactions", len(df))
    return df


def compute_transaction_velocity(
    self,
    hours: int = 24,
) -> pd.DataFrame:
    """Aggregate add/drop counts per player over recent window.

    Returns:
        DataFrame with columns: player_key, name, team,
        adds_24h, drops_24h, net_adds_24h.
    """
    txns = self.fetch_recent_transactions(count=200)
    if txns.empty:
        return pd.DataFrame()

    cutoff = (
        datetime.utcnow().timestamp() - (hours * 3600)
    )
    txns["ts"] = txns["timestamp"].astype(float)
    recent = txns[txns["ts"] >= cutoff].copy()

    adds = (
        recent[recent["transaction_type"] == "add"]
        .groupby(["player_key", "name", "team"])
        .size()
        .reset_index(name="adds_24h")
    )
    drops = (
        recent[recent["transaction_type"] == "drop"]
        .groupby(["player_key", "name", "team"])
        .size()
        .reset_index(name="drops_24h")
    )

    velocity = adds.merge(drops, on=["player_key", "name", "team"], how="outer")
    velocity = velocity.fillna(0)
    velocity["net_adds_24h"] = velocity["adds_24h"] - velocity["drops_24h"]
    velocity = velocity.sort_values("net_adds_24h", ascending=True)

    return velocity
```

### 3. Detect Lineup Scratches via Fantasy Start Rates

```python
def detect_lineup_scratches(
    self,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Compare expected starters (by ownership) to actual start rates.

    A player with >80% ownership but <30% start rate on a given day
    is likely scratched or resting — a signal for our models.

    Args:
        schedule: Today's game schedule with team columns.

    Returns:
        DataFrame of suspected scratches with columns:
        player_key, name, team, percent_owned, percent_started,
        scratch_signal (bool).
    """
    ownership = self.fetch_all_mlb_ownership()

    scratches = ownership[
        (ownership["percent_owned"] > 80)
        & (ownership["percent_started"] < 30)
    ].copy()

    scratches["scratch_signal"] = True
    logger.info(
        "Detected %d suspected lineup scratches", len(scratches)
    )
    return scratches
```

### 4. Helper: Parse Yahoo Player Info

```python
def _parse_player_info(raw: list) -> dict:
    """Extract structured player info from Yahoo's nested response format.

    Yahoo returns player metadata as a list of dicts with varying keys.
    This normalizes them into a flat dict.
    """
    info: dict = {}
    for item in raw:
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "name":
                    info["name"] = value.get("full", "")
                elif isinstance(value, (str, int, float)):
                    info[key] = value
    return info
```

---

## Integration into Feature Pipeline

```python
# In src/models/features.py — proposed additions

# Yahoo Fantasy crowd-signal features
_YAHOO_FEATURES = [
    "home_sp_pct_owned",       # SP ownership %
    "away_sp_pct_owned",
    "home_sp_ownership_delta", # Ownership trend (positive = rising)
    "away_sp_ownership_delta",
    "home_sp_pct_started",     # Start rate (low = potential scratch)
    "away_sp_pct_started",
    "home_lineup_scratches",   # Count of key hitters suspected scratched
    "away_lineup_scratches",
    "home_sp_drops_24h",       # Transaction velocity for starting pitcher
    "away_sp_drops_24h",
]
```

### How These Features Improve Each Model

| Model | Feature | Signal |
|-------|---------|--------|
| **Underdog ML** | `sp_ownership_delta` | SP with dropping ownership → team may be undervalued by public but model can fade |
| **Underdog ML** | `lineup_scratches` | Key hitter resting → line may not adjust fast enough |
| **Spread** | `sp_pct_started` | Low start rate for a high-ownership SP → potential demotion or injury |
| **Totals** | `sp_drops_24h` | Mass drops of a SP → expect higher scoring game (bullpen game) |
| **Totals** | `home_lineup_scratches` + `away_lineup_scratches` | Multiple scratches on both sides → lean Under |

---

## Proposed Scheduler Integration

```python
# Addition to src/ingestion/scheduler.py

from src.ingestion.yahoo_fantasy import YahooFantasyClient

@scheduler.scheduled_job(CronTrigger(hour=8, minute=30, timezone="America/New_York"))
def yahoo_ownership_pull() -> None:
    """8:30 AM ET – Fetch Yahoo Fantasy ownership and transaction signals."""
    if not in_season():
        return

    logger.info("Running Yahoo Fantasy ownership pull...")
    client = YahooFantasyClient()

    # Ownership percentages
    ownership = client.fetch_all_mlb_ownership()
    logger.info("Fetched ownership for %d players", len(ownership))

    # Transaction velocity (last 24h)
    velocity = client.compute_transaction_velocity(hours=24)
    if not velocity.empty:
        outpath = config.raw_dir / "yahoo" / f"velocity_{date.today().isoformat()}.csv"
        velocity.to_csv(outpath, index=False)
        logger.info("Transaction velocity saved → %s", outpath)


@scheduler.scheduled_job(CronTrigger(hour=14, minute=0, timezone="America/New_York"))
def yahoo_lineup_check() -> None:
    """2 PM ET – Check for lineup scratches via fantasy start rates."""
    if not in_season():
        return

    logger.info("Running Yahoo Fantasy lineup scratch detection...")
    from src.ingestion.mlb_stats import fetch_todays_probable_pitchers

    client = YahooFantasyClient()
    schedule = fetch_todays_probable_pitchers()
    scratches = client.detect_lineup_scratches(schedule)

    if not scratches.empty:
        outpath = config.raw_dir / "yahoo" / f"scratches_{date.today().isoformat()}.csv"
        scratches.to_csv(outpath, index=False)
        logger.info("Scratches saved → %s", outpath)
```

---

## Recommended Fetch Frequency Summary

| Data | Endpoint | Frequency | Schedule |
|------|----------|-----------|----------|
| Player Ownership % | `/game/mlb/players;position={pos}/ownership` | Daily | 8:30 AM ET |
| Transaction Velocity | `/game/mlb/transactions;types=add,drop` | Every 6 hours | 8:30 AM, 2 PM, 8 PM ET |
| Lineup Scratches | Derived from ownership + start rates | Twice daily | 11 AM, 2 PM ET |
| Draft ADP / Rankings | `/game/mlb/players;sort=AR` | Monthly | 1st of month |
| League Scoring Weights | `/league/{key}/settings` | Once per season | Pre-season |

---

## Rate Limits & Costs

- **Rate limit:** ~2,000 requests/hour (shared across all Yahoo APIs)
- **Cost:** Free (requires Yahoo Developer app registration)
- **Auth:** OAuth 2.0 (client credentials for public data; 3-legged for private leagues)
- **Format:** XML by default; add `?format=json` or `Accept: application/json` header

---

## Implementation Priority

1. **Player Ownership % (daily)** — Highest standalone value; one new API call per position per day
2. **Transaction Velocity (6-hourly)** — Early injury/role-change detection
3. **Lineup Scratch Detection (2x daily)** — Combines ownership with start rates
4. **Draft ADP (monthly)** — Baseline expectation feature for over/underperformance
5. **League Scoring Weights (seasonal)** — Low priority; useful for feature importance validation

---

> **Next:** Implement `src/ingestion/yahoo_fantasy.py` with the `YahooFantasyClient` class and wire it into the scheduler.
