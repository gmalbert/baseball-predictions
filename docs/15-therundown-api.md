# 15 – The Rundown API Integration

This document covers the integration of The Rundown API as the primary odds data source.

---

## Overview

The Rundown API (`https://therundown.io/api/v2`) provides MLB odds from three free-tier sportsbooks. It serves as the primary odds source with automatic fallback to The Odds API on failure.

### Free Tier Limits
- **20,000 data points/day** (reference data like `/affiliates` and `/markets` is free)
- **3 sportsbooks**: DraftKings (ID 19), BetMGM (ID 22), FanDuel (ID 23)
- **Rate limit**: 1 request/second
- Pre-match odds only (5-minute delay)
- **Per-fetch cost**: ~135 data points (15 games × 3 markets × 3 books)

---

## Configuration

### Environment Variable

```bash
THERUNDOWN_API_KEY=your_api_key_here
```

### Config Class

Added to `src/ingestion/config.py`:

```python
therundown_api_key: str = field(default_factory=lambda: os.getenv("THERUNDOWN_API_KEY", ""))
```

The config also creates `data_files/raw/therundown/` directory on initialization.

---

## API Client

**File**: `src/ingestion/therundown.py`

### Key Functions

| Function | Purpose |
|----------|---------|
| `fetch_events_for_date()` | Low-level fetch from `/sports/3/events/{date}` |
| `fetch_current_odds()` | High-level function returning DataFrame matching `odds.py` schema |
| `get_consensus_line()` | Computes median/mean odds across bookmakers |

### Market IDs

| ID | Market | Description |
|----|--------|-------------|
| 1 | `h2h` | Moneyline |
| 2 | `spreads` | Run line |
| 3 | `totals` | Over/under |

### Data Optimization

- Uses `main_line=true` parameter to minimize data-point usage
- Falls back to unfiltered request if no markets returned (common for completed games)
- Filters out "off the board" sentinel value (`0.0001`)
- For spreads/totals, filters alternate lines keeping only the main line per game/participant

---

## Output Schema

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

### Consensus Output

`get_consensus_line()` adds:
- `median_price` — Median price across books
- `mean_price` — Mean price across books
- `median_point` — Median point value for spreads/totals
- `num_books` — Number of bookmakers with odds

---

## Integration Points

All three consumers use the same pattern: try TheRundown first, catch any exception, log a warning, and fall back to The Odds API.

### 1. Scheduler (`src/ingestion/scheduler.py`)

```python
def _fetch_odds_primary() -> pd.DataFrame:
    """Try TheRundown first, fall back to The Odds API."""
    try:
        return therundown.fetch_current_odds()
    except Exception as exc:
        logger.warning("TheRundown fetch failed (%s), falling back to Odds API", exc)
        return fetch_current_odds()
```

Used by:
- `morning_data_pull()` — 8 AM ET
- `midday_odds_pull()` — 11 AM ET

### 2. Daily Pipeline (`src/picks/daily_pipeline.py`)

- Imports `therundown` module
- Has its own `_fetch_odds_primary()` with identical fallback pattern
- Called from `run_daily_pipeline()`

### 3. Afternoon Refresh (`src/picks/afternoon_refresh.py`)

- Imports `therundown` module
- Has its own `_fetch_odds_primary()` with identical fallback pattern
- Called from `afternoon_picks_refresh()` for line movement detection

---

## Data Storage

- **Raw CSVs**: `data_files/raw/therundown/therundown_{YYYYMMDD_HHMM}.csv`
- **Consensus snapshots**: Saved as Parquet to `data_files/processed/consensus_{date}_{label}.parquet`

---

## Known Gaps

1. **CI/CD**: `.github/workflows/ingestion.yml` does not pass `THERUNDOWN_API_KEY` secret, so GitHub Actions always falls back to The Odds API
2. **Documentation**: Other docs (`01-data-sources.md`, `02-data-ingestion.md`, `architecture.md`) not yet updated to mention TheRundown
3. **Tests**: No test files exist for TheRundown integration

---

## API Response Headers

The client logs usage from response headers:
- `X-Datapoints-Used` — Data points consumed by request
- `X-Datapoints-Remaining` — Remaining daily quota
