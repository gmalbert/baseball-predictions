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
