# src/ingestion/odds.py
"""Fetch current and historical odds from The Odds API."""

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from src.contracts.domain import stable_id
from src.ingestion.base import RetrievedPayload
from src.ingestion.raw_store import RawStore
from src.normalization.odds import normalize_quote

from .config import config


def fetch_current_odds(
    markets: str = "h2h,spreads,totals",
    bookmakers: str = "draftkings,fanduel,betmgm,caesars,pointsbet",
    *,
    target_date: date | None = None,
    as_of: datetime | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch live MLB odds for today's games.

    Markets:
        h2h      = moneyline (underdog picks)
        spreads  = run line (+/- 1.5 typically)
        totals   = over/under

    Raises:
        ValueError: If ODDS_API_KEY environment variable is not set.
        requests.HTTPError: If the API request fails.
    """
    if not config.odds_api_key:
        raise ValueError("Set ODDS_API_KEY environment variable")
    requested = target_date or date.today()
    if requested != date.today():
        raise ValueError(
            "Live odds endpoint cannot replay historical target dates; use the archived quote repository"
        )
    observed_at = as_of or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": config.odds_api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": bookmakers,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Odds API requests remaining: {remaining}")

    run_id = run_id or stable_id("ingestion", "odds", observed_at.isoformat())
    raw_root = config.project_root / "data" / "bronze"
    observation = RawStore(raw_root).persist(
        RetrievedPayload(
            source="odds_api",
            body=resp.content,
            observed_at=observed_at,
            request_params={key: value for key, value in params.items() if key != "apiKey"},
            http_metadata={
                "status_code": resp.status_code,
                "requests_remaining": remaining,
                "content_type": resp.headers.get("content-type"),
            },
        ),
        ingestion_run_id=run_id,
    )

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
                    rows.append(
                        {
                            "game_id": game_id,
                            "commence_time": commence,
                            "away_team": away,
                            "home_team": home,
                            "bookmaker": book_name,
                            "market": market_key,
                            "outcome_name": outcome["name"],
                            "outcome_price": outcome["price"],
                            "outcome_point": outcome.get("point"),
                            "source_updated_at": market.get("last_update")
                            or book.get("last_update"),
                            "is_suspended": bool(market.get("is_suspended", False)),
                            "is_actionable": True,
                            "fetched_at": observed_at.isoformat(),
                            "raw_payload_hash": observation.payload_sha256,
                            "ingestion_run_id": run_id,
                        }
                    )

    df = pd.DataFrame(rows)

    # Save with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = config.raw_dir / "odds" / f"odds_{ts}.csv"
    df.to_csv(outpath, index=False)
    _archive_normalized_quotes(
        df, observed_at=observed_at, raw_payload_hash=observation.payload_sha256, run_id=run_id
    )
    print(f"  {len(df)} odds rows → {outpath}")

    return df


def _archive_normalized_quotes(
    frame: pd.DataFrame,
    *,
    observed_at: datetime,
    raw_payload_hash: str,
    run_id: str,
) -> Path | None:
    """Persist immutable selection-level quote ticks in partitioned Parquet."""
    if frame.empty:
        return None
    records = [
        normalize_quote(
            row,
            observed_at=observed_at,
            raw_payload_hash=raw_payload_hash,
            ingestion_run_id=run_id,
        ).model_dump(mode="json")
        for row in frame.to_dict("records")
    ]
    target = (
        config.project_root
        / "data"
        / "silver"
        / "odds_quote"
        / f"observed_date={observed_at.date().isoformat()}"
        / f"quotes_{observed_at.strftime('%H%M%S')}_{raw_payload_hash[:12]}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(target, index=False)
    return target


def get_consensus_line(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate decimal prices/implied probabilities, never American prices directly."""
    from src.markets.pricing import american_to_decimal, decimal_to_american

    if df.empty:
        return df.copy()
    work = df.copy()
    work["decimal_price"] = work["outcome_price"].map(lambda value: american_to_decimal(int(value)))
    work["implied_probability"] = 1.0 / work["decimal_price"]
    work["outcome_point"] = pd.to_numeric(work["outcome_point"], errors="coerce")
    point_sentinel = -9_999_999.0
    work["_point_for_aggregate"] = work["outcome_point"].fillna(point_sentinel)
    consensus = (
        work.groupby(["game_id", "away_team", "home_team", "market", "outcome_name"])
        .agg(
            median_decimal_price=("decimal_price", "median"),
            mean_decimal_price=("decimal_price", "mean"),
            median_implied_probability=("implied_probability", "median"),
            implied_probability_dispersion=("implied_probability", "std"),
            median_point=("_point_for_aggregate", "median"),
            num_books=("bookmaker", "nunique"),
            quote_time=("fetched_at", "max"),
        )
        .reset_index()
    )
    # Compatibility columns retain American display form but are derived after aggregation.
    consensus["median_price"] = consensus["median_decimal_price"].map(decimal_to_american)
    consensus["mean_price"] = consensus["mean_decimal_price"].map(decimal_to_american)
    consensus.loc[consensus["median_point"] == point_sentinel, "median_point"] = float("nan")
    return consensus


if __name__ == "__main__":
    odds_df = fetch_current_odds()
    consensus = get_consensus_line(odds_df)
    print(consensus.to_string())
