# src/ingestion/odds.py
"""Fetch current and historical odds from The Odds API."""

import requests
import pandas as pd
from datetime import datetime

from .config import config


def fetch_current_odds(
    markets: str = "h2h,spreads,totals",
    bookmakers: str = "draftkings,fanduel,betmgm,caesars,pointsbet",
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
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "away_team": away,
                        "home_team": home,
                        "bookmaker": book_name,
                        "market": market_key,
                        "outcome_name": outcome["name"],
                        "outcome_price": outcome["price"],
                        "outcome_point": outcome.get("point"),
                        "fetched_at": datetime.utcnow().isoformat(),
                    })

    df = pd.DataFrame(rows)

    # Save with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = config.raw_dir / "odds" / f"odds_{ts}.csv"
    df.to_csv(outpath, index=False)
    print(f"  {len(df)} odds rows → {outpath}")

    return df


def get_consensus_line(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the consensus (median) line across bookmakers for each game/market."""
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
