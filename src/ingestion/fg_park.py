"""FanGraphs park factors split by batter handedness (L/R).

Handedness-split park factors are more accurate than scalar park factors
because the same stadium can play very differently for left-handed vs.
right-handed hitters (e.g., Fenway's Green Monster favours LHH doubles).

Usage:
    from src.ingestion.fg_park import load_fg_park_factors

    pf = load_fg_park_factors(2024)
    # Returns DataFrame with columns: team, hand (L/R), season,
    # and factor columns (basic, hr, 1b, 2b, 3b, runs …)
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import sleep

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_PROCESSED = Path(__file__).resolve().parents[2] / "data_files" / "processed"

# FanGraphs JSON API for park factors (no auth required)
_FG_PARK_URL = "https://www.fangraphs.com/api/stadium/parkfactors"

# Column renaming from FanGraphs JSON keys → project-standard names
_COL_RENAME = {
    "teamid": "fg_team_id",
    "teamabbrev": "team_abbrev",
    "teamname": "team",
    "hand": "hand",
    "season": "season",
    "basic": "pf_basic",
    "basicrun": "pf_runs",
    "hr": "pf_hr",
    "h": "pf_h",
    "1b": "pf_1b",
    "2b": "pf_2b",
    "3b": "pf_3b",
    "so": "pf_so",
    "ubb": "pf_bb",
}

# Neutral fallback (all factors = 100 = league average) — returned when fetch fails
def _neutral_fallback(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"team": t, "hand": h, "season": year, "pf_basic": 100}
         for t in _MLB_TEAM_NAMES for h in ("L", "R")]
    )

_MLB_TEAM_NAMES = [
    "Angels", "Astros", "Athletics", "Blue Jays", "Braves",
    "Brewers", "Cardinals", "Cubs", "Diamondbacks", "Dodgers",
    "Giants", "Guardians", "Mariners", "Marlins", "Mets",
    "Nationals", "Orioles", "Padres", "Phillies", "Pirates",
    "Rangers", "Rays", "Red Sox", "Reds", "Rockies",
    "Royals", "Tigers", "Twins", "White Sox", "Yankees",
]


def fetch_fg_park_factors(year: int, save: bool = True) -> pd.DataFrame:
    """Fetch FanGraphs park factors (L+R handedness) for one season.

    Calls the FanGraphs API once per handedness split (two requests total)
    and returns a combined DataFrame.

    Args:
        year: The MLB season.
        save: If True, caches result to ``data_files/processed/fg_park_{year}.parquet``.

    Returns:
        DataFrame with columns: team, hand, season, pf_basic (and more
        if the API returns them).  Returns a neutral-factor fallback if
        both API calls fail.
    """
    results: list[dict] = []
    for hand in ("L", "R"):
        params = {
            "startseason": year,
            "endseason": year,
            "leaguetype": "mlb",
            "hand": hand,
        }
        try:
            resp = requests.get(_FG_PARK_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for row in data:
                    row["hand"] = hand
                    row["season"] = year
                    results.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fg_park_factors fetch failed for %d/%s: %s", year, hand, exc)
        sleep(0.5)  # polite pause between requests

    if not results:
        logger.warning("fg_park_factors: all requests failed for %d; returning neutral fallback", year)
        return _neutral_fallback(year)

    df = pd.DataFrame(results)
    # Normalize column names: lowercase, strip whitespace
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})

    # Numeric factor columns
    factor_cols = [c for c in df.columns if c.startswith("pf_")]
    for col in factor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if save:
        _PROCESSED.mkdir(parents=True, exist_ok=True)
        outpath = _PROCESSED / f"fg_park_{year}.parquet"
        df.to_parquet(outpath, index=False)
        logger.info("Saved %s (%d rows)", outpath, len(df))

    return df.reset_index(drop=True)


def load_fg_park_factors(year: int) -> pd.DataFrame:
    """Load cached park factors for the given season, fetching if absent.

    Args:
        year: The MLB season.

    Returns:
        DataFrame with at least: team, hand (L/R), season, pf_basic (index 100).
    """
    path = _PROCESSED / f"fg_park_{year}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            pass
    return fetch_fg_park_factors(year, save=True)


def get_park_factor(
    team: str,
    hand: str,
    year: int,
    column: str = "pf_basic",
    pf_df: pd.DataFrame | None = None,
) -> float:
    """Return a single park factor value for a team / handedness / season.

    Args:
        team:   Team name (as it appears in the FanGraphs data).
        hand:   "L" or "R".
        year:   MLB season.
        column: Factor column to return (default: ``pf_basic``).
        pf_df:  Pre-loaded park factor DataFrame.  Auto-loaded if None.

    Returns:
        Park factor index (100 = neutral).  Returns 100 on lookup failure.
    """
    if pf_df is None:
        pf_df = load_fg_park_factors(year)
    mask = (pf_df.get("season") == year) & (pf_df.get("hand") == hand)
    # Fuzzy team name match — FanGraphs uses full city names
    if "team" in pf_df.columns:
        mask = mask & pf_df["team"].str.contains(team, case=False, na=False)
    elif "team_abbrev" in pf_df.columns:
        mask = mask & (pf_df["team_abbrev"] == team)
    rows = pf_df[mask]
    if rows.empty or column not in rows.columns:
        return 100.0
    val = rows[column].iloc[0]
    return float(val) if pd.notna(val) else 100.0
