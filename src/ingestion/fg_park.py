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
_RETRO = Path(__file__).resolve().parents[2] / "data_files" / "retrosheet"

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
        [
            {"team": t, "hand": h, "season": year, "pf_basic": 100}
            for t in _MLB_TEAM_NAMES
            for h in ("L", "R")
        ]
    )


def _retrosheet_park_factors(year: int) -> pd.DataFrame | None:
    """Compute basic park factors from Retrosheet game-level run totals.

    Formula: PF = (runs/g at home) / (runs/g away) * 100, using the same
    home team across all its home and road games in that season.  Same value
    is applied to both handedness splits since Retrosheet lacks that detail.

    Returns None if Retrosheet data is unavailable for the requested year.
    """
    gi_path_pq = _RETRO / "gameinfo.parquet"
    gi_path_csv = _RETRO / "gameinfo.csv"
    try:
        if gi_path_pq.exists():
            gi = pd.read_parquet(
                gi_path_pq, columns=["visteam", "hometeam", "vruns", "hruns", "date"]
            )
        elif gi_path_csv.exists():
            gi = pd.read_csv(
                gi_path_csv,
                usecols=["visteam", "hometeam", "vruns", "hruns", "date"],
                dtype=str,
                low_memory=False,
            )
        else:
            return None
    except Exception:
        return None

    gi["season"] = pd.to_numeric(gi["date"].astype(str).str[:4], errors="coerce")
    gi = gi[gi["season"] == year].copy()
    if gi.empty:
        return None

    gi["vruns"] = pd.to_numeric(gi["vruns"], errors="coerce")
    gi["hruns"] = pd.to_numeric(gi["hruns"], errors="coerce")
    gi = gi.dropna(subset=["vruns", "hruns"])

    rows = []
    all_teams = set(gi["hometeam"]) | set(gi["visteam"])
    for team in all_teams:
        home = gi[gi["hometeam"] == team]
        away = gi[gi["visteam"] == team]
        if home.empty or away.empty:
            continue
        home_rpg = (home["hruns"].sum() + home["vruns"].sum()) / len(home)
        away_rpg = (away["vruns"].sum() + away["hruns"].sum()) / len(away)
        pf = round(home_rpg / away_rpg * 100, 1) if away_rpg > 0 else 100.0
        for hand in ("L", "R"):
            rows.append(
                {"team": team, "team_abbrev": team, "hand": hand, "season": year, "pf_basic": pf}
            )

    if not rows:
        return None
    return pd.DataFrame(rows)


_MLB_TEAM_NAMES = [
    "Angels",
    "Astros",
    "Athletics",
    "Blue Jays",
    "Braves",
    "Brewers",
    "Cardinals",
    "Cubs",
    "Diamondbacks",
    "Dodgers",
    "Giants",
    "Guardians",
    "Mariners",
    "Marlins",
    "Mets",
    "Nationals",
    "Orioles",
    "Padres",
    "Phillies",
    "Pirates",
    "Rangers",
    "Rays",
    "Red Sox",
    "Reds",
    "Rockies",
    "Royals",
    "Tigers",
    "Twins",
    "White Sox",
    "Yankees",
]


def _read_fg_guts_table(url: str, expected_columns: int) -> pd.DataFrame:
    """Read the substantive table from one FanGraphs Guts! page.

    The Guts pages are the public, browser-facing source for park factors.
    They are substantially more stable than FanGraphs' undocumented JSON API,
    which is protected by Cloudflare and regularly rejects GitHub-hosted
    runners.
    """
    tables = pd.read_html(url, header=0)
    table = next((t for t in tables if t.shape[1] >= expected_columns), None)
    if table is None:
        raise ValueError(f"Park-factor table with {expected_columns} columns not found")
    return table.iloc[:, :expected_columns].copy()


def _fetch_fg_guts_park_factors(year: int) -> pd.DataFrame:
    """Fetch park factors from FanGraphs' public Guts! tables.

    ``type=pf`` supplies the overall park factor and ``type=pfh`` supplies
    the handedness-specific hit and home-run factors.  The overall factor is
    applied to both handedness rows because FanGraphs does not publish a
    handedness-specific ``Basic`` value.
    """
    base_url = "https://www.fangraphs.com/guts.aspx?teamid=0&season="
    overall = _read_fg_guts_table(f"{base_url}{year}&type=pf", 16)
    handed = _read_fg_guts_table(f"{base_url}{year}&type=pfh", 10)

    overall.columns = [
        "season",
        "team",
        "pf_basic",
        "pf_3yr",
        "pf_1yr",
        "pf_1b",
        "pf_2b",
        "pf_3b",
        "pf_hr",
        "pf_so",
        "pf_bb",
        "pf_gb",
        "pf_fb",
        "pf_ld",
        "pf_iffb",
        "pf_fip",
    ]
    handed.columns = [
        "season",
        "team",
        "pf_1b_l",
        "pf_1b_r",
        "pf_2b_l",
        "pf_2b_r",
        "pf_3b_l",
        "pf_3b_r",
        "pf_hr_l",
        "pf_hr_r",
    ]

    for frame in (overall, handed):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame.dropna(subset=["season", "team"], inplace=True)
        frame["season"] = frame["season"].astype(int)

    overall = overall[overall["season"] == year]
    handed = handed[handed["season"] == year]
    merged = overall.merge(handed, on=["season", "team"], how="inner")
    if merged.empty:
        raise ValueError(f"No FanGraphs park factors returned for {year}")

    rows: list[dict] = []
    for hand in ("L", "R"):
        suffix = hand.lower()
        for _, row in merged.iterrows():
            rows.append(
                {
                    "team": row["team"],
                    "hand": hand,
                    "season": year,
                    "pf_basic": row["pf_basic"],
                    "pf_hr": row[f"pf_hr_{suffix}"],
                    "pf_1b": row[f"pf_1b_{suffix}"],
                    "pf_2b": row[f"pf_2b_{suffix}"],
                    "pf_3b": row[f"pf_3b_{suffix}"],
                }
            )

    df = pd.DataFrame(rows)
    for column in df.columns:
        if column.startswith("pf_"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.reset_index(drop=True)


def fetch_fg_park_factors(year: int, save: bool = True) -> pd.DataFrame:
    """Fetch FanGraphs park factors (L+R handedness) for one season.

    Reads FanGraphs' public Guts! park-factor tables and returns a combined
    handedness-split DataFrame. Falls back to the legacy API, then to
    Retrosheet-derived factors, if the public tables are unavailable.

    Args:
        year: The MLB season.
        save: If True, caches result to ``data_files/processed/fg_park_{year}.parquet``.

    Returns:
        DataFrame with columns: team, hand, season, pf_basic (and more
        if the source returns them). Returns a neutral-factor fallback only
        after all live and Retrosheet sources fail.
    """
    try:
        df = _fetch_fg_guts_park_factors(year)
        if save:
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            outpath = _PROCESSED / f"fg_park_{year}.parquet"
            df.to_parquet(outpath, index=False)
            logger.info("Saved %s (%d rows) from FanGraphs Guts!", outpath, len(df))
        return df
    except Exception as exc:
        logger.info("FanGraphs Guts! park-factor fetch failed for %d: %s", year, exc)

    # Legacy fallback: this undocumented API is occasionally still available,
    # but Cloudflare commonly blocks it on GitHub Actions runners.
    results: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.fangraphs.com/",
        "Accept": "application/json, text/plain, */*",
    }
    for hand in ("L", "R"):
        params = {
            "startseason": year,
            "endseason": year,
            "leaguetype": "mlb",
            "hand": hand,
        }
        try:
            resp = requests.get(_FG_PARK_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for row in data:
                    row["hand"] = hand
                    row["season"] = year
                    results.append(row)
        except Exception as exc:
            logger.debug("fg_park_factors fetch failed for %d/%s: %s", year, hand, exc)
        sleep(0.5)  # polite pause between requests

    if not results:
        retro_df = _retrosheet_park_factors(year)
        if retro_df is not None:
            logger.debug(
                "fg_park_factors: API unavailable for %d; using Retrosheet-computed fallback", year
            )
            if save:
                _PROCESSED.mkdir(parents=True, exist_ok=True)
                retro_df.to_parquet(_PROCESSED / f"fg_park_{year}.parquet", index=False)
            return retro_df
        logger.warning(
            "fg_park_factors: all sources failed for %d; returning neutral fallback", year
        )
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
        except Exception:
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
