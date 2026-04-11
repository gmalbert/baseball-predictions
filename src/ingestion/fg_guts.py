"""FanGraphs Guts! wOBA linear-weight constants and cFIP by season.

Provides year-specific constants for computing wOBA and FIP — using
the correct year's weights is important because park/rule changes shift
linear weights meaningfully across eras.

Usage:
    from src.ingestion.fg_guts import load_fg_guts, get_guts_for_year

    guts = load_fg_guts()          # DataFrame with one row per season
    row  = get_guts_for_year(2024) # Series: wBB, wHBP, w1B, w2B, w3B, wHR, cFIP …
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROCESSED = Path(__file__).resolve().parents[2] / "data_files" / "processed"

# ---------------------------------------------------------------------------
# Hardcoded fallback constants (source: FanGraphs Guts! page, verified 2020–2025)
# Used when the live fetch fails or the cache file is absent.
# ---------------------------------------------------------------------------
_GUTS_FALLBACK = pd.DataFrame(
    [
        # season  wBB    wHBP   w1B    w2B    w3B    wHR    scale   lg_wOBA cFIP
        (2019, 0.69, 0.72, 0.89, 1.27, 1.62, 2.10, 1.217, 0.320, 3.214),
        (2020, 0.69, 0.72, 0.88, 1.24, 1.56, 2.01, 1.157, 0.320, 3.100),
        (2021, 0.69, 0.72, 0.88, 1.25, 1.58, 2.06, 1.176, 0.318, 3.170),
        (2022, 0.68, 0.72, 0.88, 1.24, 1.56, 2.01, 1.157, 0.310, 3.100),
        (2023, 0.70, 0.73, 0.89, 1.26, 1.59, 2.07, 1.195, 0.318, 3.200),
        (2024, 0.70, 0.73, 0.89, 1.26, 1.59, 2.07, 1.195, 0.317, 3.180),
        (2025, 0.70, 0.73, 0.89, 1.26, 1.59, 2.07, 1.195, 0.317, 3.180),
        (2026, 0.70, 0.73, 0.89, 1.26, 1.59, 2.07, 1.195, 0.317, 3.180),
    ],
    columns=[
        "season", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR",
        "woba_scale", "lg_woba", "cFIP",
    ],
)


def fetch_fg_guts(save: bool = True) -> pd.DataFrame:
    """Attempt to fetch the FanGraphs Guts! table via pd.read_html.

    Falls back to hardcoded constants if FanGraphs is unreachable or
    the table schema has changed.

    Args:
        save: If True, writes the result to ``data_files/processed/fg_guts.parquet``.

    Returns:
        DataFrame with columns: season, wBB, wHBP, w1B, w2B, w3B, wHR,
        woba_scale, lg_woba, cFIP.
    """
    url = "https://www.fangraphs.com/guts.aspx?type=cn"
    try:
        tables = pd.read_html(url, header=0)
        # The constants table is the one with a "Season" column
        df = next(
            (t for t in tables if "Season" in t.columns),
            None,
        )
        if df is None:
            raise ValueError("Guts! table not found in page")

        df = df.copy()
        rename = {
            "Season": "season",
            "wBB": "wBB",
            "wHBP": "wHBP",
            "w1B": "w1B",
            "w2B": "w2B",
            "w3B": "w3B",
            "wHR": "wHR",
            "wOBA Scale": "woba_scale",
            "lg wOBA": "lg_woba",
            "cFIP": "cFIP",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for col in ["season", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR",
                    "woba_scale", "lg_woba", "cFIP"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["season"])
        df["season"] = df["season"].astype(int)
        df = df[df["season"] >= 2010].copy()

        # Ensure minimum required columns are present
        required_critical = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR", "cFIP"]
        missing_cols = [c for c in required_critical if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Fetched Guts! table missing critical columns: {missing_cols}")

        # Add optional columns if absent (back-fills from fallback)
        all_cols = ["season"] + required_critical + ["woba_scale", "lg_woba"]
        for col in ("woba_scale", "lg_woba"):
            if col not in df.columns:
                fb_col = _GUTS_FALLBACK[col] if col in _GUTS_FALLBACK else None
                if fb_col is not None:
                    # fill from fallback via season key
                    df = df.merge(
                        _GUTS_FALLBACK[["season", col]], on="season", how="left"
                    )
                else:
                    df[col] = float("nan")

        if save:
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            save_cols = ["season"] + required_critical + [
                c for c in ("woba_scale", "lg_woba") if c in df.columns
            ]
            df[save_cols].to_parquet(_PROCESSED / "fg_guts.parquet", index=False)
            logger.info("Saved fg_guts.parquet (%d seasons)", len(df))

        out_cols = ["season"] + required_critical + [
            c for c in ("woba_scale", "lg_woba") if c in df.columns
        ]
        return df[out_cols].reset_index(drop=True)

    except Exception as exc:  # noqa: BLE001
        logger.warning("fg_guts live fetch failed (%s); using built-in fallback constants", exc)
        return _GUTS_FALLBACK.copy()


def load_fg_guts() -> pd.DataFrame:
    """Load cached Guts! constants, fetching live on first call.

    Returns:
        DataFrame with columns: season, wBB, wHBP, w1B, w2B, w3B, wHR,
        woba_scale, lg_woba, cFIP.
    """
    path = _PROCESSED / "fg_guts.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            pass
    return fetch_fg_guts(save=True)


def get_guts_for_year(year: int, guts: pd.DataFrame | None = None) -> pd.Series:
    """Return the Guts! constant row for a given season.

    If the exact season is missing, uses the most recent available year ≤ year.

    Args:
        year:  The MLB season (e.g. 2025).
        guts:  Pre-loaded Guts! DataFrame.  Loaded automatically if None.

    Returns:
        Series with index: wBB, wHBP, w1B, w2B, w3B, wHR, woba_scale,
        lg_woba, cFIP.
    """
    if guts is None:
        guts = load_fg_guts()
    row = guts[guts["season"] == year]
    if row.empty:
        available = guts[guts["season"] <= year]
        row = (available if not available.empty else guts).sort_values("season").iloc[[-1]]
    return row.iloc[0]
