"""Chadwick Bureau player ID cross-reference registry.

Provides a mapping between player ID systems used by this project:

    key_retro     — Retrosheet player ID (e.g., "aaroh101")
    key_mlbam     — MLB Advanced Media (Statcast) player_id
    key_fangraphs — FanGraphs player_id

Usage:
    from src.ingestion.chadwick import load_player_registry, retro_to_mlbam

    registry = load_player_registry()   # full DataFrame
    mlbam_id = retro_to_mlbam("kersc001", registry=registry)  # -> 477132
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CHADWICK_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
)

_PROCESSED = Path(__file__).resolve().parents[2] / "data_files" / "processed"
_REGISTRY_PATH = _PROCESSED / "player_registry.parquet"

# Columns we need; Chadwick has ~100+ columns — keep only the useful ones
_KEEP_COLS = [
    "key_uuid",
    "key_mlbam",
    "key_retro",
    "key_fangraphs",
    "name_first",
    "name_last",
    "name_given",
    "birth_year",
]


def load_player_registry(force_refresh: bool = False) -> pd.DataFrame:
    """Load the Chadwick Bureau player ID registry.

    Downloads from GitHub on first call and caches as Parquet for fast
    subsequent access.  Set ``force_refresh=True`` to re-download.

    Returns:
        DataFrame with columns: key_retro, key_mlbam (int64, nullable),
        key_fangraphs (int64, nullable), name_first, name_last.
        Rows with no Retrosheet *and* no MLBAM ID are excluded.
    """
    if not force_refresh and _REGISTRY_PATH.exists():
        try:
            return pd.read_parquet(_REGISTRY_PATH)
        except Exception:  # noqa: BLE001
            pass

    logger.info("Fetching Chadwick Bureau player registry from GitHub…")
    try:
        df = pd.read_csv(CHADWICK_URL, dtype=str, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch Chadwick registry: %s", exc)
        return pd.DataFrame(columns=_KEEP_COLS)

    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    # Strip whitespace on string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Numeric ID columns
    for col in ("key_mlbam", "key_fangraphs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that lack both a Retrosheet ID and an MLBAM ID
    mask = df.get("key_retro", pd.Series(dtype=str)).notna() | df.get(
        "key_mlbam", pd.Series(dtype=float)
    ).notna()
    df = df[mask].reset_index(drop=True)

    _PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_REGISTRY_PATH, index=False)
    logger.info("Saved player_registry.parquet (%d players)", len(df))
    return df


def retro_to_mlbam(
    retro_id: str,
    registry: pd.DataFrame | None = None,
) -> int | None:
    """Look up an MLBAM (Statcast) player_id from a Retrosheet player ID.

    Args:
        retro_id:  Retrosheet player ID string (e.g., "kersc001").
        registry:  Pre-loaded registry DataFrame.  Loaded if None.

    Returns:
        Integer MLBAM player_id, or None if not found.
    """
    if registry is None:
        registry = load_player_registry()
    row = registry[registry["key_retro"] == retro_id]
    if row.empty or pd.isna(row["key_mlbam"].iloc[0]):
        return None
    return int(row["key_mlbam"].iloc[0])


def build_retro_mlbam_map(registry: pd.DataFrame | None = None) -> dict[str, int]:
    """Return a dict mapping Retrosheet ID → MLBAM ID for every mapped player.

    Useful for vectorized lookups when processing many players at once.
    """
    if registry is None:
        registry = load_player_registry()
    mapped = registry.dropna(subset=["key_retro", "key_mlbam"]).copy()
    mapped["key_mlbam"] = mapped["key_mlbam"].astype(int)
    return dict(zip(mapped["key_retro"], mapped["key_mlbam"]))
