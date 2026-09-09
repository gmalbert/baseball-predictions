"""One-off script to fetch Baseball Savant leaderboard CSVs.

Usage:
    python scripts/fetch_savant_leaderboards.py

The Savant endpoint accepts multiple comma-separated years in a single request,
so the normal path uses 2 HTTP calls (one for batters, one for pitchers).
If Savant returns an invalid combined-season response, the downloader retries
one season at a time. Results are saved as:

    data_files/raw/batting/savant_batter_2020_2025.csv
    data_files/raw/pitching/savant_pitcher_2020_2025.csv

Adjust the `YEARS` list if you need a different range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running the script directly even when src isn't on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.savant_leaderboard import fetch_all_savant_leaderboards

# Inclusive range 2020–2025 — normally fetched in 2 HTTP requests
YEARS = list(range(2020, 2026))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download historical leaderboards instead of reusing cached files",
    )
    args = parser.parse_args()

    print(f"Preparing Savant leaderboards for years: {YEARS}")
    fetch_all_savant_leaderboards(years=YEARS, force=args.force)
    print("Done!")
