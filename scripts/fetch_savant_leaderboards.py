"""One-off script to fetch Baseball Savant leaderboard CSVs.

Usage:
    python scripts/fetch_savant_leaderboards.py

The Savant endpoint accepts multiple comma-separated years in a single request,
so all seasons are fetched in exactly 2 HTTP calls (one for batters, one for
pitchers).  Results are saved as:

    data_files/raw/batting/savant_batter_2020_2025.csv
    data_files/raw/pitching/savant_pitcher_2020_2025.csv

Adjust the `YEARS` list if you need a different range.
"""
from __future__ import annotations

from src.ingestion.savant_leaderboard import fetch_all_savant_leaderboards

# Inclusive range 2020–2025 — all fetched in 2 HTTP requests total
YEARS = list(range(2020, 2026))

if __name__ == "__main__":
    print(f"Downloading Savant leaderboards for years: {YEARS}")
    fetch_all_savant_leaderboards(years=YEARS)
    print("Done!")
