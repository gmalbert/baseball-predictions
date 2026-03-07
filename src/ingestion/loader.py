# src/ingestion/loader.py
"""Consolidate raw CSV files into optimized Parquet files.

Parquet is the preferred storage format:
- Columnar compression (5-10x smaller than CSV)
- Fast reads with pandas / pyarrow
- Schema enforcement
- Works well with Git LFS for version control
"""

from pathlib import Path

import pandas as pd

from .config import config


def csv_to_parquet(csv_path: Path, parquet_path: Path, **kwargs) -> pd.DataFrame:
    """Convert a single CSV file to Parquet.

    Args:
        csv_path:     Source CSV file.
        parquet_path: Destination Parquet file (parent dirs created automatically).
        **kwargs:     Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns:
        The loaded DataFrame.
    """
    df = pd.read_csv(csv_path, **kwargs)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    csv_size = csv_path.stat().st_size / 1024
    pq_size = parquet_path.stat().st_size / 1024
    print(
        f"  {csv_path.name} ({csv_size:.0f} KB) "
        f"→ {parquet_path.name} ({pq_size:.0f} KB)"
    )
    return df


def consolidate_all() -> None:
    """Consolidate all raw CSVs into processed Parquet files."""
    processed = config.processed_dir

    # Schedules
    sched_csv = config.raw_dir / "gamelogs" / "schedule_all.csv"
    if sched_csv.exists():
        csv_to_parquet(sched_csv, processed / "schedules.parquet")

    # Retrosheet game logs
    retro_csv = config.raw_dir / "gamelogs" / "retrosheet_all.csv"
    if retro_csv.exists():
        csv_to_parquet(retro_csv, processed / "gamelogs.parquet")

    # Batting stats (combine yearly files)
    bat_files = sorted(config.raw_dir.glob("batting/batting_*.csv"))
    bat_files = [f for f in bat_files if "team_" not in f.name and "statcast" not in f.name]
    if bat_files:
        dfs = [pd.read_csv(f) for f in bat_files]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "batting_stats.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(bat_files)} batting files → {out.name} ({len(combined)} rows)")

    # Pitching stats (combine yearly files)
    pitch_files = sorted(config.raw_dir.glob("pitching/pitching_*.csv"))
    pitch_files = [f for f in pitch_files if "team_" not in f.name]
    if pitch_files:
        dfs = [pd.read_csv(f) for f in pitch_files]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "pitching_stats.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(pitch_files)} pitching files → {out.name} ({len(combined)} rows)")

    # Team stats
    for stat_type in ["batting", "pitching"]:
        team_files = sorted(config.raw_dir.glob(f"{stat_type}/team_{stat_type}_*.csv"))
        if team_files:
            dfs = [pd.read_csv(f) for f in team_files]
            combined = pd.concat(dfs, ignore_index=True)
            out = processed / f"team_{stat_type}.parquet"
            combined.to_parquet(out, index=False, engine="pyarrow")
            print(f"  {len(team_files)} team {stat_type} files → {out.name}")

    # Odds (append all snapshots)
    odds_csvs = sorted(config.raw_dir.glob("odds/odds_*.csv"))
    if odds_csvs:
        dfs = [pd.read_csv(f) for f in odds_csvs]
        combined = pd.concat(dfs, ignore_index=True)
        out = processed / "odds_history.parquet"
        combined.to_parquet(out, index=False, engine="pyarrow")
        print(f"  {len(odds_csvs)} odds snapshots → {out.name} ({len(combined)} rows)")

    print("\nConsolidation complete.")
    _report_sizes(processed)


def _report_sizes(directory: Path) -> None:
    """Print file sizes for all Parquet files in a directory."""
    print("\nProcessed data files:")
    for f in sorted(directory.glob("*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        df = pd.read_parquet(f)
        print(f"  {f.name:30s}  {size_mb:6.1f} MB  {len(df):>8,} rows")


if __name__ == "__main__":
    consolidate_all()
