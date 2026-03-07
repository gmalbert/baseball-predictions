"""Convert retrosheet CSV files to slim Parquet files for deployment.

Run once locally after updating the CSV data:

    python scripts/build_parquet_data.py

Creates .parquet files next to the .csv files in data_files/retrosheet/.
Only the columns used by retrosheet.py and src/models/features.py are kept,
and rows are pre-filtered to regular-season / value stattype to minimise size.
The .parquet files are committed to git; the large .csv files are gitignored.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data_files" / "retrosheet"


def build_gameinfo() -> None:
    cols = [
        "gid", "visteam", "hometeam", "site", "date", "number",
        "daynight", "usedh", "innings", "timeofgame", "attendance",
        "fieldcond", "precip", "sky", "temp", "winddir", "windspeed",
        "vruns", "hruns", "wteam", "lteam", "gametype", "season",
    ]
    available = set(pd.read_csv(RAW_DIR / "gameinfo.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "gameinfo.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[df["gametype"] == "regular"].copy()
    out = RAW_DIR / "gameinfo.parquet"
    df.to_parquet(out, index=False)
    print(f"gameinfo.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB)")


def build_teamstats() -> None:
    cols = [
        "gid", "team", "stattype",
        "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
        "b_rbi", "b_sh", "b_sf", "b_hbp", "b_w", "b_iw", "b_k",
        "b_sb", "b_cs", "b_gdp",
        "p_ipouts", "p_bfp", "p_h", "p_hr", "p_r", "p_er",
        "p_w", "p_iw", "p_k", "p_hbp", "p_wp", "p_bk",
        "d_po", "d_a", "d_e", "d_dp",
        "date", "vishome", "opp", "win", "loss", "tie", "gametype",
    ]
    available = set(pd.read_csv(RAW_DIR / "teamstats.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "teamstats.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[(df["stattype"] == "value") & (df["gametype"] == "regular")].copy()
    out = RAW_DIR / "teamstats.parquet"
    df.to_parquet(out, index=False)
    print(f"teamstats.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB)")


def build_batting() -> None:
    cols = [
        "gid", "id", "team", "stattype",
        "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
        "b_rbi", "b_w", "b_k", "b_sb", "b_hbp", "b_sf",
        "date", "vishome", "opp", "win", "loss", "gametype",
    ]
    available = set(pd.read_csv(RAW_DIR / "batting.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "batting.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[(df["stattype"] == "value") & (df["gametype"] == "regular")].copy()
    out = RAW_DIR / "batting.parquet"
    df.to_parquet(out, index=False)
    print(f"batting.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB)")


def build_pitching() -> None:
    cols = [
        "gid", "id", "team", "stattype",
        "p_ipouts", "p_bfp", "p_h", "p_hr", "p_r", "p_er",
        "p_w", "p_iw", "p_k", "p_hbp", "p_wp", "p_bk",
        "p_gs", "p_gf", "p_cg",
        "wp", "lp", "save",
        "date", "vishome", "opp", "win", "loss", "gametype",
    ]
    available = set(pd.read_csv(RAW_DIR / "pitching.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "pitching.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[(df["stattype"] == "value") & (df["gametype"] == "regular")].copy()
    out = RAW_DIR / "pitching.parquet"
    df.to_parquet(out, index=False)
    print(f"pitching.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB)")


def build_allplayers() -> None:
    df = pd.read_csv(RAW_DIR / "allplayers.csv", low_memory=False)
    out = RAW_DIR / "allplayers.parquet"
    df.to_parquet(out, index=False)
    print(f"allplayers.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    print(f"Reading CSVs from {RAW_DIR}\n")
    build_gameinfo()
    build_teamstats()
    build_batting()
    build_pitching()
    build_allplayers()
    print("\nDone. Commit the .parquet files to git.")
