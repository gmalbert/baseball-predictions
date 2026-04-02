"""Convert retrosheet CSV files to slim Parquet files for deployment.

Run once locally after updating the CSV data:

    python scripts/build_parquet_data.py

Creates .parquet files next to the .csv files in data_files/retrosheet/.
Only the columns used by retrosheet.py and src/models/features.py are kept,
and rows are pre-filtered to regular-season / value stattype to minimise size.
The .parquet files are committed to git; the large .csv files are gitignored.

Memory-optimisation strategy
-----------------------------
* BUILD_MIN_YEAR=2000 drops ~80 % of rows (data back to 1871 was unused).
* Constant columns 'stattype' and 'gametype' are dropped after filtering.
* Low-cardinality string columns are stored as 'category' dtype; this turns
  30 MB+ of object arrays into a few KB of integer codes + a small dict.
* Binary flag columns (win/loss/save/wp/lp) are stored as int8 (1 byte vs 8).
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data_files" / "retrosheet"

# Only keep rows from this year onward — enough for all model training.
# Drops ~80 % of rows in batting/pitching (data goes back to 1871).
BUILD_MIN_YEAR = 2000


def _lean_write(
    df: pd.DataFrame,
    cat_cols: list[str],
    int8_cols: list[str],
    out: Path,
    label: str,
) -> None:
    """Drop constant filter cols, downcast types, and write parquet."""
    # 'stattype' and 'gametype' are constant after the caller's filter — drop them.
    df = df.drop(columns=["stattype", "gametype"], errors="ignore")
    # Low-cardinality string columns → category (saves ~10-50× vs object dtype).
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Binary flag columns → int8 (1 byte per value vs 8).
    for col in int8_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")
    df.to_parquet(out, index=False)
    mb_disk = out.stat().st_size / 1e6
    mb_mem  = df.memory_usage(deep=True).sum() / 1e6
    print(f"{label}: {len(df):,} rows  ({mb_disk:.1f} MB on disk, ~{mb_mem:.0f} MB in memory)")


def build_gameinfo() -> None:
    cols = [
        "gid", "visteam", "hometeam", "site", "date", "number",
        "daynight", "usedh", "innings", "timeofgame", "attendance",
        "fieldcond", "precip", "sky", "temp", "winddir", "windspeed",
        "vruns", "hruns", "wteam", "lteam", "gametype", "season",
        "umphome", "ump1b", "ump2b", "ump3b", "umplf", "umprf",
    ]
    available = set(pd.read_csv(RAW_DIR / "gameinfo.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "gameinfo.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[df["gametype"] == "regular"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"] >= BUILD_MIN_YEAR].copy()
    _lean_write(
        df,
        cat_cols=["visteam", "hometeam", "site", "daynight", "usedh",
                  "fieldcond", "precip", "sky", "winddir",
                  "wteam", "lteam", "umphome", "ump1b", "ump2b",
                  "ump3b", "umplf", "umprf"],
        int8_cols=[],
        out=RAW_DIR / "gameinfo.parquet",
        label="gameinfo.parquet",
    )


def build_teamstats() -> None:
    cols = [
        "gid", "team", "stattype",
        "b_pa", "b_ab", "b_r", "b_h", "b_d", "b_t", "b_hr",
        "b_rbi", "b_sh", "b_sf", "b_hbp", "b_w", "b_iw", "b_k",
        "b_sb", "b_cs", "b_gdp",
        "p_ipouts", "p_bfp", "p_h", "p_hr", "p_r", "p_er",
        "p_w", "p_iw", "p_k", "p_hbp", "p_wp", "p_bk",
        "d_po", "d_a", "d_e", "d_dp", "lob",
        "date", "vishome", "opp", "win", "loss", "tie", "gametype",
    ]
    available = set(pd.read_csv(RAW_DIR / "teamstats.csv", nrows=0).columns)
    cols = [c for c in cols if c in available]
    df = pd.read_csv(RAW_DIR / "teamstats.csv", usecols=cols, low_memory=False, on_bad_lines="skip")
    df = df[(df["stattype"] == "value") & (df["gametype"] == "regular")].copy()
    df["season"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
    df = df[df["season"] >= BUILD_MIN_YEAR].drop(columns=["season"]).copy()
    _lean_write(
        df,
        cat_cols=["gid", "team", "vishome", "opp"],
        int8_cols=["win", "loss", "tie"],
        out=RAW_DIR / "teamstats.parquet",
        label="teamstats.parquet",
    )


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
    df["season"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
    df = df[df["season"] >= BUILD_MIN_YEAR].drop(columns=["season"]).copy()
    _lean_write(
        df,
        cat_cols=["gid", "id", "team", "vishome", "opp"],
        int8_cols=["win", "loss"],
        out=RAW_DIR / "batting.parquet",
        label="batting.parquet",
    )


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
    df["season"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
    df = df[df["season"] >= BUILD_MIN_YEAR].drop(columns=["season"]).copy()
    _lean_write(
        df,
        cat_cols=["gid", "id", "team", "vishome", "opp"],
        int8_cols=["win", "loss", "wp", "lp", "save"],
        out=RAW_DIR / "pitching.parquet",
        label="pitching.parquet",
    )


def build_allplayers() -> None:
    df = pd.read_csv(RAW_DIR / "allplayers.csv", low_memory=False)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] >= BUILD_MIN_YEAR].copy()
    cat_cols = [c for c in ("team", "bat", "throw", "pos") if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    out = RAW_DIR / "allplayers.parquet"
    df.to_parquet(out, index=False)
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"allplayers.parquet: {len(df):,} rows  ({out.stat().st_size / 1e6:.1f} MB on disk, ~{mb:.0f} MB in memory)")


if __name__ == "__main__":
    print(f"Reading CSVs from {RAW_DIR}\n")
    build_gameinfo()
    build_teamstats()
    build_batting()
    build_pitching()
    build_allplayers()
    print("\nDone. Commit the .parquet files to git.")
