"""Fetch current-season game-by-game MLB player stats from the MLB Stats API.

Writes supplemental parquet files alongside the retrosheet parquets:
    data_files/retrosheet/batting_current.parquet
    data_files/retrosheet/pitching_current.parquet
    data_files/retrosheet/allplayers_current.parquet

These are merged at load time by retrosheet.py (load_batting, load_pitching,
load_players).  Run this script once to backfill, then daily to refresh.

Usage:
    python scripts/fetch_current_season.py              # current year
    python scripts/fetch_current_season.py --season 2025
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import statsapi

CURRENT_YEAR = datetime.date.today().year

OUT_DIR = Path(__file__).resolve().parents[1] / "data_files" / "retrosheet"

# Seconds to sleep between boxscore API calls — stay well under rate limits.
RATE_SLEEP = 0.15

# MLB Stats API team abbreviation → retrosheet 3-letter team code.
ABBREV_TO_RETRO: dict[str, str] = {
    "ATH": "ATH",
    "ATL": "ATL",
    "AZ":  "ARI",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "CWS": "CHA",
    "DET": "DET",
    "HOU": "HOU",
    "KC":  "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD":  "SDN",
    "SEA": "SEA",
    "SF":  "SFN",
    "STL": "SLN",
    "TB":  "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
}

# Known Opening Day dates.  Anything not listed defaults to March 27.
OPENING_DAYS: dict[int, str] = {
    2026: "2026-03-27",
    2025: "2025-03-27",
    2024: "2024-03-20",
    2023: "2023-03-30",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_team_map(season: int) -> dict[int, str]:
    """Return {mlbam_team_id: retrosheet_3_letter_code}."""
    teams = statsapi.get("teams", {"sportId": 1, "season": season})["teams"]
    return {
        t["id"]: ABBREV_TO_RETRO.get(t.get("abbreviation", ""), t.get("abbreviation", "UNK"))
        for t in teams
    }


def _ip_to_outs(ip_str: str) -> int:
    """Convert a traditional IP string ('3.1' = 3⅓ innings) to total outs."""
    try:
        s = str(ip_str).strip()
        if "." in s:
            full, frac = s.split(".", 1)
            return int(full) * 3 + int(frac[0])
        return int(s) * 3
    except (ValueError, IndexError, AttributeError):
        return 0


def _safe_int(val: object) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _last_word(s: str | None) -> str:
    """Return the last word (surname) of a full name string."""
    if not s:
        return ""
    return s.strip().split()[-1]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def _fetch_schedule(season: int, season_start: str) -> list[dict]:
    """Return all completed regular-season games from Opening Day through today."""
    today = datetime.date.today().isoformat()
    raw = statsapi.schedule(start_date=season_start, end_date=today, sportId=1)
    completed = [
        g for g in raw
        if g.get("game_type") == "R"
        and g.get("status") in ("Final", "Game Over", "Completed Early")
    ]
    print(f"Found {len(completed)} completed regular-season games through {today}.")
    return completed


# ---------------------------------------------------------------------------
# Boxscore processing
# ---------------------------------------------------------------------------

def _process_game(
    game: dict,
    team_map: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    """Return (batter_rows, pitcher_rows) for one completed game."""
    game_pk   = game["game_id"]
    date_str  = game["game_date"]          # "2026-03-27"
    date_int  = int(date_str.replace("-", ""))
    game_num  = game.get("game_num", 1) or 1
    home_id   = game["home_id"]
    away_id   = game["away_id"]

    home_retro = team_map.get(home_id, "UNK")
    away_retro = team_map.get(away_id, "UNK")
    gid        = f"{home_retro}{date_int}{game_num - 1}"

    home_score = _safe_int(game.get("home_score"))
    away_score = _safe_int(game.get("away_score"))
    home_win   = 1 if home_score > away_score else 0
    away_win   = 1 if away_score > home_score else 0

    try:
        bs = statsapi.boxscore_data(game_pk)
    except Exception as exc:
        print(f"  Warning: boxscore_data({game_pk}) failed — {exc}")
        return [], []

    batter_rows:  list[dict] = []
    pitcher_rows: list[dict] = []

    # ── Batters ───────────────────────────────────────────────────────────────
    for side, team_id, opp_retro, vishome, win in (
        ("away", away_id, home_retro, "v", away_win),
        ("home", home_id, away_retro, "h", home_win),
    ):
        team_retro = team_map.get(team_id, "UNK")
        for batter in bs.get(f"{side}Batters", []):
            pid = batter.get("personId", 0)
            if pid == 0 or batter.get("battingOrder", "") == "":
                continue  # skip header rows
            ab = _safe_int(batter.get("ab"))
            bb = _safe_int(batter.get("bb"))
            batter_rows.append(
                {
                    "gid":     gid,
                    "id":      str(pid),
                    "team":    team_retro,
                    "b_pa":    ab + bb,          # estimate: AB + BB (HBP/SF not exposed)
                    "b_ab":    ab,
                    "b_r":     _safe_int(batter.get("r")),
                    "b_h":     _safe_int(batter.get("h")),
                    "b_d":     _safe_int(batter.get("doubles")),
                    "b_t":     _safe_int(batter.get("triples")),
                    "b_hr":    _safe_int(batter.get("hr")),
                    "b_rbi":   _safe_int(batter.get("rbi")),
                    "b_w":     bb,
                    "b_k":     _safe_int(batter.get("k")),
                    "b_sb":    _safe_int(batter.get("sb")),
                    "b_hbp":   0,
                    "b_sf":    0,
                    "date":    date_int,
                    "vishome": vishome,
                    "opp":     opp_retro,
                    "win":     win,
                    "loss":    1 - win,
                }
            )

    # ── Pitchers ──────────────────────────────────────────────────────────────
    win_last  = _last_word(game.get("winning_pitcher"))
    loss_last = _last_word(game.get("losing_pitcher"))
    save_last = _last_word(game.get("save_pitcher"))

    for side, team_id, opp_retro, vishome, win in (
        ("away", away_id, home_retro, "v", away_win),
        ("home", home_id, away_retro, "h", home_win),
    ):
        team_retro = team_map.get(team_id, "UNK")
        pitchers   = [
            p for p in bs.get(f"{side}Pitchers", [])
            if p.get("personId", 0) != 0
        ]
        n = len(pitchers)
        for idx, pitcher in enumerate(pitchers):
            pid   = pitcher.get("personId", 0)
            pname = _last_word(pitcher.get("name", ""))
            pitcher_rows.append(
                {
                    "gid":      gid,
                    "id":       str(pid),
                    "team":     team_retro,
                    "p_ipouts": _ip_to_outs(pitcher.get("ip", "0")),
                    "p_bfp":    0,   # not in standard boxscore
                    "p_h":      _safe_int(pitcher.get("h")),
                    "p_hr":     _safe_int(pitcher.get("hr")),
                    "p_r":      _safe_int(pitcher.get("r")),
                    "p_er":     _safe_int(pitcher.get("er")),
                    "p_w":      _safe_int(pitcher.get("bb")),
                    "p_iw":     0,
                    "p_k":      _safe_int(pitcher.get("k")),
                    "p_hbp":    0,
                    "p_wp":     0,
                    "p_bk":     0,
                    "p_gs":     1 if idx == 0 else 0,   # first listed = starter
                    "p_gf":     1 if idx == n - 1 else 0,
                    "p_cg":     1 if n == 1 else 0,
                    "wp":       1 if pname == win_last  and win_last  else 0,
                    "lp":       1 if pname == loss_last and loss_last else 0,
                    "save":     1 if pname == save_last and save_last else 0,
                    "date":     date_int,
                    "vishome":  vishome,
                    "opp":      opp_retro,
                    "win":      win,
                    "loss":     1 - win,
                }
            )

    return batter_rows, pitcher_rows


# ---------------------------------------------------------------------------
# Player registry
# ---------------------------------------------------------------------------

def _build_gameinfo(games: list[dict], team_map: dict[int, str], season: int) -> pd.DataFrame:
    """Build a gameinfo-compatible DataFrame from the schedule list."""
    rows: list[dict] = []
    for game in games:
        date_str  = game["game_date"]
        date_int  = int(date_str.replace("-", ""))
        game_num  = game.get("game_num", 1) or 1
        home_id   = game["home_id"]
        away_id   = game["away_id"]
        home_retro = team_map.get(home_id, "UNK")
        away_retro = team_map.get(away_id, "UNK")
        gid        = f"{home_retro}{date_int}{game_num - 1}"
        home_score = _safe_int(game.get("home_score"))
        away_score = _safe_int(game.get("away_score"))
        wteam = home_retro if home_score > away_score else away_retro
        lteam = away_retro if home_score > away_score else home_retro
        rows.append({
            "gid":      gid,
            "date":     date_int,
            "season":   season,
            "visteam":  away_retro,
            "hometeam": home_retro,
            "vruns":    away_score,
            "hruns":    home_score,
            "wteam":    wteam,
            "lteam":    lteam,
            "gametype": "R",
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["gid"])


def _build_teamstats(bat_df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual batter/pitcher rows into one team-per-game row each."""
    if bat_df.empty:
        return pd.DataFrame()
    group_keys = ["gid", "team", "date", "vishome", "opp", "win", "loss"]
    bat_agg = bat_df.groupby(group_keys, as_index=False).agg(
        b_pa=("b_pa", "sum"),
        b_ab=("b_ab", "sum"),
        b_r=("b_r", "sum"),
        b_h=("b_h", "sum"),
        b_d=("b_d", "sum"),
        b_t=("b_t", "sum"),
        b_hr=("b_hr", "sum"),
        b_rbi=("b_rbi", "sum"),
        b_w=("b_w", "sum"),
        b_k=("b_k", "sum"),
        b_sb=("b_sb", "sum"),
        b_hbp=("b_hbp", "sum"),
        b_sf=("b_sf", "sum"),
    )
    if not pit_df.empty:
        pit_agg = pit_df.groupby(["gid", "team"], as_index=False).agg(
            p_ipouts=("p_ipouts", "sum"),
            p_h=("p_h", "sum"),
            p_hr=("p_hr", "sum"),
            p_r=("p_r", "sum"),
            p_er=("p_er", "sum"),
            p_w=("p_w", "sum"),
            p_k=("p_k", "sum"),
            p_hbp=("p_hbp", "sum"),
            p_wp=("p_wp", "sum"),
            p_bk=("p_bk", "sum"),
        )
        df = bat_agg.merge(pit_agg, on=["gid", "team"], how="left")
    else:
        df = bat_agg
    df["stattype"] = "game"
    return df


def _build_allplayers(
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Build an allplayers-compatible DataFrame by querying the MLB people API."""
    all_ids = sorted(
        int(x) for x in set(bat_df["id"]) | set(pit_df["id"])
        if str(x).isdigit()
    )

    rows: list[dict] = []
    for i in range(0, len(all_ids), 50):
        batch   = all_ids[i : i + 50]
        ids_str = ",".join(str(x) for x in batch)
        try:
            result = statsapi.get("people", {"personIds": ids_str})
            for person in result.get("people", []):
                rows.append(
                    {
                        "id":     str(person.get("id", "")),
                        "last":   person.get("lastName", ""),
                        "first":  person.get("firstName", ""),
                        "bat":    person.get("batSide", {}).get("code", ""),
                        "throw":  person.get("pitchHand", {}).get("code", ""),
                        "team":   "",
                        "g":      0,
                        "g_p":    0, "g_sp": 0, "g_rp": 0, "g_c":  0,
                        "g_1b":   0, "g_2b": 0, "g_3b": 0, "g_ss": 0,
                        "g_lf":   0, "g_cf": 0, "g_rf": 0, "g_of": 0,
                        "g_dh":   0, "g_ph": 0, "g_pr": 0,
                        "first_g": 0,
                        "last_g":  0,
                        "season":  season,
                    }
                )
        except Exception as exc:
            print(f"  Warning: people batch at index {i} failed — {exc}")
        time.sleep(RATE_SLEEP)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Most-recent team per player (from batting logs, fall back to pitching).
    def _latest_team(log_df: pd.DataFrame) -> pd.Series:
        return log_df.sort_values("date").groupby("id")["team"].last()

    team_ser = _latest_team(bat_df)
    if not pit_df.empty:
        pit_team = _latest_team(pit_df)
        # Batting takes priority; fill gaps with pitching.
        team_ser = team_ser.combine_first(pit_team)
    df["team"] = df["id"].map(team_ser).fillna("")

    # Games played counts.
    g_bat = bat_df.groupby("id").size().rename("g_bat")
    g_pit = pit_df.groupby("id").size().rename("g_pit")
    df = df.set_index("id").join(g_bat, how="left").join(g_pit, how="left").reset_index()
    df["g"]   = df["g_bat"].fillna(0).astype(int) + df["g_pit"].fillna(0).astype(int)
    df["g_p"] = df["g_pit"].fillna(0).astype(int)
    df = df.drop(columns=["g_bat", "g_pit"], errors="ignore")

    # First and last game dates — drop 0-initialized placeholders before merging.
    if not bat_df.empty:
        df = df.drop(columns=["first_g", "last_g"], errors="ignore")
        df = df.merge(
            bat_df.groupby("id")["date"].agg(first_g="min", last_g="max").reset_index(),
            on="id", how="left",
        )
        df["first_g"] = df["first_g"].fillna(0).astype(int)
        df["last_g"]  = df["last_g"].fillna(0).astype(int)

    ordered_cols = [
        "id", "last", "first", "bat", "throw", "team",
        "g", "g_p", "g_sp", "g_rp", "g_c",
        "g_1b", "g_2b", "g_3b", "g_ss", "g_lf", "g_cf", "g_rf", "g_of",
        "g_dh", "g_ph", "g_pr", "first_g", "last_g", "season",
    ]
    return df[[c for c in ordered_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch current-season MLB player stats into supplement parquets."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=CURRENT_YEAR,
        help=f"Season year (default: {CURRENT_YEAR})",
    )
    args   = parser.parse_args()
    season = args.season

    season_start = OPENING_DAYS.get(season, f"{season}-03-27")
    print(f"=== Fetching {season} season data (from {season_start}) ===\n")

    team_map = _build_team_map(season)
    games    = _fetch_schedule(season, season_start)

    all_batters:  list[dict] = []
    all_pitchers: list[dict] = []

    for idx, game in enumerate(games, 1):
        if idx % 10 == 0 or idx == 1:
            print(
                f"  [{idx}/{len(games)}] {game['game_id']}  "
                f"{game['away_name']} @ {game['home_name']}  ({game['game_date']})"
            )
        bat_rows, pit_rows = _process_game(game, team_map)
        all_batters.extend(bat_rows)
        all_pitchers.extend(pit_rows)
        time.sleep(RATE_SLEEP)

    print(f"\nRaw rows — batters: {len(all_batters)}, pitchers: {len(all_pitchers)}")

    if not all_batters and not all_pitchers:
        print("No data fetched.  Nothing written.")
        return

    bat_df = pd.DataFrame(all_batters).drop_duplicates(subset=["gid", "id"])
    pit_df = pd.DataFrame(all_pitchers).drop_duplicates(subset=["gid", "id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bat_out = OUT_DIR / "batting_current.parquet"
    pit_out = OUT_DIR / "pitching_current.parquet"
    bat_df.to_parquet(bat_out, index=False)
    pit_df.to_parquet(pit_out, index=False)
    print(f"Wrote {len(bat_df):,} batting rows  → {bat_out.name}")
    print(f"Wrote {len(pit_df):,} pitching rows → {pit_out.name}")

    print("\nBuilding gameinfo supplement …")
    gi_df = _build_gameinfo(games, team_map, season)
    if not gi_df.empty:
        gi_out = OUT_DIR / "gameinfo_current.parquet"
        gi_df.to_parquet(gi_out, index=False)
        print(f"Wrote {len(gi_df):,} gameinfo rows → {gi_out.name}")

    print("\nBuilding teamstats supplement …")
    ts_df = _build_teamstats(bat_df, pit_df)
    if not ts_df.empty:
        ts_out = OUT_DIR / "teamstats_current.parquet"
        ts_df.to_parquet(ts_out, index=False)
        print(f"Wrote {len(ts_df):,} teamstats rows → {ts_out.name}")
    else:
        print("Warning: teamstats supplement was empty.")

    print("\nBuilding player registry …")
    ap_df = _build_allplayers(bat_df, pit_df, season)
    if not ap_df.empty:
        ap_out = OUT_DIR / "allplayers_current.parquet"
        ap_df.to_parquet(ap_out, index=False)
        print(f"Wrote {len(ap_df):,} player rows  → {ap_out.name}")
    else:
        print("Warning: player registry was empty — allplayers_current.parquet not written.")

    print("\nDone.  Commit the *_current.parquet files or re-run daily to refresh.")


if __name__ == "__main__":
    main()
