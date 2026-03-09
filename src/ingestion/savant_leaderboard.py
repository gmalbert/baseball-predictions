"""Download pre-computed Statcast leaderboard CSVs from Baseball Savant.

Uses the official CSV download endpoint — no scraping required.
The Custom Leaderboard page provides a 'Download CSV' button;
this module programmatically builds the same URL with all columns selected.

The endpoint accepts multiple comma-separated years in a single request,
so all seasons can be fetched in one HTTP call per player type.

Verified column counts (qualified players):
  Batter:  128 columns, ~129 rows per season
  Pitcher: 132 columns, ~58 rows per season
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from io import StringIO

from .config import config

logger = logging.getLogger(__name__)

BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

# ── Batter selections (124 keys → 128 total columns with 4 auto-included) ──
BATTER_SELECTIONS: list[str] = [
    # Standard batting (15)
    "ab", "pa", "hit", "single", "double", "triple", "home_run",
    "strikeout", "walk", "k_percent", "bb_percent",
    "batting_avg", "slg_percent", "on_base_percent", "on_base_plus_slg",
    # Expected stats (13)
    "xba", "xslg", "woba", "xwoba", "xobp", "xiso",
    "wobacon", "xwobacon", "bacon", "xbacon",
    "xbadiff", "xslgdiff", "wobadiff",
    # Bat tracking (12)
    "avg_swing_speed", "fast_swing_rate", "blasts_contact", "blasts_swing",
    "squared_up_contact", "squared_up_swing", "avg_swing_length", "swords",
    "attack_angle", "attack_direction", "ideal_angle_rate", "vertical_swing_path",
    # Batted ball quality (13)
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "barrel", "barrel_batted_rate", "solidcontact_percent",
    "flareburner_percent", "poorlyunder_percent", "poorlytopped_percent",
    "poorlyweak_percent", "hard_hit_percent", "avg_best_speed", "avg_hyper_speed",
    # Plate discipline (20)
    "z_swing_percent", "z_swing_miss_percent",
    "oz_swing_percent", "oz_swing_miss_percent", "oz_contact_percent",
    "out_zone_swing_miss", "out_zone_swing", "out_zone_percent", "out_zone",
    "meatball_swing_percent", "meatball_percent",
    "iz_contact_percent", "in_zone_swing_miss", "in_zone_swing",
    "in_zone_percent", "in_zone",
    "edge_percent", "edge", "whiff_percent", "swing_percent",
    # Pitch counts (4)
    "pitch_count_offspeed", "pitch_count_fastball",
    "pitch_count_breaking", "pitch_count",
    # Spray & distribution (13)
    "pull_percent", "straightaway_percent", "opposite_percent", "batted_ball",
    "f_strike_percent",
    "groundballs_percent", "groundballs", "flyballs_percent", "flyballs",
    "linedrives_percent", "linedrives", "popups_percent", "popups",
    # Catching (10)
    "pop_2b_sba_count", "pop_2b_sba", "pop_2b_sb", "pop_2b_cs",
    "pop_3b_sba_count", "pop_3b_sba", "pop_3b_sb", "pop_3b_cs",
    "exchange_2b_3b_sba", "maxeff_arm_2b_3b_sba",
    # Fielding OAA (16)
    "n_outs_above_average",
    "n_fieldout_5stars", "n_opp_5stars", "n_5star_percent",
    "n_fieldout_4stars", "n_opp_4stars", "n_4star_percent",
    "n_fieldout_3stars", "n_opp_3stars", "n_3star_percent",
    "n_fieldout_2stars", "n_opp_2stars", "n_2star_percent",
    "n_fieldout_1stars", "n_opp_1stars", "n_1star_percent",
    # Running & speed (8)
    "rel_league_reaction_distance", "rel_league_burst_distance",
    "rel_league_routing_distance", "rel_league_bootup_distance",
    "f_bootup_distance", "n_bolts", "hp_to_1b", "sprint_speed",
]

# ── Pitcher selections (128 keys → 132 total columns with 4 auto-included) ─
PITCHER_SELECTIONS: list[str] = [
    # Standard batting-against (15)
    "ab", "pa", "hit", "single", "double", "triple", "home_run",
    "strikeout", "walk", "k_percent", "bb_percent",
    "batting_avg", "slg_percent", "on_base_percent", "on_base_plus_slg",
    # Expected stats — includes xera (14)
    "xba", "xslg", "woba", "xwoba", "xobp", "xiso",
    "wobacon", "xwobacon", "bacon", "xbacon",
    "xbadiff", "xslgdiff", "wobadiff", "xera",
    # Batted ball quality allowed (13)
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "barrel", "barrel_batted_rate", "solidcontact_percent",
    "flareburner_percent", "poorlyunder_percent", "poorlytopped_percent",
    "poorlyweak_percent", "hard_hit_percent", "avg_best_speed", "avg_hyper_speed",
    # Plate discipline induced (20)
    "z_swing_percent", "z_swing_miss_percent",
    "oz_swing_percent", "oz_swing_miss_percent", "oz_contact_percent",
    "out_zone_swing_miss", "out_zone_swing", "out_zone_percent", "out_zone",
    "meatball_swing_percent", "meatball_percent",
    "iz_contact_percent", "in_zone_swing_miss", "in_zone_swing",
    "in_zone_percent", "in_zone",
    "edge_percent", "edge", "whiff_percent", "swing_percent",
    # Pitch counts + f-strike (5)
    "pitch_count_offspeed", "pitch_count_fastball",
    "pitch_count_breaking", "pitch_count", "f_strike_percent",
    # Spray & distribution (12)
    "pull_percent", "straightaway_percent", "opposite_percent", "batted_ball",
    "groundballs_percent", "groundballs", "flyballs_percent", "flyballs",
    "linedrives_percent", "linedrives", "popups_percent", "popups",
    # Traditional pitcher stats (9)
    "p_era", "p_opp_batting_avg", "p_quality_start", "p_game",
    "p_formatted_ip", "pitch_hand", "velocity", "release_extension", "arm_angle",
    # Pitch-type: Four-Seam Fastball (5)
    "n_ff_formatted", "ff_avg_speed", "ff_avg_spin", "ff_avg_break_x", "ff_avg_break_z",
    # Pitch-type: Sinker (5)
    "n_si_formatted", "si_avg_speed", "si_avg_spin", "si_avg_break_x", "si_avg_break_z",
    # Pitch-type: Slider (5)
    "n_sl_formatted", "sl_avg_speed", "sl_avg_spin", "sl_avg_break_x", "sl_avg_break_z",
    # Pitch-type: Changeup (5)
    "n_ch_formatted", "ch_avg_speed", "ch_avg_spin", "ch_avg_break_x", "ch_avg_break_z",
    # Pitch-type: Curveball (5)
    "n_cu_formatted", "cu_avg_speed", "cu_avg_spin", "cu_avg_break_x", "cu_avg_break_z",
    # Pitch-type: Cutter (5)
    "n_fc_formatted", "fc_avg_speed", "fc_avg_spin", "fc_avg_break_x", "fc_avg_break_z",
    # Pitch-type: Splitter (5)
    "n_fs_formatted", "fs_avg_speed", "fs_avg_spin", "fs_avg_break_x", "fs_avg_break_z",
    # Pitch-type: Knuckleball (5)
    "n_kn_formatted", "kn_avg_speed", "kn_avg_spin", "kn_avg_break_x", "kn_avg_break_z",
]


def download_savant_leaderboard(
    years: int | list[int],
    player_type: str = "batter",
    min_pa: str = "q",
    sort_col: str = "xwoba",
) -> pd.DataFrame:
    """Download the Savant Custom Leaderboard for one or more seasons.

    Args:
        years: A single season (e.g. 2025) or a list of seasons
               (e.g. [2020, 2021, 2022]).  Multiple years are fetched in
               a single HTTP request.
        player_type: "batter" or "pitcher".
        min_pa: "q" for qualified, or integer string (e.g. "50").
        sort_col: Column to sort by.

    Returns:
        DataFrame with one row per player-season and all selected columns.
    """
    if isinstance(years, int):
        years = [years]

    year_str = ",".join(str(y) for y in years)
    columns = BATTER_SELECTIONS if player_type == "batter" else PITCHER_SELECTIONS

    params = {
        "year": year_str,
        "type": player_type,
        "filter": "",
        "min": min_pa,
        "selections": ",".join(columns),
        "chart": "false",
        "x": "pa",
        "y": "pa",
        "r": "no",
        "chartType": "beeswarm",
        "sort": sort_col,
        "sortDir": "desc",
        "csv": "true",
    }

    logger.info(
        "Downloading Savant %s leaderboard for %s (min=%s)...",
        player_type, year_str, min_pa,
    )

    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    # Normalize the split name header Savant emits
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    logger.info("  Received %d rows, %d columns", len(df), len(df.columns))
    return df


def fetch_and_save_batter_leaderboard(
    years: int | list[int],
    min_pa: str = "q",
) -> pd.DataFrame:
    """Download batter leaderboard(s) and save to data_files/raw/batting/.

    If multiple years are supplied they are combined into one CSV named
    ``savant_batter_<first>_<last>.csv``.
    """
    df = download_savant_leaderboard(years, "batter", min_pa)

    if isinstance(years, int):
        fname = f"savant_batter_{years}.csv"
    else:
        fname = f"savant_batter_{min(years)}_{max(years)}.csv"

    outpath = config.raw_dir / "batting" / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    logger.info("Saved → %s", outpath)
    return df


def fetch_and_save_pitcher_leaderboard(
    years: int | list[int],
    min_pa: str = "q",
) -> pd.DataFrame:
    """Download pitcher leaderboard(s) and save to data_files/raw/pitching/.

    If multiple years are supplied they are combined into one CSV named
    ``savant_pitcher_<first>_<last>.csv``.
    """
    df = download_savant_leaderboard(years, "pitcher", min_pa)

    if isinstance(years, int):
        fname = f"savant_pitcher_{years}.csv"
    else:
        fname = f"savant_pitcher_{min(years)}_{max(years)}.csv"

    outpath = config.raw_dir / "pitching" / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    logger.info("Saved → %s", outpath)
    return df


def fetch_all_savant_leaderboards(
    years: list[int] | None = None,
) -> None:
    """Download batter + pitcher leaderboards in two HTTP requests total.

    Args:
        years: List of seasons to include.  Defaults to
               ``range(config.start_year, config.end_year + 1)``.
    """
    if years is None:
        years = list(range(config.start_year, config.end_year + 1))

    fetch_and_save_batter_leaderboard(years)
    sleep(5)
    fetch_and_save_pitcher_leaderboard(years)

    logger.info("All Savant leaderboard CSVs downloaded.")
