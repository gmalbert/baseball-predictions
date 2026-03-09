# 13 – Baseball Savant Custom Leaderboard CSV Download

## Overview

The [Baseball Savant Custom Leaderboard](https://baseballsavant.mlb.com/leaderboard/custom) lets you select any combination of Statcast metrics, render them in a table, and **download the result as a CSV** — no scraping required. The data is exposed via a clean URL-parameter API: you specify columns in the `selections` query parameter, and appending `&csv=true` returns the exact table as a CSV file.

This document catalogs every verified column for both **batters (128 columns)** and **pitchers (132 columns)**, maps URL parameter names to display names, provides Python code to download the full CSVs programmatically, and outlines how to integrate these metrics into the feature pipeline.

### Verified Downloads (2024 Season)

| Type    | Columns | Rows (qualified) | Size     |
|---------|---------|-------------------|----------|
| Batter  | 128     | 129               | ~77 KB   |
| Pitcher | 132     | 58                | ~41 KB   |

> **Note:** The manual UI download includes a `player_age` column (129 batter cols) that the programmatic CSV endpoint omits. All other columns match exactly.

---

## How the CSV Download Works

The leaderboard UI has a **"Download CSV"** button. Under the hood, it hits the same URL with `&csv=true` appended. No authentication is required.

```
Base URL: https://baseballsavant.mlb.com/leaderboard/custom

Key query parameters:
  year         – Season (e.g. 2025)
  type         – "batter" or "pitcher"
  filter       – Empty for all players
  min          – "q" (qualified), or an integer minimum PA/IP
  selections   – Comma-separated column keys (see tables below)
  chart        – "false" (we only want the data table)
  sort         – Sort column key (e.g. "xwoba")
  sortDir      – "desc" or "asc"
  csv          – "true" to get CSV instead of HTML
```

Every response automatically includes 4 identifier columns (`last_name, first_name`, `player_id`, `year`) regardless of your `selections`.

---

## All Available Batter Columns (128 total)

### Auto-Included (4 columns)

| Column | Description |
|--------|-------------|
| `last_name, first_name` | Player name (combined into one header) |
| `player_id` | MLB player ID |
| `year` | Season |

### Standard Batting (15 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `ab` | AB | At bats |
| `pa` | PA | Plate appearances |
| `hit` | H | Hits |
| `single` | 1B | Singles |
| `double` | 2B | Doubles |
| `triple` | 3B | Triples |
| `home_run` | HR | Home runs |
| `strikeout` | SO | Strikeouts |
| `walk` | BB | Walks |
| `k_percent` | K% | Strikeout rate |
| `bb_percent` | BB% | Walk rate |
| `batting_avg` | AVG | Batting average |
| `slg_percent` | SLG | Slugging percentage |
| `on_base_percent` | OBP | On-base percentage |
| `on_base_plus_slg` | OPS | OBP + SLG |

### Expected Stats (13 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `xba` | xBA | Expected batting average |
| `xslg` | xSLG | Expected slugging |
| `woba` | wOBA | Weighted on-base average |
| `xwoba` | xwOBA | Expected wOBA (best single quality metric) |
| `xobp` | xOBP | Expected on-base percentage |
| `xiso` | xISO | Expected isolated power |
| `wobacon` | wOBACON | wOBA on contact |
| `xwobacon` | xwOBACON | Expected wOBA on contact |
| `bacon` | BACON | Batting avg on contact |
| `xbacon` | xBACON | Expected batting avg on contact |
| `xbadiff` | xBA Diff | BA − xBA (luck indicator) |
| `xslgdiff` | xSLG Diff | SLG − xSLG |
| `wobadiff` | wOBA Diff | wOBA − xwOBA |

### Bat Tracking (12 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `avg_swing_speed` | Avg Swing Speed | Average bat speed at contact (mph) |
| `fast_swing_rate` | Fast Swing Rate | % of swings ≥ 75 mph |
| `blasts_contact` | Blasts (Contact) | High-quality contact events |
| `blasts_swing` | Blasts (Swing) | High-quality swing events |
| `squared_up_contact` | Squared Up (Contact) | % of contact that is squared up |
| `squared_up_swing` | Squared Up (Swing) | % of swings that are squared up |
| `avg_swing_length` | Avg Swing Length | Average swing path length (ft) |
| `swords` | Swords | Swings & misses on pitches in zone (bad whiffs) |
| `attack_angle` | Attack Angle | Bat angle at contact (°) |
| `attack_direction` | Attack Direction | Bat head direction at contact |
| `ideal_angle_rate` | Ideal Angle Rate | % of swings at ideal bat angle |
| `vertical_swing_path` | Vertical Swing Path | Vertical plane of bat through zone |

### Batted Ball Quality (13 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `exit_velocity_avg` | Avg EV (mph) | Average exit velocity |
| `launch_angle_avg` | Avg LA (°) | Average launch angle |
| `sweet_spot_percent` | Sweet Spot % | Launch angle 8–32° rate |
| `barrel` | Barrels | Total barrel count |
| `barrel_batted_rate` | Barrel % | Barrel rate (ideal EV + LA combo) |
| `solidcontact_percent` | Solid Contact % | Hard + medium contact rate |
| `flareburner_percent` | Flare/Burner % | Weak-ish contact that finds holes |
| `poorlyunder_percent` | Under % | Poorly hit — under the ball |
| `poorlytopped_percent` | Topped % | Poorly hit — topped |
| `poorlyweak_percent` | Weak % | Poorly hit — weak contact |
| `hard_hit_percent` | Hard Hit % | Balls ≥ 95 mph rate |
| `avg_best_speed` | EV50 | Avg of hardest 50% of batted balls |
| `avg_hyper_speed` | Adjusted EV | max(88, actual EV) per BBE |

### Plate Discipline (20 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `z_swing_percent` | Zone Swing % | Swing rate on pitches in zone |
| `z_swing_miss_percent` | Zone Swing & Miss % | Whiff rate on zone pitches |
| `oz_swing_percent` | O-Zone Swing % | Chase rate (swings outside zone) |
| `oz_swing_miss_percent` | O-Zone Swing & Miss % | Whiff rate on chases |
| `oz_contact_percent` | O-Zone Contact % | Contact rate on chases |
| `out_zone_swing_miss` | O-Zone SwgMiss | Out-of-zone swing-and-miss count |
| `out_zone_swing` | O-Zone Swing | Out-of-zone swing count |
| `out_zone_percent` | O-Zone % | % of pitches outside zone |
| `out_zone` | O-Zone | Pitches outside zone count |
| `meatball_swing_percent` | Meatball Swing % | Swing rate on meatballs |
| `meatball_percent` | Meatball % | % of pitches middle-middle |
| `iz_contact_percent` | Zone Contact % | Contact rate on zone pitches |
| `in_zone_swing_miss` | Zone SwgMiss | In-zone swing-and-miss count |
| `in_zone_swing` | Zone Swing | In-zone swing count |
| `in_zone_percent` | Zone % | % of pitches inside zone |
| `in_zone` | Zone | Pitches inside zone count |
| `edge_percent` | Edge % | % of pitches on the edge of zone |
| `edge` | Edge | Pitches on edge count |
| `whiff_percent` | Whiff % | Swings and misses / total swings |
| `swing_percent` | Swing % | Swing rate on all pitches |

### Pitch Counts (4 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `pitch_count_offspeed` | Offspeed Count | Total offspeed pitches seen |
| `pitch_count_fastball` | Fastball Count | Total fastballs seen |
| `pitch_count_breaking` | Breaking Count | Total breaking balls seen |
| `pitch_count` | Pitch Count | Total pitches seen |

### Spray & Distribution (13 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `pull_percent` | Pull % | Pull rate |
| `straightaway_percent` | Straight % | Straightaway rate |
| `opposite_percent` | Oppo % | Opposite field rate |
| `batted_ball` | BBE | Batted ball events total |
| `f_strike_percent` | F-Strike % | First-pitch strike rate |
| `groundballs_percent` | GB % | Ground ball rate |
| `groundballs` | GB | Ground ball count |
| `flyballs_percent` | FB % | Fly ball rate |
| `flyballs` | FB | Fly ball count |
| `linedrives_percent` | LD % | Line drive rate |
| `linedrives` | LD | Line drive count |
| `popups_percent` | PU % | Popup rate |
| `popups` | PU | Popup count |

### Catching (10 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `pop_2b_sba_count` | Pop 2B SBA Count | Steal attempts vs catcher (to 2B) |
| `pop_2b_sba` | Pop 2B SBA | Pop time on throws to 2B |
| `pop_2b_sb` | Pop 2B SB | Pop time on successful steals (2B) |
| `pop_2b_cs` | Pop 2B CS | Pop time on caught stealings (2B) |
| `pop_3b_sba_count` | Pop 3B SBA Count | Steal attempts vs catcher (to 3B) |
| `pop_3b_sba` | Pop 3B SBA | Pop time on throws to 3B |
| `pop_3b_sb` | Pop 3B SB | Pop time on successful steals (3B) |
| `pop_3b_cs` | Pop 3B CS | Pop time on caught stealings (3B) |
| `exchange_2b_3b_sba` | Exchange Time | Catch-to-throw exchange time |
| `maxeff_arm_2b_3b_sba` | Max Eff. Arm | Maximum effective arm strength |

### Fielding — Outs Above Average (16 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `n_outs_above_average` | OAA | Outs above average |
| `n_fieldout_5stars` | 5★ Plays Made | Five-star (hardest) plays made |
| `n_opp_5stars` | 5★ Opportunities | Five-star play opportunities |
| `n_5star_percent` | 5★ Success % | Five-star conversion rate |
| `n_fieldout_4stars` | 4★ Plays Made | Four-star plays made |
| `n_opp_4stars` | 4★ Opportunities | Four-star play opportunities |
| `n_4star_percent` | 4★ Success % | Four-star conversion rate |
| `n_fieldout_3stars` | 3★ Plays Made | Three-star plays made |
| `n_opp_3stars` | 3★ Opportunities | Three-star play opportunities |
| `n_3star_percent` | 3★ Success % | Three-star conversion rate |
| `n_fieldout_2stars` | 2★ Plays Made | Two-star plays made |
| `n_opp_2stars` | 2★ Opportunities | Two-star play opportunities |
| `n_2star_percent` | 2★ Success % | Two-star conversion rate |
| `n_fieldout_1stars` | 1★ Plays Made | One-star (routine) plays made |
| `n_opp_1stars` | 1★ Opportunities | One-star play opportunities |
| `n_1star_percent` | 1★ Success % | One-star conversion rate |

### Running & Speed (8 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `rel_league_reaction_distance` | Reaction Dist. (rel) | Relative-to-league reaction distance |
| `rel_league_burst_distance` | Burst Dist. (rel) | Relative-to-league burst distance |
| `rel_league_routing_distance` | Routing Dist. (rel) | Relative-to-league routing efficiency |
| `rel_league_bootup_distance` | Bootup Dist. (rel) | Relative-to-league bootup distance |
| `f_bootup_distance` | Bootup Dist. | Raw bootup distance |
| `n_bolts` | Bolts | Sprint events ≥ 30 ft/sec |
| `hp_to_1b` | HP to 1B | Home-to-first time (sec) |
| `sprint_speed` | Sprint Speed | Top running speed (ft/sec) |

**Selection count: 15 + 13 + 12 + 13 + 20 + 4 + 13 + 10 + 16 + 8 = 124 selections → 128 total columns (with 4 auto-included)**

---

## All Available Pitcher Columns (132 total)

Pitcher columns include many of the same quality-of-contact and plate discipline metrics (representing what pitchers *allow*), plus **traditional pitching stats**, **pitch-type velocity/spin/movement** for 8 pitch types, and **xERA**.

### Auto-Included (4 columns)

Same as batter: `last_name, first_name`, `player_id`, `year`.

### Shared with Batter Leaderboard (82 selections)

Pitchers share these column keys with the batter leaderboard. The same URL parameter names work — stats represent what the pitcher *allows*:

- **Standard (15):** `ab`, `pa`, `hit`, `single`, `double`, `triple`, `home_run`, `strikeout`, `walk`, `k_percent`, `bb_percent`, `batting_avg`, `slg_percent`, `on_base_percent`, `on_base_plus_slg`
- **Expected (14):** `xba`, `xslg`, `woba`, `xwoba`, `xobp`, `xiso`, `wobacon`, `xwobacon`, `bacon`, `xbacon`, `xbadiff`, `xslgdiff`, `wobadiff`, **`xera`** *(pitcher-only)*
- **Batted Ball Quality (13):** `exit_velocity_avg`, `launch_angle_avg`, `sweet_spot_percent`, `barrel`, `barrel_batted_rate`, `solidcontact_percent`, `flareburner_percent`, `poorlyunder_percent`, `poorlytopped_percent`, `poorlyweak_percent`, `hard_hit_percent`, `avg_best_speed`, `avg_hyper_speed`
- **Plate Discipline (20):** Same 20 columns as batter
- **Pitch Counts & F-Strike (5):** `pitch_count_offspeed`, `pitch_count_fastball`, `pitch_count_breaking`, `pitch_count`, `f_strike_percent`
- **Spray & Distribution (12):** `pull_percent`, `straightaway_percent`, `opposite_percent`, `batted_ball`, `groundballs_percent`, `groundballs`, `flyballs_percent`, `flyballs`, `linedrives_percent`, `linedrives`, `popups_percent`, `popups`

> **Note:** `xera` replaces `xbadiff` in importance for pitchers — it's the expected ERA based on quality of contact allowed.

### Pitcher-Only: Traditional Stats (9 selections)

| URL Parameter | Display Name | Description |
|---------------|-------------|-------------|
| `p_era` | ERA | Earned run average |
| `p_opp_batting_avg` | Opp AVG | Opponent batting average |
| `p_quality_start` | QS | Quality starts |
| `p_game` | G | Games |
| `p_formatted_ip` | IP | Innings pitched |
| `pitch_hand` | Hand | Throwing hand (L/R) |
| `velocity` | Velo | Overall average fastball velocity |
| `release_extension` | Extension | Release extension (ft) |
| `arm_angle` | Arm Angle | Release arm angle (°) |

### Pitcher-Only: Pitch-Type Arsenal (40 selections)

Each pitch type provides: usage count, average speed, average spin rate, horizontal break, and vertical break. Eight pitch types are available:

| Pitch Type | Prefix | Columns |
|------------|--------|---------|
| Four-Seam Fastball | `ff_` | `n_ff_formatted`, `ff_avg_speed`, `ff_avg_spin`, `ff_avg_break_x`, `ff_avg_break_z` |
| Sinker | `si_` | `n_si_formatted`, `si_avg_speed`, `si_avg_spin`, `si_avg_break_x`, `si_avg_break_z` |
| Slider | `sl_` | `n_sl_formatted`, `sl_avg_speed`, `sl_avg_spin`, `sl_avg_break_x`, `sl_avg_break_z` |
| Changeup | `ch_` | `n_ch_formatted`, `ch_avg_speed`, `ch_avg_spin`, `ch_avg_break_x`, `ch_avg_break_z` |
| Curveball | `cu_` | `n_cu_formatted`, `cu_avg_speed`, `cu_avg_spin`, `cu_avg_break_x`, `cu_avg_break_z` |
| Cutter | `fc_` | `n_fc_formatted`, `fc_avg_speed`, `fc_avg_spin`, `fc_avg_break_x`, `fc_avg_break_z` |
| Splitter | `fs_` | `n_fs_formatted`, `fs_avg_speed`, `fs_avg_spin`, `fs_avg_break_x`, `fs_avg_break_z` |
| Knuckleball | `kn_` | `n_kn_formatted`, `kn_avg_speed`, `kn_avg_spin`, `kn_avg_break_x`, `kn_avg_break_z` |

**Selection count: 82 shared + 9 traditional + 40 pitch-type = 128 selections (+ 3 auto = 132 total columns, since `xera` is unique to pitcher)**

> **Note:** Most pitchers only throw 3-4 pitch types. Unused pitch-type columns will be `NaN` / empty.

---

## What We Have Today vs. What Savant Offers

### Current Data Sources in `src/ingestion/pybaseball_stats.py`

| Source | What We Get | Savant Gap |
|--------|-------------|------------|
| `batting_stats()` | FanGraphs batting leaderboard (200+ cols) | NOT Statcast-specific — no xBA, xwOBA, EV50, barrel%, bat tracking |
| `pitching_stats()` | FanGraphs pitching leaderboard | No expected stats, no pitch-type velocity/spin/break |
| `statcast()` | Raw pitch-level data (every pitch) | We have raw data but **never aggregate it**. The CSV gives us pre-computed aggregates |
| `team_batting()` / `team_pitching()` | Team-level FanGraphs | No Statcast quality-of-contact |

### Current Feature Matrix in `src/models/features.py`

The feature set uses **zero Statcast-quality metrics**. All features derive from Retrosheet box scores:
- Team features: WPct, ERA, WHIP, K/9, BA, SLG (from `retrosheet.py`)
- SP features: ERA, WHIP, K/9 (season aggregates)
- Context: temp, windspeed, is_day

**None** of the 128 batter or 132 pitcher Statcast metrics are in the model.

---

## High-Value Columns for Betting Models

### Tier 1 — Strongest predictive signals

| Column | Why It Matters | Model Target |
|--------|---------------|--------------|
| `xwoba` | Best single measure of true offensive quality, regresses luck | All 3 models |
| `barrel_batted_rate` | Barrels = quality contact; strong predictor of future HR/XBH | Totals, Spread |
| `hard_hit_percent` | Correlates with power output; stabilizes faster than BA/SLG | Totals |
| `whiff_percent` | High whiff SP = more Ks = suppresses runs | Totals, Spread |
| `xba` | Separates luck from skill in batting average | Underdog ML |
| `xera` | Expected ERA based on quality of contact (pitcher) | All 3 models |
| `ff_avg_speed` | Fastball velocity — strongest single pitcher quality predictor | Totals, Spread |

### Tier 2 — Strong supporting features

| Column | Why It Matters | Model Target |
|--------|---------------|--------------|
| `k_percent` / `bb_percent` | Discipline profile; high-K lineups fare worse vs elite SP | Spread, Totals |
| `oz_swing_percent` | Chase rate — exploitable by good SP | Underdog ML |
| `exit_velocity_avg` | Raw power; stabilizes quickly | Totals |
| `avg_best_speed` (EV50) | Better than avg EV — focuses on quality contact | Totals |
| `f_strike_percent` | First-pitch strikes set up the at-bat; affects totals | Totals |
| `avg_swing_speed` | Bat tracking — physical tool indicator | All 3 models |
| `velocity` | Overall SP velocity | Spread, Totals |
| `arm_angle` | Extreme arm slots affect platoon splits and deception | Underdog ML |

### Tier 3 — Useful context

| Column | Why It Matters | Model Target |
|--------|---------------|--------------|
| `groundballs_percent` / `flyballs_percent` | GB-heavy teams play differently in specific parks | Spread |
| `sprint_speed` | Baserunning threats affect run expectancy | Spread, Totals |
| `sweet_spot_percent` | Quality of batted ball angles | Totals |
| `squared_up_contact` | Bat tracking quality metric | Totals |
| `n_outs_above_average` | Team defense affects pitcher outcomes | All 3 models |
| `sl_avg_spin` / `cu_avg_spin` | Breaking ball quality | Totals |

---

## Code: Download the Full Leaderboard CSVs

```python
# src/ingestion/savant_leaderboard.py
"""Download pre-computed Statcast leaderboard CSVs from Baseball Savant.

Uses the official CSV download endpoint — no scraping required.
The Custom Leaderboard page provides a 'Download CSV' button;
this module programmatically builds the same URL with all columns selected.

Verified column counts (2024 qualified):
  Batter: 128 columns, ~129 rows
  Pitcher: 132 columns, ~58 rows
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import sleep

import pandas as pd
import requests

from .config import config

logger = logging.getLogger(__name__)

BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

# ── Batter selections (124 columns → 128 with auto-included) ──────────
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

# ── Pitcher selections (128 columns → 132 with auto-included) ─────────
PITCHER_SELECTIONS: list[str] = [
    # Standard batting-against (15)
    "ab", "pa", "hit", "single", "double", "triple", "home_run",
    "strikeout", "walk", "k_percent", "bb_percent",
    "batting_avg", "slg_percent", "on_base_percent", "on_base_plus_slg",
    # Expected stats (14 — includes xera)
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
    # Pitch counts (5 — includes f_strike_percent)
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
    year: int,
    player_type: str = "batter",
    min_pa: str = "q",
    sort_col: str = "xwoba",
) -> pd.DataFrame:
    """Download the full Savant Custom Leaderboard as a DataFrame.

    Args:
        year: Season (e.g. 2025).
        player_type: "batter" or "pitcher".
        min_pa: "q" for qualified, or integer string (e.g. "50").
        sort_col: Column to sort by.

    Returns:
        DataFrame with all selected columns plus player identifiers.
    """
    columns = BATTER_SELECTIONS if player_type == "batter" else PITCHER_SELECTIONS

    params = {
        "year": year,
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
        "Downloading Savant %s leaderboard for %d (min=%s)...",
        player_type, year, min_pa,
    )

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    logger.info("  Received %d rows, %d columns", len(df), len(df.columns))
    return df


def fetch_and_save_batter_leaderboard(
    year: int,
    min_pa: str = "q",
) -> pd.DataFrame:
    """Download batter leaderboard and save to data_files/raw/."""
    df = download_savant_leaderboard(year, "batter", min_pa)

    outpath = config.raw_dir / "batting" / f"savant_batter_{year}.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    logger.info("Saved → %s", outpath)
    return df


def fetch_and_save_pitcher_leaderboard(
    year: int,
    min_pa: str = "q",
) -> pd.DataFrame:
    """Download pitcher leaderboard and save to data_files/raw/."""
    df = download_savant_leaderboard(year, "pitcher", min_pa)

    outpath = config.raw_dir / "pitching" / f"savant_pitcher_{year}.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    logger.info("Saved → %s", outpath)
    return df


def fetch_all_savant_leaderboards() -> None:
    """Download batter + pitcher leaderboards for all configured years."""
    for year in range(config.start_year, config.end_year + 1):
        fetch_and_save_batter_leaderboard(year)
        sleep(5)  # Be respectful of rate limits
        fetch_and_save_pitcher_leaderboard(year)
        sleep(5)

    logger.info("All Savant leaderboard CSVs downloaded.")
```

---

## Lower Minimum PA for Daily Use

During the season, use a lower `min_pa` threshold to capture callups and recent performers:

```python
# In-season daily update — capture all active players
df_batters = download_savant_leaderboard(year=2025, player_type="batter", min_pa="25")
df_pitchers = download_savant_leaderboard(year=2025, player_type="pitcher", min_pa="10")
```

---

## Integrating Into the Feature Pipeline

Once downloaded, Savant CSVs can be joined to game-level data by `player_id` (which maps to MLB's MLBAM ID used across all Savant/Statcast data):

```python
# Proposed additions to src/models/features.py

_SAVANT_BATTER_FEATURES = [
    "home_xwoba", "away_xwoba",
    "home_barrel_rate", "away_barrel_rate",
    "home_hard_hit_pct", "away_hard_hit_pct",
    "home_k_pct", "away_k_pct",
    "home_bb_pct", "away_bb_pct",
    "home_chase_rate", "away_chase_rate",       # oz_swing_percent
    "home_whiff_pct", "away_whiff_pct",
    "home_ev50", "away_ev50",                   # avg_best_speed
    "home_gb_pct", "away_gb_pct",
    "home_avg_swing_speed", "away_avg_swing_speed",
    "home_squared_up_rate", "away_squared_up_rate",
]

_SAVANT_PITCHER_FEATURES = [
    "home_sp_xwoba", "away_sp_xwoba",
    "home_sp_xera", "away_sp_xera",
    "home_sp_barrel_rate", "away_sp_barrel_rate",
    "home_sp_hard_hit_pct", "away_sp_hard_hit_pct",
    "home_sp_whiff_pct", "away_sp_whiff_pct",
    "home_sp_chase_rate", "away_sp_chase_rate",
    "home_sp_k_pct", "away_sp_k_pct",
    "home_sp_gb_pct", "away_sp_gb_pct",
    "home_sp_ff_velo", "away_sp_ff_velo",       # ff_avg_speed
    "home_sp_velocity", "away_sp_velocity",
    "home_sp_extension", "away_sp_extension",   # release_extension
]
```

### Build Team-Level Savant Averages

```python
def build_team_savant_features(
    savant_batters: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-batter Savant stats to team level (PA-weighted)."""
    team_col = "Team" if "Team" in savant_batters.columns else "team"

    agg_cols = [
        "xwoba", "barrel_batted_rate", "hard_hit_percent",
        "k_percent", "bb_percent", "oz_swing_percent",
        "whiff_percent", "avg_best_speed", "groundballs_percent",
        "avg_swing_speed", "squared_up_contact",
    ]

    result = {}
    for team, group in savant_batters.groupby(team_col):
        total_pa = group["pa"].sum()
        row = {"team": team, "total_pa": total_pa}
        for col in agg_cols:
            if col in group.columns:
                valid = group.dropna(subset=[col])
                if len(valid) > 0:
                    row[col] = (valid[col] * valid["pa"]).sum() / valid["pa"].sum()
        result[team] = row

    return pd.DataFrame(result.values())
```

---

## Recommended Fetch Schedule

| Data | Parameters | Frequency | Rationale |
|------|-----------|-----------|-----------|
| Batter leaderboard (qualified) | `type=batter&min=q` | Weekly | Qualified batters = reliable sample |
| Batter leaderboard (low min) | `type=batter&min=25` | Daily in-season | Catch callups and recent activity |
| Pitcher leaderboard (qualified) | `type=pitcher&min=q` | Weekly | SP metrics for model features |
| Pitcher leaderboard (low min) | `type=pitcher&min=10` | Daily in-season | Capture relievers and spot starters |

---

## Rate Limiting & Etiquette

- Baseball Savant is a **free public resource** — be respectful
- The CSV download endpoint is lightweight: one request per leaderboard
- Add `sleep(5)` between requests to avoid hammering the server
- Cache CSVs locally and only re-download when stale (daily at most)
- Use the `min` parameter to limit result sizes when possible
- The endpoint **does not require authentication**

---

## Comparison: Savant CSV vs. pybaseball `statcast()`

| Aspect | Savant CSV Leaderboard | pybaseball `statcast()` |
|--------|----------------------|------------------------|
| Granularity | Player-season aggregates | Pitch-by-pitch raw data |
| Batter cols | 128 (incl. bat tracking, fielding, catching, running) | Raw EV, LA, spin per pitch |
| Pitcher cols | 132 (incl. pitch-type velo/spin/break for 8 types) | Same raw data, must aggregate |
| Size | ~150 rows × 130 cols (~40-80 KB) | 700K+ rows/month (~500 MB/season) |
| Download time | < 2 seconds | Minutes per month chunk |
| Storage | Trivial | Gigabytes |
| Update lag | ~24 hours | ~24 hours |
| Best for | Model features (player quality) | Deep analysis, custom aggregations |

**Recommendation:** Use the Savant CSV leaderboard for daily feature pipeline inputs. Reserve `statcast()` for deep research and custom metric development.

---

## Integration Plan: Savant + Retrosheet (Merge, Not Replace)

### Philosophy

Savant data **supplements** the existing Retrosheet pipeline — it does not replace it. The two sources serve different roles:

| Source | Role | Granularity | What It Provides |
|--------|------|-------------|------------------|
| **Retrosheet** | Game-level backbone | One row per game | Scores, win/loss, team box-score season aggregates (WPct, ERA, WHIP, K/9, BA, SLG), SP starts, weather, day/night |
| **Savant CSV** | Quality-of-contact enrichment | One row per player-season | xwOBA, barrel%, hard-hit%, EV, whiff%, bat tracking, pitch-type arsenal, xERA, sprint speed, OAA, etc. |

The current feature matrix (`build_model_features()` in `src/models/features.py`) produces **32–44 features** per game, all from Retrosheet. Savant features are aggregated to the team level (PA-weighted batting averages, team pitching staff averages) and **merged onto each game row** as additional columns.

### Merge Architecture

```
Game row (Retrosheet)
  │
  ├─ Join (season, hometeam) ─→  home_bat_xwoba, home_bat_barrel_rate, ...
  ├─ Join (season, visteam)  ─→  away_bat_xwoba, away_bat_barrel_rate, ...
  ├─ Join (season, hometeam) ─→  home_pit_xwoba, home_pit_whiff_pct, ...
  └─ Join (season, visteam)  ─→  away_pit_xwoba, away_pit_whiff_pct, ...
```

Each Savant feature becomes **two columns** (home + away), plus optional **differential** columns (e.g., `bat_xwoba_diff = home_bat_xwoba - away_bat_xwoba`).

### What Changes vs. What Stays

| Component | Change? | Details |
|-----------|---------|---------|
| `retrosheet.py` | No change | Continues to load game info, standings, team stats |
| `src/models/features.py` | **Extended** | New `_SAVANT_BATTER_FEATURES` and `_SAVANT_PITCHER_FEATURES` lists added to `ALL_FEATURE_COLS`; `build_model_features()` gains Savant merge step |
| `src/ingestion/savant_leaderboard.py` | **New** ✅ | Already created — downloads batter + pitcher CSVs |
| `src/ingestion/scheduler.py` | **Extended** | Nightly Savant refresh added as a scheduled job |
| Model training | Retrained | Models retrained with expanded feature set after Monte Carlo selects optimal columns |
| Historical CSVs | Already pulled | `savant_batter_2020_2025.csv` and `savant_pitcher_2020_2025.csv` in `data_files/raw/` |

### Merge Key: Team Mapping Challenge

Savant CSV rows have `player_id` (MLBAM ID) but the team association comes from the Savant response itself (not always present as a column). Strategy:

1. **Multi-year historical pull** (2020–2025): The Savant CSV includes player-season rows. We use the Savant "team" column when present.
2. **Nightly refresh**: Current-year pulls include active rosters — team association is reliable.
3. **Fallback**: Cross-reference `player_id` against the MLB Stats API roster endpoint (already in `src/ingestion/mlb_stats.py`).

### Lookahead Bias Note

The current Retrosheet pipeline acknowledges lookahead bias — it joins full-season aggregates. For Savant:

- **Backtesting**: Use the historical pull (2020–2025 archived CSVs). Each player-season row represents the entire season. This is the same level of lookahead as the existing Retrosheet pipeline.
- **Live deployment**: The nightly refresh script (`nightly_savant_refresh.py`) pulls year-to-date cumulative stats — no future data leakage.

---

## Nightly Data Refresh

### Script: `scripts/nightly_savant_refresh.py`

Runs once per night during the MLB season (April–October). Designed for cron / Windows Task Scheduler / GitHub Actions.

```
Schedule:  5:00 AM ET daily
Rationale: Games end by ~1 AM ET → Savant updates by ~4 AM ET
```

**What it does:**

1. Checks if we're in the season window (April–October). Exits early in off-season.
2. Downloads the **current year** batter leaderboard with `min_pa=10` (low threshold to capture callups).
3. Downloads the **current year** pitcher leaderboard with `min_pa=1` (captures relievers and spot starters).
4. Saves as `savant_batter_{year}_latest.csv` and `savant_pitcher_{year}_latest.csv`, overwriting the previous night's file.

**Usage:**
```bash
python scripts/nightly_savant_refresh.py              # auto-detects year
python scripts/nightly_savant_refresh.py --year 2026
python scripts/nightly_savant_refresh.py --force       # run even in off-season
```

**Crontab example (Linux/Mac):**
```cron
0 5 * * * cd /path/to/baseball-predictions && python scripts/nightly_savant_refresh.py
```

**GitHub Actions (`.github/workflows/nightly-savant.yml`):**
```yaml
name: Nightly Savant Refresh
on:
  schedule:
    - cron: '0 9 * * *'  # 9 AM UTC = 5 AM ET
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scripts/nightly_savant_refresh.py
      - uses: actions/upload-artifact@v4
        with:
          name: savant-csvs
          path: data_files/raw/
```

### Wiring Into the Existing Scheduler

The existing scheduler in `src/ingestion/scheduler.py` can add a Savant refresh job:

```python
@scheduler.scheduled_job(CronTrigger(hour=5, minute=0, timezone="America/New_York"))
def nightly_savant_pull() -> None:
    """5 AM ET – Refresh Savant leaderboards with yesterday's games included."""
    if not in_season():
        return
    from src.ingestion.savant_leaderboard import (
        fetch_and_save_batter_leaderboard,
        fetch_and_save_pitcher_leaderboard,
    )
    from datetime import date
    year = date.today().year
    fetch_and_save_batter_leaderboard(year, min_pa="10")
    fetch_and_save_pitcher_leaderboard(year, min_pa="1")
```

---

## Monte Carlo Feature Selection

### Problem

We have **~55 candidate batter features** and **~45 candidate pitcher features** from the Savant leaderboard. Adding all of them would cause overfitting and curse-of-dimensionality issues. We need to find the **optimal subset** for each model.

### Approach: `scripts/monte_carlo_features.py`

Instead of exhaustive search (computationally infeasible with 100 choose 10 combinations), we use **Monte Carlo random search**:

1. Each trial randomly samples **6 batter features** + **4 pitcher features** from the candidate pools.
2. Savant features are aggregated to team-season level (PA-weighted) and merged onto the Retrosheet game matrix.
3. A lightweight XGBoost model is trained with **3-fold TimeSeriesSplit** for each target:
   - `home_win` (moneyline)
   - `home_cover` (spread)
   - `went_over` (totals)
4. ROC-AUC is recorded for each target.
5. After all trials, we identify the **top 10%** by mean AUC across all 3 targets.
6. Features are ranked by **appearance frequency in top trials** — features that consistently appear in winning combinations are the most robustly predictive.

### Why Monte Carlo Over Alternatives?

| Method | Pros | Cons |
|--------|------|------|
| **Forward selection** | Deterministic, finds local optimum | Misses interaction effects; greedy |
| **LASSO / L1** | Fast, built-in regularization | Assumes linear relationships |
| **Exhaustive search** | Guaranteed global optimum | Infeasible with 100+ features |
| **Monte Carlo** ✅ | Captures interactions; simple; parallelizable; shows feature "robustness" | Requires many trials; not guaranteed optimal |

### Usage

```bash
python scripts/monte_carlo_features.py                          # 500 trials (default)
python scripts/monte_carlo_features.py --trials 2000 --top-pct 10
python scripts/monte_carlo_features.py --n-bat 8 --n-pit 5     # sample more per trial
```

### Output Files

| File | Content |
|------|---------|
| `data_files/processed/mc_feature_ranking.csv` | Feature name, type (batter/pitcher), top-trial appearances, appearance rate — sorted by importance |
| `data_files/processed/mc_feature_trials.parquet` | Raw trial results: sampled columns, AUCs for each target, mean AUC |

### Interpreting Results

The ranking CSV will look something like:

```
feature                  type      top_trial_appearances  appearance_rate
xwoba                    batter    42                     0.84
barrel_batted_rate       batter    38                     0.76
whiff_percent            pitcher   36                     0.72
hard_hit_percent         batter    35                     0.70
xera                     pitcher   33                     0.66
k_percent                pitcher   31                     0.62
oz_swing_percent         batter    28                     0.56
...
```

Features with ≥50% appearance rate are strong candidates for the final model. Features below 20% are likely noise or redundant.

### Baseline Comparison

The script also computes a **baseline AUC** using only the existing Retrosheet features (no Savant). This lets you measure the exact lift that Savant features provide:

```
Baseline AUC (no Savant):  moneyline=0.5823  spread=0.5412  totals=0.5567
Top 10% trials (w/ Savant): moneyline=0.6142  spread=0.5689  totals=0.5834
Lift:                        +3.2 pts          +2.8 pts       +2.7 pts
```

---

## Additional Considerations

### 1. Savant Data Availability by Season

Not all columns exist for all years:

| Feature Group | Available Since | Notes |
|---------------|----------------|-------|
| Expected stats (xwOBA, xBA, etc.) | 2015 | First full Statcast season |
| Bat tracking (swing speed, attack angle) | 2024 | Hawk-Eye bat tracking debut |
| Sprint speed | 2015 | First Statcast season |
| OAA (Outs Above Average) | 2016 | Range-based fielding metric |
| Catching (pop time, exchange) | 2015 | Statcast tracking |
| Pitch-type arsenal (velo/spin/break) | 2015 | Per-pitch aggregates |

For the 2020 and 2021 seasons, **bat tracking columns will be entirely null**. The Monte Carlo script handles this gracefully — if a column is all-NaN after merging, it's dropped and the trial uses fewer features.

### 2. 2020 Shortened Season

The 2020 MLB season was only 60 games (COVID). Consider:

- **Small sample sizes**: Rate stats (barrel%, K%, etc.) are noisier with fewer PA.
- **Qualification thresholds**: Savant's "qualified" filter uses pro-rated minimums for 2020.
- **Option**: Weight 2020 data lower or exclude it from training. The Monte Carlo script can be run with `--start-year 2021` if desired.

### 3. Team-Level vs. SP-Level Aggregation

The current merge joins team-level batting and pitching averages. A more precise approach:

- **Batting**: Use lineup-specific averages (starting 9 only, not bench players). Requires roster/lineup data.
- **Pitching**: Join the **starting pitcher's individual Savant stats** rather than team-average pitching. The `player_id` in the Savant CSV maps to the SP identified in Retrosheet's `pitching.csv`.

SP-level joining is a future enhancement planned after validating team-level signals.

### 4. Temporal Staleness

Savant stats are cumulative season-to-date. Early-season data (April) is noisy:
- A batter with 30 PA and a .400 xwOBA might regress to .330 by September.
- **Mitigation**: Weight recent data more heavily, or use multi-year blended averages.

Possible approach:
```python
# Blend current year (60%) with prior year (40%) early in season
savant_blend = current_year * weight + prior_year * (1 - weight)
weight = min(1.0, team_pa / 2000)  # reaches 1.0 ~mid-June
```

### 5. Correlation Between Savant Features

Many Savant features are highly correlated:
- `xwoba` ↔ `xba` ↔ `xslg` (all based on expected stats model)
- `exit_velocity_avg` ↔ `hard_hit_percent` ↔ `avg_best_speed`
- `barrel_batted_rate` ↔ `solidcontact_percent`

The Monte Carlo approach naturally handles this — if two features are redundant, including both won't improve AUC, so only one will appear frequently in top trials. Still, running a **correlation matrix** on the final selected features is recommended before committing to a feature set.

### 6. Feature Stability Across Seasons

Some metrics stabilize faster than others:

| Metric | PA to Stabilize | Predictiveness |
|--------|----------------|----------------|
| `xwoba` | ~200 PA | Very high |
| `barrel_batted_rate` | ~150 PA | High |
| `hard_hit_percent` | ~100 PA | High (stabilizes fast) |
| `whiff_percent` | ~100 PA | High |
| `batting_avg` | ~400+ PA | Low (high variance) |
| `sprint_speed` | ~20 sprints | Very high (physical trait) |
| `avg_swing_speed` | ~50 swings | High (physical trait) |

Features that stabilize fastest provide the most value early in the season when we have limited data.

### 7. Missing Column: Team Mapping

The Savant CSV doesn't always include a team column in the multi-year historical pull. Solutions:

1. **Join via MLB roster API**: Use `player_id` to look up team assignment by date.
2. **Use pybaseball's `playerid_lookup()`**: Maps MLBAM IDs to names/teams.
3. **Current-year nightly pulls**: Team info is more reliably included.

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `src/ingestion/savant_leaderboard.py` | ✅ Created | Download functions, column definitions |
| `scripts/fetch_savant_leaderboards.py` | ✅ Created | One-off bulk download (2020–2025) |
| `scripts/nightly_savant_refresh.py` | ✅ Created | Daily in-season refresh |
| `scripts/monte_carlo_features.py` | ✅ Created | Feature selection via random search |
| `data_files/raw/batting/savant_batter_2020_2025.csv` | ✅ Downloaded | 812 rows × 127 cols |
| `data_files/raw/pitching/savant_pitcher_2020_2025.csv` | ✅ Downloaded | 278 rows × 131 cols |
| `data_files/processed/mc_feature_ranking.csv` | ❌ Pending | Run Monte Carlo to generate |
| `data_files/processed/mc_feature_trials.parquet` | ❌ Pending | Run Monte Carlo to generate |

---

## Next Steps

1. **Run Monte Carlo** — `python scripts/monte_carlo_features.py --trials 1000` to identify optimal feature subset.
2. **Team mapping** — Verify Savant CSVs have team columns; if not, build player_id → team lookup via MLB API.
3. **Update `features.py`** — Add winning Savant features to `_SAVANT_BATTER_FEATURES` and `_SAVANT_PITCHER_FEATURES`; extend `build_model_features()` with the merge step.
4. **Retrain models** — `python scripts/train_models.py` with the expanded feature set.
5. **Evaluate lift** — Compare new model AUCs against current baseline to quantify Savant's contribution.
6. **Early-season blending** — Implement multi-year weighting for April/May when sample sizes are small.
7. **SP-level features** — Join individual pitcher Savant stats instead of team-level averages.
