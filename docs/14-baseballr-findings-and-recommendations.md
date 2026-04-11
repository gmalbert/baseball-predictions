cd # 14 — baseballr Findings & Recommendations

Review of **[BillPetti/baseballr](https://github.com/BillPetti/baseballr)** (R package, v1.6.0) for data source ideas,
metric calculations, scraping patterns, and model features that can be ported to this Python project.

---

## What is baseballr?

An R package providing two categories of functionality:

1. **Data acquisition** — scraping and API wrappers for Baseball Savant/Statcast, FanGraphs, Baseball
   Reference, the official MLB Stats API, Retrosheet, the Chadwick Bureau, and Spotrac.
2. **Metric calculation** — wOBA, FIP, run expectancy matrices, linear weights, barrel coding, edge
   frequency, and team consistency.

The Python ecosystem already covers much of this via `pybaseball`, `statsapi`, and the project's own
`src/ingestion/` layer — but several capabilities are missing or under-used.

---

## Data Sources to Add or Expand

### 1. FanGraphs Guts! Table — `fg_guts()`

**What it provides:** Season-by-season wOBA linear weight constants — `wBB`, `wHBP`, `w1B`, `w2B`,
`w3B`, `wHR`, `woba_scale`, `lg_woba`, `runSB`, `runCS`, `lg_r_pa`, `lg_r_w`, `cFIP`.

**Why it matters:** Without year-specific constants, any wOBA or FIP calculation uses stale weights.
The constants shift meaningfully between dead-ball and juiced-ball eras and year to year.

**Python implementation:**

```python
import pandas as pd

def fetch_fg_guts() -> pd.DataFrame:
    """Fetch FanGraphs Guts! wOBA/FIP constants for all seasons."""
    url = "https://www.fangraphs.com/guts.aspx?type=cn"
    tables = pd.read_html(url, header=0)
    return tables[0]
```

`pybaseball` does not expose this table directly, but a simple `pd.read_html` against
`https://www.fangraphs.com/guts.aspx?type=cn` works without authentication.

**Recommended use:** Store annually in `data_files/processed/fg_guts.parquet`; join to any season-level
wOBA or FIP calculation in `src/models/features.py`.

---

### 2. FanGraphs Park Factors by Handedness — `fg_park_hand()`

**What it provides:** Park-factor index (base 100) split by batter handedness for singles, doubles,
triples, and HR at every stadium.

**Why it matters:** The project already uses a scalar park factor. Adding LHH/RHH splits enables
platoon-aware park adjustments — crucial for matchup analysis and run-line models where a righty
lineup facing Coors Field is very different from a lefty-heavy lineup.

**Python implementation:**

```python
def fetch_fg_park_factors(year: int, handedness: bool = False) -> pd.DataFrame:
    base = "https://www.fangraphs.com/api/stadium/parkfactors"
    params = {"season": year, "hand": "L" if handedness else ""}
    # FanGraphs JSON API — no key required
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())
```

`pybaseball.fg_batting_stats()` fetches player-level FG data but not park factors specifically.
Recommend direct API call or `pd.read_html` from `fangraphs.com/stadium-splits`.

**Recommended use:** Feed into `pages/3_Matchup_Analysis.py` alongside the existing park factor column;
integrate into the run-line and totals models as handedness-adjusted park multipliers.

---

### 3. Statcast Expected Statistics — `statcast_leaderboards(leaderboard="expected_statistics")`

**What it provides:** xBA, xSLG, xwOBA, and their differentials vs. actual stats for every qualified
batter or pitcher — sourced from Baseball Savant's Statcast CSV API.

**Why it matters:** The gap between expected and actual stats (e.g., a pitcher allowing a .320 BA
against a .280 xBA) is a regression signal. Teams/pitchers "due for regression" are valuable inputs for
identifying betting edges.

**Python implementation (`pybaseball`):**

```python
from pybaseball import statcast_batter_expected_stats, statcast_pitcher_expected_stats

xstats_bat = statcast_batter_expected_stats(year=2025)
xstats_pit = statcast_pitcher_expected_stats(year=2025)
```

Or directly via Baseball Savant's JSON endpoint (same endpoint baseballr uses):

```
https://baseballsavant.mlb.com/expected_statistics?type=batter&year=2025&position=&team=&min=q&csv=true
```

**Recommended use:** Add `xwoba_diff` (actual − expected) and `xbabip_diff` as features in the moneyline
and run-line models. A large positive xwoba diff for a starting pitcher suggests he's been luckier than
his stuff indicates — bet against him.

---

### 4. Statcast Barrel Rate / Exit Velocity — `statcast_leaderboards(leaderboard="exit_velocity_barrels")`

**What it provides:** Barrel %, average EV, EV95+ %, max EV, average hit distance per batter.

**Why it matters:** Barrel rate and EV95% are among the most stable early-season predictors of true
offensive talent. They stabilize much faster than batting average (80 PA vs. 600 PA).

**Python implementation:**

```python
# Baseball Savant CSV endpoint
url = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year=2025&type=batter&filter=&sort=4&sortDir=desc"
    "&min=q&selections=xba,xslg,xwoba,barrel_batted_rate,exit_velocity_avg"
    "&csv=true"
)
df = pd.read_csv(url)
```

**Recommended use:** Add `barrel_rate` and `avg_ev` as batter-quality proxies in the totals model;
use opposing pitcher barrel-rate-allowed as a stuff-quality signal in the moneyline model.

---

### 5. Statcast Outs Above Average — `statcast_leaderboards(leaderboard="outs_above_average")`

**What it provides:** Fielder-level OAA by position — a single number quantifying how many outs a
fielder adds or costs relative to an average player at their position.

**Why it matters:** Team-level OAA aggregation gives a defensive quality metric beyond ERA, which
conflates pitching and defense. This is especially useful for run-prevention features in the totals model.

**Python implementation:**

```
https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder&year=2025&team=&range=year&min=q&pos=c&roles=&viz=Show+All&csv=true
```

**Recommended use:** Compute team-level OAA sum as `team_oaa`; include in the totals model alongside
bullpen ERA.

---

### 6. Statcast Sprint Speed — `statcast_leaderboards(leaderboard="sprint_speed")`

**What it provides:** Feet-per-second sprint speed for every player with enough running attempts.

**Why it matters:** Lineup-average sprint speed is a proxy for baserunning quality,
stolen-base threat, and the ability to convert hits to extra bases — all relevant to run scoring.

**Python implementation:**

```
https://baseballsavant.mlb.com/sprint_speed?year=2025&position=&team=&min=10&csv=true
```

**Recommended use:** Aggregate to team average sprint speed (`team_sprint_speed`); use as a feature in
the totals model.

---

### 7. Statcast Arm Strength — `statcast_leaderboards(leaderboard="arm_strength")`

**What it provides:** Max and average arm strength by outfielder/infielder, including throw counts by
base.

**Why it matters:** Weak outfield arms allow more extra bases, boosting scoring. This can complement
OAA for a fuller defensive picture.

---

### 8. Baseball Reference Daily Batter/Pitcher Splits — `bref_daily_batter()` / `bref_daily_pitcher()`

**What it provides:** Window-based (arbitrary date range) batting and pitching aggregates including
wOBA-ready columns (PA, AB, H, 2B, 3B, HR, BB, HBP, etc.).

**Why it matters:** The project's rolling features currently derive from Retrosheet CSVs. BRef's split
endpoint allows arbitrary rolling windows (e.g., last-30-days splits) without building a full game-log
aggregation pipeline.

**Python implementation (`pybaseball`):**

```python
from pybaseball import bref_team_batting_stats, bref_team_pitching_stats
# pybaseball wraps the same bref endpoints; date-range player splits require direct scraping:
url = "https://www.baseball-reference.com/leagues/MLB/2025-schedule.shtml"
```

For date-range batter splits, direct HTML scraping with `requests` + `BeautifulSoup` against
`/friv/dailyleaders.fcgi?type=b&dates=...` works.

---

### 9. FanGraphs Game Logs — `fg_batter_game_logs()` / `fg_pitcher_game_logs()`

**What it provides:** Game-by-game FG stats (including FIP, xFIP, BABIP, LOB%, Hard%) per player.

**Why it matters:** Game logs enable the same rolling-window features we currently derive from
Retrosheet, but include FanGraphs-specific metrics (xFIP, SIERA, SwStr%) that are stronger pitching
predictors than ERA.

**Python implementation:**

```python
from pybaseball import pitching_stats_bref  # per-season
# For FG game logs, construct URL:
# https://www.fangraphs.com/api/players/stats?playerid=XXXX&position=P&z=2025&type=id&stats=game
```

**Recommended use:** Pull SwStr%, CSW% (called strikes + whiffs / total pitches), xFIP for starting
pitcher game logs; these are stronger features than ERA alone in the moneyline model.

---

### 10. Spotrac Payroll Data — `sptrc_league_payrolls()` / `sptrc_team_active_payroll()`

**What it provides:** League-wide and team-level payroll breakdown including active 40-man roster
salary, injured list, and retained salary.

**Why it matters:** Payroll is a rough proxy for roster quality and team depth that doesn't require
model training. A team with 20% of payroll on the IL provides useful context for underdogs.

**Python implementation:** Spotrac has no public API; requires `requests` + HTML scraping:

```python
url = f"https://www.spotrac.com/mlb/payroll/{year}/"
```

**Recommended use:** Optional feature; likely low predictive value but useful for the `7_Info.py`
dashboard page as a team context card.

---

### 11. Chadwick Bureau Player ID Registry

**What it provides:** Cross-reference table linking MLBAM ID, BBRef ID, Retrosheet ID, and FanGraphs ID
for every professional player.

**Why it matters:** The project currently uses Retrosheet IDs for historical data and MLBAM IDs for
Statcast. Joining them without a registry is fragile (name-matching). A single player_lookup table
eliminates downstream merge bugs.

**Python implementation:**

```python
CHADWICK_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
)

def load_player_registry() -> pd.DataFrame:
    return pd.read_csv(CHADWICK_URL, dtype=str, low_memory=False)
```

**Recommended use:** Store as `data_files/processed/player_registry.parquet`; use as the canonical
join key whenever merging Statcast data with Retrosheet or FanGraphs data.

---

## Metrics to Implement in Python

### 1. Team Consistency — `team_consistency()`

**What it does:** Computes the proportion of games in which a team scores ≥ 1 run (offense) and ≤
league average runs (defense) as a measure of "game-to-game consistency" vs. boom-or-bust scoring.

**Formula (conceptually):**

```
Con_R  = proportion of games where team scored > median_runs_scored
Con_RA = proportion of games where team allowed < median_runs_allowed
```

**Why it matters for betting:** High Con_R + high Con_RA → strong totals "under" lean. Highly
inconsistent teams (low Con_R) have wider run distributions — relevant for live betting and
over/under line shopping.

**Python port:**

```python
def team_consistency(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    game_log : DataFrame with columns [team, game_date, runs_scored, runs_allowed]
    """
    med_rs = game_log["runs_scored"].median()
    med_ra = game_log["runs_allowed"].median()
    return (
        game_log.groupby("team")
        .agg(
            con_r=("runs_scored", lambda x: (x > med_rs).mean()),
            con_ra=("runs_allowed", lambda x: (x < med_ra).mean()),
        )
        .reset_index()
    )
```

**Recommended placement:** `src/models/extra_features.py` alongside existing features; feed
`con_r` and `con_ra` into the totals model.

---

### 2. Proper wOBA Calculation — `woba_plus()`

**What it does:** Calculates wOBA using current-year FanGraphs Guts! linear weights rather than
fixed hardcoded constants.

**Why it matters:** The project's Retrosheet-based features currently use counting stats (BA, OPS).
wOBA is a better per-PA offensive value metric, and using the correct year's weights matters in
low-run environments (2014, 2022) vs. high-run years (2019).

**Python port:**

```python
def woba_plus(df: pd.DataFrame, guts: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    df     : batting DataFrame with columns uBB, HBP, H1B, H2B, H3B, HR, AB, SF
    guts   : fg_guts DataFrame indexed by season
    year   : season year
    """
    w = guts[guts["season"] == year].iloc[0]
    df = df.copy()
    df["wOBA"] = (
        w["wBB"] * df["uBB"]
        + w["wHBP"] * df["HBP"]
        + w["w1B"] * df["H1B"]
        + w["w2B"] * df["H2B"]
        + w["w3B"] * df["H3B"]
        + w["wHR"] * df["HR"]
    ) / (df["AB"] + df["uBB"] - df["IBB"] + df["SF"] + df["HBP"])
    df["wOBA_CON"] = (
        w["w1B"] * df["H1B"]
        + w["w2B"] * df["H2B"]
        + w["w3B"] * df["H3B"]
        + w["wHR"] * df["HR"]
    ) / (df["AB"] - df["K"] + df["SF"])
    return df
```

**Recommended placement:** `src/models/features.py`; replace existing OPS usage in the moneyline
and totals feature matrices.

---

### 3. FIP+ Calculation — `fip_plus()`

**What it does:** Calculates FIP (Fielding Independent Pitching) using the year's `cFIP` constant from
FanGraphs Guts! and optionally normalizes to ERA scale (ERA-adjusted).

**Formula:**

$$\text{FIP} = \frac{13 \cdot HR + 3 \cdot (BB + HBP) - 2 \cdot SO}{IP} + cFIP$$

**Why it matters:** FIP outperforms ERA as a predictor of future performance because it removes
defense and luck. Starting-pitcher FIP vs. team ERA differential is a strong moneyline signal.

**Python port:**

```python
def fip_plus(df: pd.DataFrame, guts: pd.DataFrame, year: int) -> pd.DataFrame:
    c_fip = guts[guts["season"] == year]["cFIP"].iloc[0]
    df = df.copy()
    df["FIP"] = (13 * df["HR"] + 3 * (df["BB"] + df["HBP"]) - 2 * df["SO"]) / df["IP"] + c_fip
    return df
```

**Recommended placement:** `src/models/features.py`; use `starter_fip` instead of `starter_era` in
the moneyline model input matrix.

---

### 4. Run Expectancy Matrix — `run_expectancy_code()`

**What it does:** Given pitch-by-pitch Statcast data, attaches the 24-base/out-state run expectancy
(RE24) to each play and calculates plate appearance–level run values.

**Why it matters:** RE24 enables calculation of accurate `linear_weights_above_average` for any event
(single, double, etc.) which is the theoretically correct way to weight offensive value. The current
project uses FanGraphs constants from a table, which is a reasonable approximation, but building the
matrix from the current season's actual data is more accurate.

**Python port concept:**

1. Pull season-level Statcast via `pybaseball.statcast(start_dt, end_dt)`.
2. Construct the 24-state table: `(on_1b, on_2b, on_3b, outs_when_up)`.
3. Compute `avg_re` per state, `next_avg_re` after pitch, `change_re = next_avg_re - avg_re + runs_scored`.

This is a medium-complexity implementation but gives the most accurate linear weights for the current
environment (e.g., 2025 rule changes, shift elimination effects).

**Recommended use:** Annual update; store as `data_files/processed/re24_matrix_{year}.parquet`.

---

### 5. Barrel Coding — `code_barrel()`

**What it does:** Tags each batted ball as a "barrel" if it meets the exit velocity / launch angle
sweet spot criteria defined by Statcast (roughly EV ≥ 98 mph + LA between 26–30°, expanding to
±10° per additional mph).

**Why it matters:** Barrel rate is the single most predictive per-contact quality metric, stabilizing
in ~80 contact events. It's already partially captured by the existing Savant leaderboard ingestion
but useful as a per-pitch/event flag in the Statcast game-by-game data.

**Python port:**

```python
def code_barrel(df: pd.DataFrame) -> pd.DataFrame:
    """Label each batted ball row as barrel=1/0 per Statcast definition."""
    ev = df["launch_speed"]
    la = df["launch_angle"]
    min_ev = 98.0
    is_barrel = (
        (ev >= min_ev)
        & (la >= (26 - (ev - min_ev)))
        & (la <= (30 + (ev - min_ev)))
    )
    df = df.copy()
    df["barrel"] = is_barrel.astype(int)
    return df
```

---

### 6. Edge Codes — `edge_code()` / `edge_frequency()`

**What it does:** Tags each pitch with a zone location code (heart, shadow/edge, chase, waste) based
on `plate_x` and `plate_z` relative to the batter's strike zone. `edge_frequency()` then summarizes
what proportion of pitches a pitcher/hitter throws/sees in each zone.

**Why it matters:** A pitcher who throws a high proportion of pitches to the edge of the zone (shadow
zone) induces more soft contact and strikeouts than a pitcher who lives in the heart. Shadow% is one of
the better predictors of strikeout and hard-hit rate going forward.

**Python port:**

```python
def edge_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns zone labels to each pitch row.
    Requires columns: plate_x, plate_z, sz_top, sz_bot, stand
    """
    # Simplified implementation — full version mirrors baseballr's sch_edge_code.R logic
    half_w = 0.7083  # half strike zone width in feet
    df = df.copy()
    margin = 0.167  # ~2 inches shadow margin
    in_heart_x = df["plate_x"].abs() < (half_w - margin)
    in_edge_x  = df["plate_x"].abs().between(half_w - margin, half_w + margin)
    sz_mid     = (df["sz_top"] + df["sz_bot"]) / 2
    in_heart_z = df["plate_z"].between(df["sz_bot"] + margin, df["sz_top"] - margin)
    in_edge_z  = (
        df["plate_z"].between(df["sz_bot"] - margin, df["sz_bot"] + margin)
        | df["plate_z"].between(df["sz_top"] - margin, df["sz_top"] + margin)
    )
    df["edge_zone"] = "waste"
    df.loc[in_edge_x | in_edge_z, "edge_zone"] = "shadow"
    df.loc[in_heart_x & in_heart_z, "edge_zone"] = "heart"
    df.loc[~in_heart_x & ~in_edge_x & ~in_edge_z, "edge_zone"] = "chase"
    return df
```

**Recommended use:** Aggregate pitcher's shadow% and heart% for each start as `sp_shadow_pct` and
`sp_heart_pct`; include in the moneyline model.

---

## MLb Stats API Functions Worth Porting

`baseballr` wraps every endpoint of the official MLB Stats API at
`https://statsapi.mlb.com/api/v1/...`. The project's `src/ingestion/mlb_stats.py` covers some of
these. Functions with the highest value that may not yet be covered:

| baseballr function | MLB Stats API endpoint | Betting value |
|---|---|---|
| `mlb_game_wp()` | `/game/{gamePk}/winProbability` | In-game WP delta; calibration reference |
| `mlb_game_context_metrics()` | `/game/{gamePk}/contextMetrics` | Leverage index; run-value context |
| `mlb_probables()` | `/schedule?gameType=R&hydrate=probablePitcher` | Daily SP lookup |
| `mlb_batting_orders()` | `/game/{gamePk}/boxscore` | Confirmed lineup once posted |
| `mlb_team_leaders()` | `/teams/stats/leaders` | Quick team-level rankings |
| `mlb_stats_streaks()` | `/stats/streaks` | Hot/cold streak detection |
| `mlb_rosters()` | `/teams/{teamId}/roster?rosterType=fullRoster` | IL check / depth |
| `mlb_game_pace()` | `/schedule/games/pace?season=YYYY` | Innings pace; totals context |

The `mlb_umpire_games()` endpoint is already partially used in the project (umpire runs/g is listed as
an existing context factor in `pages/1_Today.py`). The `load_umpire_ids()` data file (MLBAM IDs for
every umpire since 2008) is a useful complement to call that umpire data more reliably.

---

## Model Ideas Inspired by baseballr

### 1. xStats vs. Actual Stats as Regression Features

**Idea:** For each starting pitcher and opposing lineup, compute:

- `sp_xfip_minus_fip` — how much of SP's ERA is luck vs. skill
- `opp_xwoba_minus_woba` — how much of the lineup's production is sustainable

Positive values → these parties are due for regression; negative → they've been unlucky.
This directly powers the "edge vs. implied odds" calculation in the pick generation pipeline.

### 2. Team Consistency in Totals Model

As described above, `Con_R` and `Con_RA` (scoring consistency percentiles) should be added to the
totals feature matrix alongside existing run-scoring features. High-consistency teams have tighter
run distributions, implying lower variance bets.

### 3. Platoon-Aware Park Factors

FanGraphs' handedness-split park factors (`fg_park_hand`) should replace the single scalar
`park_factor` in the run-line model for more accurate platoon matchup predictions.

### 4. Pitch Arsenal Quality Score

Combining edge% + barrel-rate-allowed + SwStr% into a composite starting-pitcher arsenal score
(similar to how FanGraphs constructs "Stuff+") provides a more stable early-season pitching signal
than ERA or even FIP alone.

### 5. IL & Roster Depth Signal

Using `mlb_rosters(rosterType="fullRoster")` vs. `rosterType="active"` to compute the "payroll on IL"
percentage. A team with >15% of payroll on the IL is a meaningful moneyline fade signal, especially
for underdogs vs. a full-strength opponent.

---

## Implementation Priority

| Priority | Item | Effort | Model Impact |
|---|---|---|---|
| **1** | FanGraphs Guts! table → wOBA/FIP with correct constants | Low | High (all models) |
| **2** | Chadwick Bureau player ID registry | Low | Foundational |
| **3** | Statcast xStats (xwOBA, xBA, xSLG) as regression signal | Low | Moneyline, Run-line |
| **4** | `team_consistency()` Python port | Low | Totals |
| **5** | `fip_plus()` Python port → replace ERA in features | Low | Moneyline |
| **6** | FanGraphs park factors by handedness | Medium | Run-line, Totals |
| **7** | Statcast barrel rate / EV95% leaderboard | Low | Totals, Moneyline |
| **8** | Edge code / shadow% aggregation per SP | Medium | Moneyline |
| **9** | MLB Stats API — `mlb_game_pace`, `mlb_stats_streaks` | Medium | Totals |
| **10** | Run expectancy matrix (RE24) from current season | High | Feature calibration |
| **11** | OAA team aggregate as defensive feature | Medium | Totals |
| **12** | Sprint speed lineup aggregate | Low | Totals |
| **13** | `woba_plus()` full Python port | Medium | All models |
| **14** | Spotrac payroll / IL % | High (scraping) | Context only |

---

## Notes on Scraping Legality and Stability

- **Baseball Savant CSV endpoints** are the same ones `statcast_search()` calls; they are publicly
  documented at `baseballsavant.mlb.com/csv-docs` and are free to use for non-commercial research.
- **FanGraphs** does not have a public API ToS prohibition for light read scraping. Use polite delays
  (`time.sleep(1–2)`) and cache aggressively. The `fg_guts` and `fg_park` pages are stable HTML tables.
- **Baseball Reference** has a known rate-limit of approximately 20 requests/minute. Use the
  `requests_cache` library or save to Parquet on first fetch.
- **Spotrac** is more fragile (JavaScript-rendered sections); consider `selenium` or `playwright` or
  simply skip this source unless payroll context is high-priority.
- **MLB Stats API** is an official API; no authentication is required for read-only game data.

---

## References

- baseballr source: <https://github.com/BillPetti/baseballr>
- baseballr docs: <https://billpetti.github.io/baseballr/reference/index.html>
- Baseball Savant CSV docs: <https://baseballsavant.mlb.com/csv-docs>
- FanGraphs Guts! page: <https://www.fangraphs.com/guts.aspx?type=cn>
- Chadwick Bureau register: <https://github.com/chadwickbureau/register>
- pybaseball (Python equivalent): <https://github.com/jldbc/pybaseball>
