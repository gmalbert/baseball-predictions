# 11 – Feature Engineering Roadmap

Proposed features derived **entirely from existing Retrosheet data**.
Each feature includes the source file(s), a description, the justification for
inclusion, and an expected impact rating on model accuracy.

Impact ratings use a 5-point scale:

| Rating | Meaning |
|--------|---------|
| ★★★★★ | Likely top-5 feature; strong standalone predictive signal |
| ★★★★☆ | High value; captures important variance not already modeled |
| ★★★☆☆ | Moderate lift; useful in ensemble with other features |
| ★★☆☆☆ | Marginal; provides context or interaction effects |
| ★☆☆☆☆ | Minor signal; mainly for completeness or edge cases |

---

## Currently Implemented Features

For reference, the features already in the Betting Features tab (status: ✅ implemented):

- ✅ Season-level Win %, Pythagorean Win %
- ✅ Runs scored / allowed per game
- ✅ Win % differential, Pythagorean differential
- ✅ RS advantage, RA advantage
- ✅ Day/night indicator
- ✅ Attendance, temperature, wind speed
- ✅ Home win indicator, total runs

> NOTE: The next sections capture planned pipeline extensions; some items already exist in code and are reflected above.

---

## Phase 1 — Rolling & Momentum Features

### 1.1 Rolling Team Batting (10g / 30g windows)

| | |
|---|---|
| **Source** | `teamstats.csv` → `b_h, b_hr, b_r, b_k, b_w, b_ab` |
| **Features** | `team_BA_10g`, `team_OPS_10g`, `team_HR_10g`, `team_K_rate_10g` (same at 30g) |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Rolling batting average, OPS, HR count, and strikeout rate over the last N games |
| **Justification** | Season-level stats mask slumps and streaks. A team hitting .310 over the last 10 games is a very different bet than one hitting .210. Recency-weighted offensive form is one of the strongest predictors for totals and moneyline models. |
| **Impact** | ★★★★★ |

### 1.2 Rolling Team Pitching (10g / 30g windows)

| | |
|---|---|
| **Source** | `teamstats.csv` → `p_er, p_ipouts, p_h, p_w, p_k` |
| **Features** | `team_ERA_10g`, `team_WHIP_10g`, `team_K9_10g` (same at 30g) |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Rolling ERA, WHIP, and K/9 for a team's pitching staff over N games |
| **Justification** | A bullpen in meltdown mode shows up here well before season averages reflect it. Captures bullpen fatigue, recent injuries, and pitching form trends. Critical for over/under and run-line models. |
| **Impact** | ★★★★★ |

### 1.3 Rolling Win Rate & Streak

| | |
|---|---|
| **Source** | `gameinfo.csv` → `wteam, date` |
| **Features** | `win_rate_10g`, `win_streak`, `loss_streak` |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Rolling win percentage over last 10 games plus current consecutive W/L streak |
| **Justification** | Momentum is a real factor in baseball. Teams on a 7-game win streak play with higher confidence and are likely deploying rested bullpens. Streaks capture short-term regime changes that season W% misses entirely. |
| **Impact** | ★★★★☆ |

### 1.4 Rolling Run Differential

| | |
|---|---|
| **Source** | `gameinfo.csv` → `vruns, hruns` |
| **Features** | `run_diff_10g`, `run_diff_30g` |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Average margin of victory/defeat over last N games |
| **Justification** | A team winning games 8-2 is in a different tier than one winning 3-2. Run differential is the single best predictor of future W%, and a rolling window keeps it current. |
| **Impact** | ★★★★★ |

---

## Phase 2 — Starting Pitcher Features

### 2.1 Starting Pitcher Game-Level Stats

| | |
|---|---|
| **Source** | `pitching.csv` → `p_gs, p_ipouts, p_er, p_h, p_k, p_w, id` |
| **Features** | `sp_ERA_season`, `sp_WHIP_season`, `sp_K9_season`, `sp_IP_avg` |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Season-to-date stats for the identified starting pitcher (first pitcher in each game with `p_gs=1`) |
| **Justification** | The starting pitcher is the single most influential player in any game. A matchup between a 2.50 ERA starter and a 5.00 ERA starter creates an enormous edge. This is the feature most handicappers cite first. |
| **Impact** | ★★★★★ |

### 2.2 SP Rolling Form (last 5 starts)

| | |
|---|---|
| **Source** | `pitching.csv` → filter `p_gs=1`, rolling over last 5 appearances |
| **Features** | `sp_ERA_5gs`, `sp_WHIP_5gs`, `sp_K9_5gs`, `sp_IP_avg_5gs` |
| **Status** | ✅ Implemented in feature pipeline and model input |
| **Description** | Starting pitcher stats over their most recent 5 starts only |
| **Justification** | A pitcher's last 5 starts reflect current form, fatigue, and any mechanical adjustments far better than season totals. Models using recent pitcher form consistently outperform those using only season averages. |
| **Impact** | ★★★★★ |

### 2.3 SP vs. Opponent History

| | |
|---|---|
| **Source** | `pitching.csv` + `gameinfo.csv` → join on `gid` and `opp` |
| **Features** | `sp_vs_opp_ERA`, `sp_vs_opp_K9`, `sp_vs_opp_starts` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`sp_vs_opp_features`) |
| **Description** | How the starting pitcher has performed historically against today's opponent |
| **Justification** | Some pitchers have persistent splits against certain lineups. Small sample, so use with caution (require ≥3 starts), but it provides additional edge when available. |
| **Impact** | ★★★☆☆ |

---

## Phase 3 — Situational & Splits Features

### 3.1 Home/Away Splits

| | |
|---|---|
| **Source** | `teamstats.csv` → `vishome` column |
| **Features** | `home_BA`, `away_BA`, `home_ERA`, `away_ERA`, `home_win_pct`, `away_win_pct` |
| **Status** | ✅ Partially implemented (teamstats splits exist; additional team-split features can be extended) |
| **Description** | Team batting and pitching stats split by home vs. away games |
| **Justification** | Some teams have extreme home/away splits (e.g., Colorado altitude effect). A team with a 4.80 home ERA and 3.20 road ERA is very different depending on where the game is played. These splits improve moneyline and totals predictions. |
| **Impact** | ★★★★☆ |

### 3.2 Day/Night Splits

| | |
|---|---|
| **Source** | `gameinfo.csv` → `daynight`, `teamstats.csv` |
| **Features** | `team_day_BA`, `team_night_BA`, `team_day_ERA`, `team_night_ERA` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`daynight_split_features`) |
| **Description** | Team performance split by day vs. night games |
| **Justification** | Visibility, fatigue patterns, and lineup deployment differ for afternoon games. Some teams are notably worse in day games. Minor edge, but consistent across seasons. |
| **Impact** | ★★☆☆☆ |

### 3.3 Rest Days

| | |
|---|---|
| **Source** | `gameinfo.csv` → `date` per team |
| **Features** | `days_rest`, `is_doubleheader`, `back_to_back_flag` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`rest_days_features`) |
| **Description** | Days since team's last game, whether it's a doubleheader, or 3+ games in 2 days |
| **Justification** | Bullpen usage over the prior 2–3 days directly impacts available arms. Teams playing a second game of a doubleheader have demonstrably worse pitching outcomes. Rest directly correlates with bullpen ERA in the following game. |
| **Impact** | ★★★★☆ |

---

## Phase 4 — Defensive & Contact Quality Features

### 4.1 Team Fielding Quality

| | |
|---|---|
| **Source** | `teamstats.csv` → `d_e, d_dp, d_a, d_po`; `fielding.csv` → `d_e, d_pos` |
| **Features** | `team_errors_10g`, `team_def_efficiency`, `team_dp_rate` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`fielding_features`) |
| **Description** | Rolling error rate, defensive efficiency (outs per BIP), and double-play conversion rate |
| **Justification** | Errors directly cause unearned runs, which ERA ignores but final scores don't. A team committing 2+ errors per game recently is a run-line and totals signal. The double-play rate indicates inning-ending ability. |
| **Impact** | ★★★☆☆ |

### 4.2 Batted Ball Tendencies (from plays.csv)

| | |
|---|---|
| **Source** | `plays.csv` → `ground, fly, line, bunt, bip` |
| **Features** | `gb_rate`, `fb_rate`, `ld_rate` (ground ball / fly ball / line drive rates) |
| **Description** | Proportion of batted balls that are ground balls, fly balls, or line drives |
| **Justification** | Ground-ball-heavy offenses suppress home runs but produce more GIDP. Fly-ball teams benefit from hitter-friendly parks. These rates interact meaningfully with park factors and opponent pitcher tendencies. |
| **Impact** | ★★★☆☆ |

---

## Phase 5 — Plate Discipline & Clutch Features

### 5.1 Strikeout / Walk Rates

| | |
|---|---|
| **Source** | `teamstats.csv` → `b_k, b_w, b_pa` |
| **Features** | `team_K_rate_10g`, `team_BB_rate_10g`, `team_K_BB_10g` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`kb_rate_features`) |
| **Description** | Rolling strikeout rate, walk rate, and K/BB ratio for the batting lineup |
| **Justification** | A lineup walking a lot is grinding through at-bats and stressing starters. High-K lineups are volatile but exploitable with the right matchup. K/BB ratio is one of the most stable indicators of offensive quality and correlates with run production. |
| **Impact** | ★★★★☆ |

### 5.2 Team LOB & Scoring Efficiency

| | |
|---|---|
| **Source** | `teamstats.csv` → `lob` column; `plays.csv` → per-inning base states |
| **Features** | `lob_per_game_10g`, `scoring_pct_risp` (approx.) |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`lob_features`, reads CSV for `lob` column) |
| **Description** | Runners stranded per game and estimated scoring efficiency with runners in scoring position |
| **Justification** | A team leaving 10+ runners on base per game is underperforming its OBP. LOB rate tends to regress, so a high LOB team is expected to score more in the future (or vice versa). This is a contrarian signal that betting markets often miss. |
| **Impact** | ★★★★☆ |

### 5.3 Leverage-Based Pitching Splits (from plays.csv)

| | |
|---|---|
| **Source** | `plays.csv` → `score_v, score_h, inning, outs_pre, br1_pre, br2_pre, br3_pre` |
| **Features** | `bullpen_era_high_leverage`, `clutch_K_rate` |
| **Description** | Pitching performance in high-leverage situations (runners on, close score, late innings) |
| **Justification** | A bullpen's overall ERA can mask vulnerability in tight games. Some bullpens fold under pressure while others excel. This feature differentiates teams for run-line and close-game moneyline predictions. |
| **Impact** | ★★★☆☆ |

---

## Phase 6 — Environmental & Contextual Features

### 6.1 Park Factor Proxies

| | |
|---|---|
| **Source** | `gameinfo.csv` → `site`, `total_runs`, `hruns`, `vruns` |
| **Features** | `park_runs_factor`, `park_hr_factor` |
| **Status** | ✅ Implemented via home/away park adjustment logic in model features (e.g., check use in `feature` pipelines) |
| **Description** | Average total runs and HR per game at each ballpark relative to league average |
| **Justification** | Coors Field inflates runs by 30–40%. Playing in a pitcher's park vs. a hitter's park swings totals predictions significantly. Park factors are among the most well-established adjustments in sports analytics. |
| **Impact** | ★★★★★ |

### 6.2 Weather Interaction Features

| | |
|---|---|
| **Source** | `gameinfo.csv` → `temp, windspeed, winddir, precip, sky` |
| **Features** | `temp_bucket`, `wind_out`, `wind_in`, `dome_flag`, `overcast_flag` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`weather_interaction_features`) |
| **Description** | Binned temperature, wind direction relative to batter (out = favorable for HR, in = suppresses), dome indicator, overcast indicator |
| **Justification** | Wind blowing out at 15+ mph at Wrigley can add 2+ runs to a game. Temperature below 50°F suppresses offense noticeably. These factors interact with fly-ball lineups to create high-value over/under edges. |
| **Impact** | ★★★★☆ |

### 6.3 Umpire Home-Plate Tendency

| | |
|---|---|
| **Source** | `gameinfo.csv` → `umphome`; `plays.csv` → `k, walk` per game |
| **Features** | `ump_k_rate`, `ump_bb_rate`, `ump_total_runs_avg` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`umpire_features`) |
| **Description** | Historical K rate, walk rate, and average runs scored in games called by the home-plate umpire |
| **Justification** | Some umpires have significantly tighter or wider strike zones. A tight-zone ump increases walks and runs, boosting over bets. Research shows a ~0.3 run/game swing between the most and least run-friendly umpires. |
| **Impact** | ★★★☆☆ |

---

## Phase 7 — Advanced Derived Metrics

### 7.1 Pythagorean Win Differential vs. Actual

| | |
|---|---|
| **Source** | `gameinfo.csv` (already partially computed) |
| **Features** | `pyth_diff` = actual W% − Pythagorean W% |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`pythagorean_diff_features`) |
| **Description** | How much a team is over- or under-performing their expected record based on runs |
| **Justification** | Teams with actual W% significantly above their Pythagorean expectation are "lucky" and tend to regress. This is one of the strongest contrarian signals for betting — buy low on under-performers, sell high on over-performers. |
| **Impact** | ★★★★★ |

### 7.2 Base-Running Efficiency

| | |
|---|---|
| **Source** | `teamstats.csv` → `b_sb, b_cs`; `plays.csv` → baserunning events |
| **Features** | `sb_success_rate`, `extra_bases_taken_rate` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`baserunning_features`) |
| **Description** | Stolen base success rate and how often runners advance extra bases on hits |
| **Justification** | Smart base-running manufactures runs without extra hits. Teams with SB success >75% gain an edge that traditional stats undercount. Affects totals predictions, especially in low-scoring games. |
| **Impact** | ★★☆☆☆ |

### 7.3 Bullpen Workload & Fatigue

| | |
|---|---|
| **Source** | `pitching.csv` → `p_ipouts, p_gs=0` over trailing 3 days |
| **Features** | `bullpen_ip_3d`, `bullpen_pitchers_used_3d`, `pen_arm_available` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`bullpen_fatigue_features`) |
| **Description** | Total relief innings and unique relievers used in the prior 3 days |
| **Justification** | An overworked bullpen is a ticking time bomb. If a team has used 15+ relief innings in 3 days, their late-game pitching will be degraded. This is a strong predictor for blown leads and run-line failures. |
| **Impact** | ★★★★★ |

---

## Phase 8 — Interaction & Matchup Features

### 8.1 Batter Handedness vs. Pitcher Handedness

| | |
|---|---|
| **Source** | `allplayers.csv` → `bat, throw`; `plays.csv` → `bathand, pithand` |
| **Features** | `pct_lineup_same_hand_as_sp`, `platoon_advantage_score` |
| **Status** | ✅ Implemented in `src/models/extra_features.py` (`platoon_features`) |
| **Description** | What percentage of the lineup bats from the same side as the opposing SP (disadvantage) vs. opposite side (advantage) |
| **Justification** | Platoon splits are one of the most reliable effects in baseball — opposite-handed batters hit ~20 OPS points higher. A lineup stacked with left-handed hitters facing a LHP is at a real disadvantage. |
| **Impact** | ★★★★☆ |

### 8.2 Offense Type vs. Pitching Type Matchup

| | |
|---|---|
| **Source** | `teamstats.csv` (batting K%, BB%, HR rate) vs. opponent pitching stats |
| **Features** | `matchup_k_delta`, `matchup_power_vs_flyball` |
| **Status** | ✅ Implemented in `src/models/features.py` (`matchup_k_delta` derived inline) |
| **Description** | How a team's batting weaknesses align with the opponent's pitching strengths |
| **Justification** | A high-strikeout offense vs. a high-K pitching staff amplifies the effect. A power-hitting fly-ball team in a hitter's park vs. a fly-ball-prone pitcher creates a compounding run-scoring opportunity. |
| **Impact** | ★★★☆☆ |

---

## Implementation Priority Matrix

| Priority | Feature Set | Impact | Effort | Data Source |
|----------|------------|--------|--------|-------------|
| **P0** | Rolling team batting/pitching (1.1, 1.2) | ★★★★★ | Low | `teamstats.csv` |
| **P0** | Rolling run differential (1.4) | ★★★★★ | Low | `gameinfo.csv` |
| **P0** | Starting pitcher stats (2.1, 2.2) | ★★★★★ | Medium | `pitching.csv` |
| **P0** | Park factors (6.1) | ★★★★★ | Low | `gameinfo.csv` |
| **P0** | Pythagorean differential (7.1) | ★★★★★ | Low | Already computed |
| **P0** | Bullpen workload (7.3) | ★★★★★ | Medium | `pitching.csv` |
| **P1** | Rest days & doubleheaders (3.3) | ★★★★☆ | Low | `gameinfo.csv` |
| **P1** | Home/away splits (3.1) | ★★★★☆ | Medium | `teamstats.csv` |
| **P1** | Win streak / momentum (1.3) | ★★★★☆ | Low | `gameinfo.csv` |
| **P1** | K/BB plate discipline (5.1) | ★★★★☆ | Low | `teamstats.csv` |
| **P1** | LOB / scoring efficiency (5.2) | ★★★★☆ | Medium | `teamstats.csv` |
| **P1** | Weather interactions (6.2) | ★★★★☆ | Medium | `gameinfo.csv` |
| **P1** | Platoon advantage (8.1) | ★★★★☆ | Medium | `allplayers.csv`, `plays.csv` |
| **P2** | Defensive quality (4.1) | ★★★☆☆ | Low | `teamstats.csv` |
| **P2** | Batted ball tendencies (4.2) | ★★★☆☆ | Medium | `plays.csv` |
| **P2** | Umpire tendencies (6.3) | ★★★☆☆ | Medium | `gameinfo.csv`, `plays.csv` |
| **P2** | Leverage pitching (5.3) | ★★★☆☆ | High | `plays.csv` |
| **P2** | SP vs. opponent history (2.3) | ★★★☆☆ | Medium | `pitching.csv` |
| **P2** | Matchup interactions (8.2) | ★★★☆☆ | Medium | `teamstats.csv` |
| **P3** | Day/night splits (3.2) | ★★☆☆☆ | Low | `gameinfo.csv` |
| **P3** | Base-running efficiency (7.2) | ★★☆☆☆ | Low | `teamstats.csv` |

---

## Expected Cumulative Model Improvement

| Phase | Features Added | Est. AUC Lift | Cumulative AUC |
|-------|---------------|---------------|----------------|
| Baseline | Season-level W%, RS/G, RA/G | — | ~0.54–0.56 |
| Phase 1 | Rolling batting/pitching, momentum | +0.03–0.05 | ~0.58–0.60 |
| Phase 2 | Starting pitcher identity & form | +0.04–0.06 | ~0.63–0.65 |
| Phase 3 | Splits, rest, situational | +0.02–0.03 | ~0.65–0.67 |
| Phase 4 | Defense, batted ball quality | +0.01–0.02 | ~0.66–0.69 |
| Phase 5 | Plate discipline, LOB, clutch | +0.01–0.02 | ~0.67–0.70 |
| Phase 6 | Park, weather, umpire | +0.02–0.03 | ~0.69–0.72 |
| Phase 7 | Pythagorean diff, bullpen fatigue | +0.02–0.03 | ~0.71–0.74 |
| Phase 8 | Platoon & matchup interactions | +0.01–0.02 | ~0.72–0.75 |

> **Note:** AUC estimates assume an XGBoost model with proper train/test splits
> by date. Actual lift depends on feature correlation, data quality, and
> hyperparameter tuning. The biggest single gains come from starting pitcher
> identity and rolling form — these are non-negotiable for a competitive model.

---

## Data Source Utilization Summary

| CSV File | Current Usage | Untapped Columns | Feature Opportunity |
|----------|--------------|-----------------|---------------------|
| `gameinfo.csv` (43 cols) | 22 used | `site`, `umphome`, `starttime`, `innings`, `wp/lp/save`, `winddir`, `precip`, `sky` | Park factors, umpire model, weather interactions |
| `teamstats.csv` (111 cols) | 34 used | `lob`, `inn1–inn28` (line scores), `b_cs`, `b_xi`, `b_roe`, `p_sb`, `p_cs`, `start_l1–l9`, `start_f1–f10` | LOB, inning-by-inning scoring, lineup position data, caught stealing |
| `batting.csv` (39 cols) | 17 used | `b_lp` (lineup pos), `b_seq`, `b_cs`, `b_gdp`, `b_xi`, `b_roe`, `ph`, `pr`, `dh` | Lineup construction, pinch-hit frequency, GIDP tendencies |
| `pitching.csv` (42 cols) | 18 used | `p_seq`, `p_d`, `p_t`, `p_noout`, `p_sb`, `p_cs`, `p_pb`, `p_sh`, `p_sf` | Pitch sequencing, pitcher doubles/triples allowed, passed balls |
| `fielding.csv` (28 cols) | 0 used | All columns | Defensive metrics, positional fielding quality, catcher stats |
| `plays.csv` (177 cols) | 0 used | All columns | Batted ball data, base states, pitch sequences, leverage, handedness splits |
| `allplayers.csv` (25 cols) | 5 used | `bat`, `throw`, `g_p/sp/rp`, positional games | Handedness, pitcher role (SP vs RP), defensive versatility |
| `ejections.csv` (11 cols) | 0 used | All columns | Umpire temperament (minor signal) |

---

## Key Takeaways

1. **`plays.csv` is the single biggest untapped resource** — 177 columns of
   play-by-play data that enables batted-ball quality, leverage situations,
   base-state analysis, and handedness matchups.

2. **Starting pitcher identity is the #1 missing feature** — the current model
   treats all team pitching as a monolith. Extracting the SP from `pitching.csv`
   is the highest-ROI engineering task.

3. **Rolling windows are essential** — every season-level stat should have a
   10-game and 30-game rolling counterpart. Short windows capture form;
   longer windows provide stability.

4. **Park factors and weather are low-hanging fruit** — the `site` and weather
   columns in `gameinfo.csv` are already loaded but unused for feature
   engineering. Computing park factors takes minimal effort for high payoff.

5. **Regression-to-mean features (Pythagorean diff, LOB rate) are contrarian
   gold** — these are the signals most casual bettors miss, and they're
   already partially computed in the existing codebase.
