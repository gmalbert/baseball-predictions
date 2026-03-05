# Baseball Predictions – Master Roadmap

## Vision
A baseball betting site that ingests 5+ years of MLB data, runs multiple prediction models (moneyline underdog, run-line spread, over/under totals), and surfaces daily picks with confidence scores and transparent model-performance metrics.

---

## Phase 1 – Data Foundation (Weeks 1-3)
| Task | Doc | Status |
|------|-----|--------|
| Identify & evaluate data sources | [01-data-sources.md](01-data-sources.md) | ⬜ |
| Build ingestion pipelines (historical + daily) | [02-data-ingestion.md](02-data-ingestion.md) | ⬜ |
| Design & provision database schema | [03-database-schema.md](03-database-schema.md) | ⬜ |

## Phase 2 – Modeling & Evaluation (Weeks 4-7)
| Task | Doc | Status |
|------|-----|--------|
| Build underdog, spread & O/U models | [04-betting-models.md](04-betting-models.md) | ⬜ |
| Create evaluation & back-testing harness | [05-model-evaluation.md](05-model-evaluation.md) | ⬜ |
| Daily picks engine (scheduler + output) | [06-daily-picks-engine.md](06-daily-picks-engine.md) | ⬜ |

## Phase 3 – Application Layer (Weeks 8-11)
| Task | Doc | Status |
|------|-----|--------|
| Data access layer (FastAPI optional) | [07-backend-api.md](07-backend-api.md) | ⬜ |
| Streamlit dashboard & UX | [08-frontend-layout.md](08-frontend-layout.md) | ⬜ |

## Phase 4 – Launch & Operations (Weeks 12+)
| Task | Doc | Status |
|------|-----|--------|
| Deployment, CI/CD, monitoring | [09-deployment-ops.md](09-deployment-ops.md) | ⬜ |
| Bankroll & risk management features | [10-bankroll-strategy.md](10-bankroll-strategy.md) | ⬜ |

---

## Things You May Not Have Considered

### Data & Modeling
- **Weather data** – Wind speed/direction at each park significantly affects totals. Several APIs (OpenWeatherMap, Visual Crossing) provide historical & forecast data.
- **Umpire tendencies** – Home-plate umpire strike zones affect run totals. Retrosheet and Baseball Savant track umpire data.
- **Lineup availability timing** – Lineups are confirmed ~2-3 hours before first pitch. Picks generated before lineups drop carry higher uncertainty.
- **Injury reports / IL tracking** – Pitcher scratches flip games dramatically. Integrate an injury feed.
- **Ballpark factors** – Coors Field vs. Oracle Park yields wildly different totals. Fangraphs publishes annual park factors.
- **Platoon splits** – A left-heavy lineup vs. a lefty starter deserves completely different projections.
- **Bullpen usage / fatigue** – Back-to-back heavy bullpen usage lowers reliever effectiveness. Model rest days.
- **Travel & scheduling** – West-coast-to-east-coast travel, day games after night games, and long road trips cause fatigue.

### Product & UX
- **Mobile-first design** – Most bettors check picks on their phone.
- **Push notifications / email alerts** – "Your top pick starts in 30 min."
- **Historical pick ledger** – Transparent, auditable track record page showing every past pick and result.
- **Odds comparison widget** – Show real-time odds from multiple sportsbooks (DraftKings, FanDuel, BetMGM) so users can line-shop.
- **Unit-based bankroll tracker** – Let users log their bets and see P/L over time.
- **Dark mode** – Table-heavy sites benefit enormously from dark mode.
- **Accessibility (WCAG 2.1 AA)** – Color-blind-safe confidence indicators.

### Legal & Compliance
- **State-by-state legality** – Some states prohibit paid picks; others require disclaimers.
- **Responsible gambling disclaimers** – Required by most affiliate programs.
- **Age verification** – If monetized via subscriptions or affiliate links.

### Monetization (optional)
- **Freemium model** – Free O/U picks; paid tier for full slate + confidence scores.
- **Affiliate links** – Partner with sportsbooks for referral revenue.
- **Tip jar / Patreon** – Community-supported.

---

## Tech Stack Recommendation

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.11+ | Dominant in data science & MLB libraries |
| Data store | CSV / Parquet (primary) | Simple, portable, version-controllable with Git |
| Data store (optional) | PostgreSQL | Use if data grows beyond flat-file comfort |
| Task scheduler | APScheduler | Run daily data pulls & model inference |
| ML framework | scikit-learn → XGBoost / LightGBM | Start simple, graduate to gradient boosting |
| Dashboard | Streamlit | All-Python, rapid prototyping, built-in charts |
| Charting | Plotly / Altair (via Streamlit) | Interactive, publication-quality visuals |
| Backend API (optional) | FastAPI | Use if external consumers need JSON endpoints |
| Deployment | Docker → Streamlit Cloud / Railway | Containers for reproducibility |
| CI/CD | GitHub Actions | Already on GitHub |
| Version control | GitHub | Code, data schemas, and model artifacts |

---

*Each numbered doc below dives deep into its topic with implementation code and guidance.*
