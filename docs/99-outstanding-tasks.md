# Outstanding Tasks (ranked by impact)

This summary is based on the current repository status and the existing `docs/` content.

## 1. Backend API (optional but high impact)
- Doc: `07-backend-api.md`
- Current: no FastAPI server code found in repo (`src/api` is not present).
- Why impact:
  - Enables external clients and mobile apps to consume predictions.
  - Supports capacity scaling and separation of model compute from UI.
- Next actions:
  1. Implement API endpoints for schedule, odds, recommendations, and history.
  2. Add OpenAPI schema + optional JWT key auth.
  3. Add tests + CI break/fix.

## 2. Production deployment & CI/CD refinement
- Doc: `09-deployment-ops.md`
- Current: no manifest files for Kubernetes/Cloud run; but Docker is referenced in repo root.
- Why impact:
  - Ensures consistent production rollout and safe updates.
  - Supports monitoring, logging, and automatic rebuilds.
- Next actions:
  1. Verify and add `Dockerfile` + `docker-compose.yml` if needed.
  2. Configure GitHub Actions CI to run lint/test and deploy on push/tag.
  3. Add rollbacks, healthchecks, and alerting (e.g., Sentry or Prometheus).

## 3. Bankroll strategy integration + modeling
- Doc: `10-bankroll-strategy.md`
- Current: there is a working Kelly calculator UI in pages, but no dedicated source module.
- Why impact:
  - Essential for responsible bankroll sizing and long-term expected utility.
- Next actions:
  1. Add a standalone library module for bankroll simulation, bet sizing by tier, risk-of-ruin reports.
  2. Add guided user flows in UI (stake recommendation, historical result analysis).
  3. Link to documented plan in `10-bankroll-strategy.md`.

## 4. Yahoo Fantasy API integration
- Doc: `12-yahoo-fantasy-api.md`
- Current: likely under-doc; no code path in repo currently.
- Why impact:
  - Adds collateral value for daily fantasy players.
- Next actions:
  1. Add authenticated calls to Yahoo Fantasy endpoints.
  2. Map MLB odds to optimized DFS lineups and projections.

## 5. Feature engineering roadmap updates
- Doc: `11-feature-engineering-roadmap.md`
- Current: doc exists; no automated enforcement.
- Why impact:
  - Keeps model feature creation structured.
- Next actions:
  1. Review and implement any remaining new feature candidates (umpire, weather, game scripts, rest/fatigue).
  2. Add tests for feature invariants.

## 6. Programmatic docs completion marker
- `00-roadmap-overview.md` has been updated with checkmarks for implemented items:
  - data sourcing, ingestion, schema, modeling, evaluation, picks engine, frontend.
  - backend, deployment, bankroll remain unfinished.

---

### Quick status checkpoint
- Implemented and verified: `predictions.py` run line orthogonality fix + unit tests in `tests/test_runline_favorite.py` (5 passing).
- Outstanding high-impact story: complete `07-backend-api.md` + API implementation > production deployment infrastructure.
