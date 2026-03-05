# 09 – Deployment & Operations

Docker setup, CI/CD, monitoring, and production infrastructure for a Streamlit + Python stack.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Streamlit Cloud  /  Railway  /  VPS          │
│  ┌──────────────┐  ┌───────────────┐                     │
│  │  Streamlit    │  │  Worker       │   data_files/       │
│  │  Dashboard    │  │  (scheduler)  │   ├── raw/*.csv     │
│  │  Port 8501    │  │  APScheduler  │   └── processed/    │
│  └──────────────┘  └───────────────┘       *.parquet      │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Optional: PostgreSQL (if scaling beyond flat files)  │ │
│  │  Optional: FastAPI on port 8000 (external consumers)  │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Docker Setup

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY models/ ./models/
COPY data_files/ ./data_files/

# ---- Streamlit dashboard ----
FROM base AS dashboard
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]

# ---- Worker (scheduler + ingestion) ----
FROM base AS worker
CMD ["python", "-m", "src.ingestion.scheduler"]

# ---- Optional: FastAPI ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Docker Compose (Local Development)

```yaml
# docker-compose.yml
version: "3.9"

services:
  dashboard:
    build:
      context: .
      target: dashboard
    ports:
      - "8501:8501"
    environment:
      ODDS_API_KEY: ${ODDS_API_KEY}
    volumes:
      - ./src:/app/src
      - ./streamlit_app:/app/streamlit_app
      - ./data_files:/app/data_files
      - ./models:/app/models

  worker:
    build:
      context: .
      target: worker
    environment:
      ODDS_API_KEY: ${ODDS_API_KEY}
    volumes:
      - ./data_files:/app/data_files
      - ./models:/app/models
      - ./src:/app/src

  # Optional: PostgreSQL for scaling beyond flat files
  # db:
  #   image: postgres:16
  #   environment:
  #     POSTGRES_USER: postgres
  #     POSTGRES_PASSWORD: postgres
  #     POSTGRES_DB: baseball_predictions
  #   ports:
  #     - "5432:5432"
  #   volumes:
  #     - pgdata:/var/lib/postgresql/data

# volumes:
#   pgdata:
```

---

## 2. GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint
        run: |
          ruff check src/ streamlit_app/
          ruff format --check src/ streamlit_app/

      - name: Run tests
        run: pytest tests/ -v --tb=short --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  deploy:
    needs: [test]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Option A: Deploy to Streamlit Cloud (auto-deploys from GitHub)
      # No action needed — Streamlit Cloud watches the main branch

      # Option B: Deploy to Railway
      - name: Deploy to Railway
        uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: dashboard
```

---

## 3. Streamlit Cloud Deployment

The easiest deployment option — free hosting from Streamlit.

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo (`gmalbert/baseball-predictions`)
4. Set main file: `streamlit_app/app.py`
5. Add secrets in the Streamlit Cloud settings:

```toml
# Streamlit Cloud Secrets (Settings → Secrets)
ODDS_API_KEY = "your_key_here"
```

Access secrets in code:

```python
import streamlit as st
api_key = st.secrets["ODDS_API_KEY"]
```

---

## 4. Environment Variables

```bash
# .env (never commit this file!)

# APIs
ODDS_API_KEY=your_odds_api_key_here

# Optional: PostgreSQL (only if using DB)
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/baseball_predictions

# Optional: Sentry for error tracking
# SENTRY_DSN=https://xxx@sentry.io/yyy
```

```gitignore
# .gitignore additions
.env
.env.local
*.joblib
data_files/raw/
models/*.joblib
__pycache__/
.pytest_cache/
```

---

## 5. Monitoring & Logging

### Structured Logging

```python
# src/logging_config.py
import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)
```

### Health Check

```python
# Streamlit has a built-in health endpoint at /_stcore/health
# For monitoring, point your uptime checker at:
#   https://your-app.streamlit.app/_stcore/health
```

### Uptime Monitoring

- **UptimeRobot** (free) — Ping `/_stcore/health` every 5 min
- **Better Uptime** — More features, free tier available
- **Sentry** — Error tracking + performance monitoring

---

## 6. Backup Strategy

```bash
# Backup Parquet/CSV data files
# scripts/backup_data.sh

#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_FILE="data_backup_${DATE}.tar.gz"

tar -czf "/backups/${BACKUP_FILE}" data_files/processed/ models/

# Upload to S3 (optional)
# aws s3 cp "/backups/${BACKUP_FILE}" "s3://your-bucket/backups/${BACKUP_FILE}"

# Keep only last 30 days
find /backups -name "data_backup_*.tar.gz" -mtime +30 -delete
```

```bash
# Optional: PostgreSQL backup (if using DB)
# pg_dump $DATABASE_URL | gzip > "/backups/db_backup_${DATE}.sql.gz"
```

---

## 7. Cost Estimates (Monthly)

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **Streamlit Cloud** (Dashboard) | Free (public apps) | — |
| **Railway** (Worker / API) | $5 credit/mo | ~$10-25/mo |
| **Odds API** | 500 req/mo free | $20-80/mo |
| **Domain** | — | $12/year |
| **PostgreSQL** (optional, Neon/Supabase) | 0.5 GB free | $25/mo |
| **Total MVP** | **~$0-5/mo** | **~$30-60/mo** |

---

> **Next:** [10-bankroll-strategy.md](10-bankroll-strategy.md) – Bankroll management and risk features.
