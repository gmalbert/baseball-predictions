from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.contracts.domain import stable_game_id
from src.ingestion.config import config
from src.ingestion.mlb_stats import fetch_schedule_for_date


def test_schedule_is_archived_before_normalization(monkeypatch, tmp_path) -> None:
    provider_game = {
        "game_id": 777001,
        "game_date": "2026-08-14",
        "game_datetime": "2026-08-14T23:05:00Z",
        "away_id": 147,
        "home_id": 121,
        "away_name": "New York Yankees",
        "home_name": "New York Mets",
        "venue_name": "Citi Field",
        "status": "Scheduled",
        "game_num": 1,
        "game_type": "R",
    }
    monkeypatch.setattr("src.ingestion.mlb_stats.statsapi.schedule", lambda **_: [provider_game])
    monkeypatch.setattr(config, "project_root", tmp_path)
    observed_at = datetime.now(UTC) - timedelta(seconds=1)

    frame = fetch_schedule_for_date(
        date(2026, 8, 14), as_of=observed_at, run_id="test-schedule-run"
    )

    expected_game_id = stable_game_id(
        season=2026,
        scheduled_start_utc=datetime(2026, 8, 14, 23, 5, tzinfo=UTC),
        away_team_id="147",
        home_team_id="121",
        doubleheader_number=1,
        mlb_game_pk=777001,
    )
    assert frame.loc[0, "game_id"] == expected_game_id
    assert frame.loc[0, "provider_game_id"] == 777001
    assert frame.loc[0, "raw_payload_hash"]
    assert list((tmp_path / "data" / "bronze").rglob("*.payload"))

    archived = list((tmp_path / "data" / "silver").rglob("*.parquet"))
    assert len(archived) == 1
    restored = pd.read_parquet(archived[0])
    assert restored.loc[0, "game_id"] == expected_game_id
    assert restored.loc[0, "ingestion_run_id"] == "test-schedule-run"
