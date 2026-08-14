from datetime import UTC, date, datetime
from pathlib import Path

from src.pipelines.replay import replay_fixture


def test_fixture_replay_hash_is_deterministic() -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "replay_small.json"
    cutoff = datetime(2026, 8, 10, 16, tzinfo=UTC)
    first, first_hash = replay_fixture(fixture, target_date=date(2026, 8, 10), as_of=cutoff)
    second, second_hash = replay_fixture(fixture, target_date=date(2026, 8, 10), as_of=cutoff)
    assert first == second
    assert first_hash == second_hash
    assert first.matched_quotes == 1
