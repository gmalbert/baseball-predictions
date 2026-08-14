from datetime import UTC, date, datetime
from pathlib import Path

from src.monitoring.incidents import Incident, IncidentStore
from src.pipelines.orchestrator import Orchestrator, RunContext


def test_completed_stage_retry_does_not_execute_or_duplicate(tmp_path: Path) -> None:
    context = RunContext.create(
        target_date=date(2026, 8, 10),
        as_of_time=datetime(2026, 8, 10, 16, tzinfo=UTC),
        code_commit="abc",
        environment_lock_hash="lock",
        config={"snapshot": "morning"},
        workspace=tmp_path,
    )
    calls = 0

    def stage() -> tuple[int, int, Path]:
        nonlocal calls
        calls += 1
        output = tmp_path / "output.txt"
        output.write_text("deterministic", encoding="utf-8")
        return 1, 1, output

    orchestrator = Orchestrator(context)
    first = orchestrator.run_stage("snapshot", stage, input_hash="input")
    second = orchestrator.run_stage("snapshot", stage, input_hash="input")
    assert calls == 1
    assert first == second


def test_incident_append_is_idempotent(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents.json")
    incident = Incident(
        incident_id="incident-1",
        opened_at="2026-08-10T16:00:00Z",
        severity="critical",
        run_id="run",
        impact="recommendations blocked",
        affected_games=("g",),
        affected_markets=("moneyline",),
        last_good_run_id="prior",
        probable_cause="stale source",
        recovery_action="restore source",
        status="open",
    )
    store.append(incident)
    store.append(incident)
    assert (tmp_path / "incidents.json").read_text(encoding="utf-8").count("incident-1") == 1
