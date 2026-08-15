"""Append-only incident history with impact and recovery actions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Incident:
    incident_id: str
    opened_at: str
    severity: str
    run_id: str
    impact: str
    affected_games: tuple[str, ...]
    affected_markets: tuple[str, ...]
    last_good_run_id: str | None
    probable_cause: str
    recovery_action: str
    status: str = "open"


class IncidentStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, incident: Incident) -> None:
        rows = [] if not self.path.exists() else json.loads(self.path.read_text(encoding="utf-8"))
        if any(row["incident_id"] == incident.incident_id for row in rows):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([*rows, asdict(incident)], indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)
