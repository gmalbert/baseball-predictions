"""Champion/challenger lifecycle with quarantine and rollback pointers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

ALLOWED_STATES = {"development", "shadow", "challenger", "champion", "retired", "quarantined"}


@dataclass(frozen=True)
class RegistryEntry:
    model_run_id: str
    model_name: str
    market_id: str
    artifact_uri: str
    manifest_uri: str
    status: str
    registered_at: str
    promoted_at: str | None = None
    reason: str | None = None


class ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> list[RegistryEntry]:
        if not self.path.exists():
            return []
        return [RegistryEntry(**row) for row in json.loads(self.path.read_text(encoding="utf-8"))]

    def _save(self, entries: list[RegistryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(row) for row in entries], indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def register(self, entry: RegistryEntry) -> None:
        if entry.status not in ALLOWED_STATES:
            raise ValueError(f"Invalid registry state: {entry.status}")
        entries = self._load()
        if any(row.model_run_id == entry.model_run_id for row in entries):
            raise ValueError(f"Duplicate model_run_id: {entry.model_run_id}")
        self._save([*entries, entry])

    def promote(self, model_run_id: str, *, gates_passed: bool, reason: str) -> RegistryEntry:
        if not gates_passed:
            raise ValueError("Promotion gates did not pass")
        now = datetime.now(UTC).isoformat()
        entries = self._load()
        target = next((row for row in entries if row.model_run_id == model_run_id), None)
        if target is None:
            raise KeyError(model_run_id)
        updated = []
        for row in entries:
            if row.market_id == target.market_id and row.status == "champion":
                row = replace(row, status="retired", reason=f"superseded by {model_run_id}")
            if row.model_run_id == model_run_id:
                row = replace(row, status="champion", promoted_at=now, reason=reason)
            updated.append(row)
        self._save(updated)
        return next(row for row in updated if row.model_run_id == model_run_id)

    def champion(self, market_id: str) -> RegistryEntry:
        matches = [
            row for row in self._load() if row.market_id == market_id and row.status == "champion"
        ]
        if len(matches) != 1:
            raise LookupError(
                f"Expected exactly one champion for {market_id}; found {len(matches)}"
            )
        return matches[0]

    def rollback(self, market_id: str, model_run_id: str, *, reason: str) -> RegistryEntry:
        """Atomically repoint a market to a previously registered immutable bundle."""
        entries = self._load()
        target = next(
            (
                row
                for row in entries
                if row.market_id == market_id and row.model_run_id == model_run_id
            ),
            None,
        )
        if target is None:
            raise KeyError(model_run_id)
        if target.status == "quarantined":
            raise ValueError("A quarantined model cannot be restored")
        now = datetime.now(UTC).isoformat()
        updated = []
        for row in entries:
            if row.market_id == market_id and row.status == "champion":
                row = replace(row, status="retired", reason=f"rollback to {model_run_id}")
            if row.model_run_id == model_run_id:
                row = replace(row, status="champion", promoted_at=now, reason=reason)
            updated.append(row)
        self._save(updated)
        return self.champion(market_id)
