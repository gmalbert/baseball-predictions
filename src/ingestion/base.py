"""Source adapter protocol and retrieval envelope for deterministic ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievedPayload:
    source: str
    body: bytes
    observed_at: datetime
    request_params: dict[str, Any] = field(default_factory=dict)
    http_metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    source_record_id: str | None = None
    event_time: datetime | None = None
    source_updated_at: datetime | None = None


class SourceAdapter(Protocol):
    source_name: str

    def retrieve(self, *, target_date: date, as_of: datetime) -> list[RetrievedPayload]: ...

    def normalize(self, payloads: list[RetrievedPayload]) -> list[dict[str, Any]]: ...


def validate_replay_cutoff(payloads: list[RetrievedPayload], *, as_of: datetime) -> None:
    late = [row for row in payloads if row.observed_at > as_of]
    if late:
        raise ValueError(f"Replay source returned {len(late)} observations after cutoff")
