"""Append-only raw payload persistence with retrieval metadata and checksums."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.contracts.domain import RawObservation, stable_id
from src.ingestion.base import RetrievedPayload


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def persist(self, payload: RetrievedPayload, *, ingestion_run_id: str) -> RawObservation:
        if payload.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        digest = sha256(payload.body).hexdigest()
        observed_utc = payload.observed_at.astimezone(UTC)
        relative = Path(
            payload.source,
            f"observed_date={observed_utc.date().isoformat()}",
            digest[:2],
            f"{digest}.payload",
        )
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload.body:
            raise RuntimeError("SHA-256 collision or raw-store corruption")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + f".{ingestion_run_id}.tmp")
            temporary.write_bytes(payload.body)
            os.replace(temporary, target)

        ingested_at = datetime.now(UTC)
        observation = RawObservation(
            observation_id=stable_id(
                "observation",
                payload.source,
                payload.source_record_id or "",
                digest,
                observed_utc.isoformat(),
            ),
            source=payload.source,
            source_record_id=payload.source_record_id,
            event_time=payload.event_time,
            source_updated_at=payload.source_updated_at,
            observed_at=payload.observed_at,
            ingested_at=ingested_at,
            request_params=payload.request_params,
            http_metadata=payload.http_metadata,
            payload_sha256=digest,
            payload_uri=relative.as_posix(),
            ingestion_run_id=ingestion_run_id,
        )
        sidecar = target.with_suffix(target.suffix + ".json")
        if not sidecar.exists():
            temporary_sidecar = sidecar.with_suffix(sidecar.suffix + f".{ingestion_run_id}.tmp")
            temporary_sidecar.write_text(
                observation.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_sidecar, sidecar)
        return observation

    def load(self, observation: RawObservation) -> bytes:
        target = (self.root / observation.payload_uri).resolve()
        if self.root not in target.parents:
            raise ValueError("Raw payload URI escapes store root")
        body = target.read_bytes()
        if sha256(body).hexdigest() != observation.payload_sha256:
            raise ValueError("Raw payload checksum mismatch")
        return body

    def manifest(self, observations: list[RawObservation]) -> dict[str, Any]:
        return {
            "count": len(observations),
            "observations": [row.model_dump(mode="json") for row in observations],
        }
