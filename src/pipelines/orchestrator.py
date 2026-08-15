"""Idempotent stage orchestration, run IDs, structured events, and atomic completion."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.contracts.domain import stable_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    target_date: date
    as_of_time: datetime
    code_commit: str
    environment_lock_hash: str
    config: dict[str, Any]
    workspace: Path

    @classmethod
    def create(
        cls,
        *,
        target_date: date,
        as_of_time: datetime,
        code_commit: str,
        environment_lock_hash: str,
        config: dict[str, Any],
        workspace: Path,
    ) -> RunContext:
        if as_of_time.tzinfo is None:
            raise ValueError("as_of_time must be timezone-aware")
        canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
        run_id = stable_id(
            "run",
            target_date.isoformat(),
            as_of_time.astimezone(UTC).isoformat(),
            code_commit,
            environment_lock_hash,
            canonical_config,
        )
        return cls(
            run_id, target_date, as_of_time, code_commit, environment_lock_hash, config, workspace
        )


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    rows_in: int
    rows_out: int
    duration_ms: int
    output_uri: str | None = None
    output_hash: str | None = None
    quality_status: str = "passed"
    error_type: str | None = None


def structured_event(
    context: RunContext, stage: str, level: str = "INFO", **fields: Any
) -> dict[str, Any]:
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "run_id": context.run_id,
        "stage": stage,
        "target_date": context.target_date.isoformat(),
        "as_of_time": context.as_of_time.isoformat(),
        **fields,
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(event, sort_keys=True))
    return event


class Orchestrator:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.run_dir = context.workspace / "runs" / context.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(
        self,
        name: str,
        function: Callable[[], tuple[int, int, Path | None]],
        *,
        input_hash: str,
    ) -> StageResult:
        state_path = self.run_dir / f"{name}.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("input_hash") == input_hash
                and state.get("result", {}).get("status") == "completed"
            ):
                return StageResult(**state["result"])
        started = time.perf_counter()
        structured_event(
            self.context, name, input_hash=input_hash, retry_count=int(state_path.exists())
        )
        try:
            rows_in, rows_out, output = function()
            output_hash = _hash_path(output) if output else None
            result = StageResult(
                stage=name,
                status="completed",
                rows_in=rows_in,
                rows_out=rows_out,
                duration_ms=round((time.perf_counter() - started) * 1000),
                output_uri=str(output) if output else None,
                output_hash=output_hash,
            )
        except Exception as exc:
            result = StageResult(
                stage=name,
                status="failed",
                rows_in=0,
                rows_out=0,
                duration_ms=round((time.perf_counter() - started) * 1000),
                quality_status="failed",
                error_type=type(exc).__name__,
            )
            _atomic_json(state_path, {"input_hash": input_hash, "result": asdict(result)})
            structured_event(self.context, name, level="ERROR", error_type=type(exc).__name__)
            raise
        _atomic_json(state_path, {"input_hash": input_hash, "result": asdict(result)})
        structured_event(
            self.context,
            name,
            rows_in=result.rows_in,
            rows_out=result.rows_out,
            duration_ms=result.duration_ms,
            quality_status=result.quality_status,
        )
        return result


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
