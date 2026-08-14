"""Persist all quality results, including passes and warnings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.quality.gate import CheckResult


def write_quality_report(
    path: Path,
    *,
    run_id: str,
    dataset: str,
    partition: str,
    results: list[CheckResult],
) -> None:
    payload = {
        "run_id": run_id,
        "dataset": dataset,
        "partition": partition,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "failed"
        if any(not row.passed and row.severity == "error" for row in results)
        else "passed",
        "results": [asdict(row) for row in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
