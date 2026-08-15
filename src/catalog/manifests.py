"""Content-addressed lineage manifests and atomic local publication."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputManifest:
    uri: str
    sha256: str
    max_observed_at: str | None


@dataclass(frozen=True)
class OutputManifest:
    dataset: str
    schema_version: str
    run_id: str
    code_commit: str
    started_at: str
    completed_at: str
    inputs: tuple[InputManifest, ...]
    row_count: int
    output_sha256: str
    quality_status: str
    config: dict[str, Any]
    min_as_of_time: str | None = None
    max_as_of_time: str | None = None


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_atomically(temp_path: Path, final_path: Path, manifest: OutputManifest) -> None:
    if not temp_path.is_file():
        raise FileNotFoundError(temp_path)
    if manifest.row_count <= 0:
        raise ValueError("Refusing to publish an empty dataset")
    if manifest.quality_status != "passed":
        raise ValueError("Refusing to publish a dataset that failed quality gates")
    if hash_file(temp_path) != manifest.output_sha256:
        raise ValueError("Output checksum changed before publication")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = final_path.with_suffix(final_path.suffix + ".manifest.json")
    temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_manifest.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_path, final_path)
    os.replace(temp_manifest, manifest_path)


def iso_now() -> str:
    return datetime.now(UTC).isoformat()
