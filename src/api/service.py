"""Repository-backed read-only application service."""

from __future__ import annotations

from datetime import date
from typing import Protocol


class ReadRepository(Protocol):
    def schedule(self, target_date: date) -> list[dict]: ...
    def snapshots(self, target_date: date) -> list[dict]: ...
    def predictions(self, target_date: date) -> list[dict]: ...
    def decisions(self, target_date: date) -> list[dict]: ...
    def performance(self) -> list[dict]: ...
    def health(self) -> dict: ...


class ReadOnlyService:
    api_version = "v1"

    def __init__(self, repository: ReadRepository) -> None:
        self.repository = repository

    def get(self, resource: str, *, target_date: date | None = None) -> dict:
        if resource in {"schedule", "snapshots", "predictions", "decisions"}:
            if target_date is None:
                raise ValueError(f"{resource} requires target_date")
            rows = getattr(self.repository, resource)(target_date)
        elif resource == "performance":
            rows = self.repository.performance()
        elif resource == "health":
            return {
                "api_version": self.api_version,
                "resource": resource,
                "data": self.repository.health(),
            }
        else:
            raise KeyError(resource)
        return {
            "api_version": self.api_version,
            "resource": resource,
            "target_date": target_date.isoformat() if target_date else None,
            "data": rows,
        }


def create_fastapi_app(service: ReadOnlyService):
    """Create an optional FastAPI app without making FastAPI a core dependency."""
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError("Install the optional API dependencies to serve HTTP") from exc
    app = FastAPI(title="Baseball Predictions Read API", version="1.0.0")

    @app.get("/v1/{resource}")
    def read(resource: str, target_date: date | None = Query(default=None)) -> dict:
        try:
            return service.get(resource, target_date=target_date)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
