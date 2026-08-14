"""Machine-readable health state used to block stale recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SourceHealth:
    source: str
    observed_at: datetime | None
    age_seconds: float | None
    completeness: float
    disagreement: float
    recent_failure_rate: float
    trust_score: float
    status: str


def source_health(
    *,
    source: str,
    observed_at: datetime | None,
    as_of: datetime,
    completeness: float,
    disagreement: float,
    recent_failure_rate: float,
    warn_after_seconds: int,
    fail_after_seconds: int,
) -> SourceHealth:
    age = (as_of - observed_at).total_seconds() if observed_at else None
    freshness = 0.0 if age is None else max(0.0, 1.0 - age / fail_after_seconds)
    trust = max(
        0.0,
        min(
            1.0,
            0.4 * freshness
            + 0.3 * completeness
            + 0.15 * (1 - disagreement)
            + 0.15 * (1 - recent_failure_rate),
        ),
    )
    status = (
        "failed"
        if age is None or age > fail_after_seconds
        else "warning"
        if age > warn_after_seconds
        else "healthy"
    )
    return SourceHealth(
        source, observed_at, age, completeness, disagreement, recent_failure_rate, trust, status
    )


@dataclass(frozen=True)
class PlatformHealth:
    status: str
    recommendation_blocked: bool
    reasons: tuple[str, ...]
    sources: tuple[SourceHealth, ...]
    model_compatible: bool
    quality_passed: bool


def platform_health(
    sources: list[SourceHealth],
    *,
    model_compatible: bool,
    quality_passed: bool,
) -> PlatformHealth:
    reasons = [f"source:{row.source}:{row.status}" for row in sources if row.status == "failed"]
    if not model_compatible:
        reasons.append("model_incompatible")
    if not quality_passed:
        reasons.append("quality_failed")
    blocked = bool(reasons)
    warning = any(row.status == "warning" for row in sources)
    return PlatformHealth(
        status="blocked" if blocked else "warning" if warning else "healthy",
        recommendation_blocked=blocked,
        reasons=tuple(reasons),
        sources=tuple(sources),
        model_compatible=model_compatible,
        quality_passed=quality_passed,
    )
