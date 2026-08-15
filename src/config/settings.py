"""Validated startup configuration; semantic changes require version changes."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SnapshotConfig(FrozenConfig):
    name: str
    local_time: str | None = None
    offset_from_start_minutes: int | None = None


class QuoteConfig(FrozenConfig):
    max_age_seconds: int = Field(gt=0)
    min_books_for_consensus: int = Field(ge=2)


class QualityConfig(FrozenConfig):
    fail_on_unknown_team: bool = True
    max_missing_critical_fraction: float = Field(ge=0, le=1)


class BankrollConfig(FrozenConfig):
    kelly_fraction: float = Field(gt=0, le=1)
    max_bet_fraction: float = Field(gt=0, le=1)
    max_game_fraction: float = Field(gt=0, le=1)
    max_team_fraction: float = Field(gt=0, le=1)
    max_market_fraction: float = Field(gt=0, le=1)
    max_factor_fraction: float = Field(gt=0, le=1)
    max_day_fraction: float = Field(gt=0, le=1)
    stop_loss_fraction: float = Field(gt=0, le=1)
    max_open_bets: int = Field(gt=0)


class AppConfig(FrozenConfig):
    environment: str
    timezone: str
    data_schema_version: str
    feature_set_version: str
    decision_policy_version: str
    snapshots: tuple[SnapshotConfig, ...]
    quotes: QuoteConfig
    models: dict[str, str]
    quality: QualityConfig
    bankroll: BankrollConfig


def load_config(path: Path) -> AppConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(payload)
