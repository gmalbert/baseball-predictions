"""Immutable records shared by ingestion, modeling, replay, API, and UI.

Every time-varying record uses timezone-aware UTC timestamps.  Persisted records
forbid undeclared fields so a provider schema change cannot silently change the
economic meaning of a prediction or wager.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class Selection(StrEnum):
    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"
    YES = "yes"
    NO = "no"


class SettlementResult(StrEnum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class RawObservation(FrozenModel):
    observation_id: str
    source: str
    source_record_id: str | None = None
    event_time: datetime | None = None
    source_updated_at: datetime | None = None
    observed_at: datetime
    ingested_at: datetime
    request_params: dict[str, Any] = Field(default_factory=dict)
    http_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_uri: str
    ingestion_run_id: str

    @model_validator(mode="after")
    def validate_times(self) -> RawObservation:
        _aware(self.observed_at, "observed_at")
        _aware(self.ingested_at, "ingested_at")
        if self.event_time is not None:
            _aware(self.event_time, "event_time")
        if self.source_updated_at is not None:
            _aware(self.source_updated_at, "source_updated_at")
        if self.observed_at > self.ingested_at:
            raise ValueError("observed_at cannot be later than ingested_at")
        return self


class Game(FrozenModel):
    game_id: str
    season: int = Field(ge=1876)
    game_type: str
    scheduled_start_utc: datetime
    venue_id: str
    home_team_id: str
    away_team_id: str
    doubleheader_number: int | None = Field(default=None, ge=1)
    ruleset_version: str
    created_at: datetime

    @field_validator("scheduled_start_utc", "created_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        return _aware(value, "game timestamp")


class GameResult(FrozenModel):
    game_id: str
    home_runs: int = Field(ge=0)
    away_runs: int = Field(ge=0)
    innings_played: int = Field(ge=0)
    completed_at: datetime
    official_at: datetime
    result_version: int = Field(ge=1)
    source_updated_at: datetime | None = None

    @field_validator("completed_at", "official_at")
    @classmethod
    def required_times_are_aware(cls, value: datetime) -> datetime:
        return _aware(value, "result timestamp")


class Quote(FrozenModel):
    quote_id: str
    game_id: str
    bookmaker_id: str
    market_id: str
    selection: Selection
    participant_id: str | None = None
    point: Decimal | None = None
    price_decimal: Decimal = Field(gt=Decimal("1"))
    price_american: int | None = None
    observed_at: datetime
    ingested_at: datetime | None = None
    source_updated_at: datetime | None = None
    is_live: bool = False
    is_suspended: bool = False
    is_actionable: bool = True
    jurisdiction: str | None = None
    limit_amount: Decimal | None = Field(default=None, ge=Decimal("0"))
    action_rule: str | None = None
    raw_payload_hash: str | None = None
    ingestion_run_id: str | None = None

    @model_validator(mode="after")
    def validate_quote(self) -> Quote:
        _aware(self.observed_at, "observed_at")
        if self.ingested_at is not None:
            _aware(self.ingested_at, "ingested_at")
        if self.source_updated_at is not None:
            _aware(self.source_updated_at, "source_updated_at")
        if self.price_american == 0:
            raise ValueError("American price cannot be zero")
        return self


class GameSnapshot(FrozenModel):
    snapshot_id: str
    game_id: str
    as_of_time: datetime
    snapshot_type: str
    feature_set_version: str
    features: dict[str, float | int | str | bool | None]
    source_watermarks: dict[str, datetime]
    row_hash: str
    build_run_id: str | None = None
    quality_status: str = "passed"

    @model_validator(mode="after")
    def no_future_watermarks(self) -> GameSnapshot:
        _aware(self.as_of_time, "as_of_time")
        future = {
            source: timestamp
            for source, timestamp in self.source_watermarks.items()
            if _aware(timestamp, f"source_watermarks[{source}]") > self.as_of_time
        }
        if future:
            raise ValueError(f"future observations in snapshot: {future}")
        return self


class Prediction(FrozenModel):
    prediction_id: str
    snapshot_id: str
    game_id: str
    model_run_id: str
    market_id: str
    selection: Selection
    probability_raw: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)
    probability_low: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_high: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_at: datetime
    feature_row_hash: str
    calibration_version: str | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> Prediction:
        _aware(self.predicted_at, "predicted_at")
        if self.probability_low is not None and self.probability_low > self.probability:
            raise ValueError("probability_low cannot exceed probability")
        if self.probability_high is not None and self.probability_high < self.probability:
            raise ValueError("probability_high cannot be below probability")
        return self


class EligibilityRecord(FrozenModel):
    eligibility_id: str
    game_id: str
    market_id: str
    selection: Selection
    as_of_time: datetime
    eligible: bool
    quote_id: str | None = None
    reason_codes: tuple[str, ...] = ()
    quality_status: str

    @field_validator("as_of_time")
    @classmethod
    def cutoff_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, "as_of_time")


class Decision(FrozenModel):
    decision_id: str
    prediction_id: str
    quote_id: str
    game_id: str
    market_id: str
    selection: Selection
    decided_at: datetime
    market_probability: float = Field(ge=0.0, le=1.0)
    fair_probability: float = Field(ge=0.0, le=1.0)
    break_even_probability: float = Field(ge=0.0, le=1.0)
    edge: float
    expected_value: float
    recommended_stake: Decimal = Field(ge=Decimal("0"))
    bankroll_before: Decimal = Field(ge=Decimal("0"))
    policy_version: str
    action: str
    reason_codes: tuple[str, ...] = ()

    @field_validator("decided_at")
    @classmethod
    def decided_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, "decided_at")


class Execution(FrozenModel):
    execution_id: str
    decision_id: str
    quote_id: str
    bookmaker_id: str
    placed_at: datetime
    accepted_price_decimal: Decimal = Field(gt=Decimal("1"))
    accepted_point: Decimal | None = None
    stake: Decimal = Field(gt=Decimal("0"))
    external_bet_id: str | None = None
    status: str
    rejection_reason: str | None = None

    @field_validator("placed_at")
    @classmethod
    def placed_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, "placed_at")


class Settlement(FrozenModel):
    execution_id: str
    settled_at: datetime
    result: SettlementResult
    profit_loss: Decimal
    settlement_rule: str
    source: str
    source_reference: str | None = None
    settlement_version: int = Field(ge=1)

    @field_validator("settled_at")
    @classmethod
    def settled_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, "settled_at")


class ClosingQuote(FrozenModel):
    execution_id: str
    quote_id: str
    close_definition: str
    closing_probability_no_vig: float | None = Field(default=None, ge=0.0, le=1.0)
    clv_probability: float | None = None
    clv_log_price: float | None = None


def stable_id(namespace: str, *parts: Any) -> str:
    """Return a deterministic, display-safe identifier for a canonical grain."""
    normalized = "|".join(str(part).strip().lower() for part in parts)
    return f"{namespace}_{sha256(normalized.encode()).hexdigest()[:24]}"


def stable_game_id(
    *,
    season: int,
    scheduled_start_utc: datetime,
    away_team_id: str,
    home_team_id: str,
    doubleheader_number: int | None,
    mlb_game_pk: int | None = None,
) -> str:
    if mlb_game_pk is not None:
        return stable_id("game", "mlb", mlb_game_pk)
    return stable_id(
        "game",
        season,
        scheduled_start_utc.astimezone(UTC).isoformat(),
        away_team_id,
        home_team_id,
        doubleheader_number or 1,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def target_date_of(game: Game) -> date:
    return game.scheduled_start_utc.date()
