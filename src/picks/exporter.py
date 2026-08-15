"""Typed selection record for generation, persistence, export, and settlement."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import Field, model_validator

from src.contracts.domain import FrozenModel, Selection


class PickRecord(FrozenModel):
    prediction_id: str
    decision_id: str
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    market_id: str
    selection: Selection
    bookmaker_id: str
    point: Decimal | None = None
    price_decimal: Decimal = Field(gt=Decimal("1"))
    price_american: int | None = None
    quote_time: datetime
    probability: float = Field(ge=0, le=1)
    probability_low: float | None = Field(default=None, ge=0, le=1)
    probability_high: float | None = Field(default=None, ge=0, le=1)
    edge: float
    expected_value: float
    recommended_stake: Decimal = Field(ge=0)
    action: str
    reason_codes: tuple[str, ...]
    snapshot_id: str
    model_run_id: str
    policy_version: str
    quality_status: str

    @model_validator(mode="after")
    def validate_time(self) -> PickRecord:
        if self.quote_time.tzinfo is None:
            raise ValueError("quote_time must be timezone-aware")
        return self
