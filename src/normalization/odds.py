"""Normalize provider quote rows without losing book, line, status, or time."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.contracts.domain import Quote, stable_id
from src.markets.pricing import american_to_decimal
from src.normalization.markets import canonical_market_id, canonical_selection


def normalize_quote(
    row: dict[str, Any],
    *,
    observed_at: datetime,
    raw_payload_hash: str,
    ingestion_run_id: str,
) -> Quote:
    selection = canonical_selection(
        str(row["outcome_name"]),
        home_team=row.get("home_team"),
        away_team=row.get("away_team"),
    )
    market_id = canonical_market_id(str(row["market"]))
    american = int(row["outcome_price"])
    point = row.get("outcome_point")
    book = str(row["bookmaker"])
    source_updated_at = row.get("source_updated_at")
    if isinstance(source_updated_at, str):
        source_updated_at = datetime.fromisoformat(source_updated_at.replace("Z", "+00:00"))
    quote_id = stable_id(
        "quote",
        row["game_id"],
        book,
        market_id,
        selection.value,
        point,
        observed_at.isoformat(),
        american,
    )
    return Quote(
        quote_id=quote_id,
        game_id=str(row["game_id"]),
        bookmaker_id=book,
        market_id=market_id,
        selection=selection,
        participant_id=row.get("participant_id"),
        point=Decimal(str(point)) if point is not None else None,
        price_decimal=Decimal(str(american_to_decimal(american))),
        price_american=american,
        observed_at=observed_at,
        source_updated_at=source_updated_at,
        is_live=bool(row.get("is_live", False)),
        is_suspended=bool(row.get("is_suspended", False)),
        is_actionable=bool(row.get("is_actionable", True)),
        jurisdiction=row.get("jurisdiction"),
        limit_amount=Decimal(str(row["limit_amount"]))
        if row.get("limit_amount") is not None
        else None,
        action_rule=row.get("action_rule"),
        raw_payload_hash=raw_payload_hash,
        ingestion_run_id=ingestion_run_id,
    )
