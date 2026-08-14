"""Sample eligibility ledger exposes every inclusion and exclusion reason."""

from __future__ import annotations

from datetime import datetime

from src.contracts.domain import EligibilityRecord, Selection, stable_id


def eligibility_record(
    *,
    game_id: str,
    market_id: str,
    selection: Selection,
    as_of: datetime,
    quote_id: str | None,
    quality_passed: bool,
    game_status: str,
    pitcher_confirmed: bool,
    lineup_required: bool,
    lineup_confirmed: bool,
    label_available: bool,
) -> EligibilityRecord:
    reasons = []
    if quote_id is None:
        reasons.append("missing_quote")
    if not quality_passed:
        reasons.append("quality_failure")
    if game_status.lower() in {"postponed", "cancelled", "suspended"}:
        reasons.append(game_status.lower())
    if not pitcher_confirmed:
        reasons.append("pitcher_unconfirmed")
    if lineup_required and not lineup_confirmed:
        reasons.append("lineup_unconfirmed")
    if not label_available:
        reasons.append("label_unavailable")
    return EligibilityRecord(
        eligibility_id=stable_id("eligibility", game_id, market_id, selection, as_of),
        game_id=game_id,
        market_id=market_id,
        selection=selection,
        as_of_time=as_of,
        eligible=not reasons,
        quote_id=quote_id,
        reason_codes=tuple(reasons),
        quality_status="passed" if quality_passed else "failed",
    )
