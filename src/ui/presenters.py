"""Convert canonical records into stable, testable page rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.contracts.domain import Decision, Prediction, Quote
from src.monitoring.health import PlatformHealth
from src.ui.components import freshness_label


def operations_summary(
    *,
    target_date: str,
    run: dict[str, Any] | None,
    health: PlatformHealth | None,
    games: int,
    predictions: int,
    eligible_bets: int,
    incidents: int,
) -> dict[str, Any]:
    if run is None:
        state, message = "pipeline_pending", "No completed pipeline run is available."
    elif health and health.recommendation_blocked:
        state, message = (
            "blocked",
            "Recommendations are blocked because data, quality, or model compatibility failed.",
        )
    elif predictions == 0 and games > 0:
        state, message = "no_opportunities", "The pipeline succeeded; no selection met the policy."
    else:
        state, message = "healthy", "The latest canonical run passed its gates."
    return {
        "target_date": target_date,
        "state": state,
        "message": message,
        "last_run_id": run.get("run_id") if run else None,
        "stage": run.get("stage") if run else None,
        "games": games,
        "predictions": predictions,
        "eligible_bets": eligible_bets,
        "incidents": incidents,
        "health": health.status if health else "unknown",
    }


def opportunity_rows(
    predictions: list[Prediction],
    quotes: list[Quote],
    decisions: list[Decision],
    *,
    as_of: datetime,
) -> pd.DataFrame:
    quote_by_id = {row.quote_id: row for row in quotes}
    prediction_by_id = {row.prediction_id: row for row in predictions}
    rows = []
    for decision in decisions:
        prediction = prediction_by_id[decision.prediction_id]
        quote = quote_by_id[decision.quote_id]
        age = (as_of - quote.observed_at).total_seconds()
        freshness, _ = freshness_label(age)
        rows.append(
            {
                "game_id": decision.game_id,
                "market": decision.market_id,
                "selection": decision.selection.value,
                "book": quote.bookmaker_id,
                "line": float(quote.point) if quote.point is not None else None,
                "price_decimal": float(quote.price_decimal),
                "price_american": quote.price_american,
                "quote_time": quote.observed_at.isoformat(),
                "quote_age_seconds": age,
                "freshness": freshness,
                "fair_probability": decision.fair_probability,
                "probability_low": prediction.probability_low,
                "probability_high": prediction.probability_high,
                "break_even_probability": decision.break_even_probability,
                "edge": decision.edge,
                "expected_value": decision.expected_value,
                "recommended_stake": float(decision.recommended_stake),
                "action": decision.action,
                "reason_codes": ", ".join(decision.reason_codes),
                "model_run_id": prediction.model_run_id,
                "snapshot_id": prediction.snapshot_id,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["risk_adjusted_ev"] = frame["expected_value"] * frame["fair_probability"]
        frame = frame.sort_values("risk_adjusted_ev", ascending=False)
    return frame


def health_rows(health: PlatformHealth | None) -> pd.DataFrame:
    if health is None:
        return pd.DataFrame(columns=["source", "status", "age_seconds", "trust_score"])
    return pd.DataFrame(
        [
            {
                "source": row.source,
                "status": row.status,
                "age_seconds": row.age_seconds,
                "completeness": row.completeness,
                "disagreement": row.disagreement,
                "failure_rate": row.recent_failure_rate,
                "trust_score": row.trust_score,
            }
            for row in health.sources
        ]
    )
