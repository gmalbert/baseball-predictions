"""Deterministic historical replay using archived observations only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from src.backtesting.quotes import match_archived_quote
from src.contracts.domain import (
    Decision,
    EligibilityRecord,
    GameSnapshot,
    Prediction,
    Quote,
    stable_id,
)
from src.decisions.policy import Policy, evaluate


class ReplayCatalog(Protocol):
    def build_snapshots(self, target_date: date, as_of: datetime) -> list[GameSnapshot]: ...
    def quotes(self, game_ids: list[str], as_of: datetime) -> list[Quote]: ...
    def save_predictions(self, rows: list[Prediction]) -> None: ...
    def save_eligibility(self, rows: list[EligibilityRecord]) -> None: ...
    def save_decisions(self, rows: list[Decision]) -> None: ...


class Predictor(Protocol):
    def predict(
        self, snapshots: list[GameSnapshot], predicted_at: datetime
    ) -> list[Prediction]: ...


@dataclass(frozen=True)
class ReplayResult:
    snapshots: int
    predictions: int
    quotes: int
    matched_quotes: int
    bets: int
    abstentions: int
    no_quote: int


class FixtureCatalog:
    """Read-only archived fixture adapter used by CLI and deterministic CI."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._snapshots = [
            GameSnapshot.model_validate(row)
            for row in payload.get("snapshots", [])  # type: ignore[union-attr]
        ]
        self._quotes = [
            Quote.model_validate(row)
            for row in payload.get("quotes", [])  # type: ignore[union-attr]
        ]
        self.predictions: list[Prediction] = []
        self.eligibility: list[EligibilityRecord] = []
        self.decisions: list[Decision] = []

    def build_snapshots(self, target_date: date, as_of: datetime) -> list[GameSnapshot]:
        return [row for row in self._snapshots if row.as_of_time == as_of]

    def quotes(self, game_ids: list[str], as_of: datetime) -> list[Quote]:
        return [row for row in self._quotes if row.game_id in game_ids and row.observed_at <= as_of]

    def save_predictions(self, rows: list[Prediction]) -> None:
        self.predictions = rows

    def save_eligibility(self, rows: list[EligibilityRecord]) -> None:
        self.eligibility = rows

    def save_decisions(self, rows: list[Decision]) -> None:
        self.decisions = rows


class FixturePredictor:
    def __init__(self, payload: dict[str, object]) -> None:
        self._predictions = [
            Prediction.model_validate(row)
            for row in payload.get("predictions", [])  # type: ignore[union-attr]
        ]

    def predict(self, snapshots: list[GameSnapshot], predicted_at: datetime) -> list[Prediction]:
        snapshot_ids = {row.snapshot_id for row in snapshots}
        return [
            row.model_copy(update={"predicted_at": predicted_at})
            for row in self._predictions
            if row.snapshot_id in snapshot_ids
        ]


def replay_fixture(
    fixture: Path, *, target_date: date, as_of: datetime
) -> tuple[ReplayResult, str]:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Replay fixture root must be an object")
    catalog = FixtureCatalog(payload)
    result = replay(
        target_date=target_date,
        as_of=as_of,
        catalog=catalog,
        predictor=FixturePredictor(payload),
        policy=Policy(),
        bankroll=Decimal(str(payload.get("bankroll", "1000"))),
        jurisdiction=str(payload["jurisdiction"]) if payload.get("jurisdiction") else None,
    )
    evidence = {
        "result": asdict(result),
        "predictions": [row.model_dump(mode="json") for row in catalog.predictions],
        "eligibility": [row.model_dump(mode="json") for row in catalog.eligibility],
        "decisions": [row.model_dump(mode="json") for row in catalog.decisions],
    }
    digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, digest


def replay(
    *,
    target_date: date,
    as_of: datetime,
    catalog: ReplayCatalog,
    predictor: Predictor,
    policy: Policy,
    bankroll: Decimal,
    jurisdiction: str | None = None,
) -> ReplayResult:
    if as_of.tzinfo is None:
        raise ValueError("Replay cutoff must be timezone-aware")
    snapshots = catalog.build_snapshots(target_date, as_of)
    if any(row.as_of_time != as_of for row in snapshots):
        raise ValueError("Catalog returned snapshots from a different cutoff")
    predictions = predictor.predict(snapshots, as_of)
    catalog.save_predictions(predictions)
    quotes = catalog.quotes(sorted({row.game_id for row in predictions}), as_of)

    decisions: list[Decision] = []
    eligibility: list[EligibilityRecord] = []
    matched_quotes = 0
    no_quote = 0
    for prediction in predictions:
        match = match_archived_quote(
            prediction,
            quotes,
            as_of=as_of,
            max_age_seconds=policy.max_quote_age_seconds,
            jurisdiction=jurisdiction,
        )
        eligibility.append(
            EligibilityRecord(
                eligibility_id=stable_id("eligibility", prediction.prediction_id, as_of),
                game_id=prediction.game_id,
                market_id=prediction.market_id,
                selection=prediction.selection,
                as_of_time=as_of,
                eligible=match.eligible,
                quote_id=match.quote.quote_id if match.quote else None,
                reason_codes=() if match.eligible else (match.reason or "ineligible",),
                quality_status="passed" if match.eligible else "ineligible",
            )
        )
        if match.quote is None:
            no_quote += 1
            continue
        matched_quotes += 1
        decisions.append(
            evaluate(
                prediction,
                match.quote,
                bankroll=bankroll,
                decided_at=as_of,
                policy=policy,
            )
        )
    catalog.save_eligibility(eligibility)
    catalog.save_decisions(decisions)
    return ReplayResult(
        snapshots=len(snapshots),
        predictions=len(predictions),
        quotes=len(quotes),
        matched_quotes=matched_quotes,
        bets=sum(row.action == "bet" for row in decisions),
        abstentions=sum(row.action == "abstain" for row in decisions),
        no_quote=no_quote,
    )


def parse_cutoff(target_date: date, value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of requires an explicit timezone")
    if parsed.date() != target_date:
        raise ValueError("--as-of date must equal --target-date")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic archived MLB replay")
    parser.add_argument("--target-date", required=True, type=date.fromisoformat)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--fixture", help="Fixture orchestration is provided by tests/replay adapters"
    )
    args = parser.parse_args()
    cutoff = parse_cutoff(args.target_date, args.as_of)
    if not args.fixture:
        parser.error("--fixture or an application catalog adapter is required")
    fixture = Path(args.fixture)
    if not fixture.is_file():
        parser.error(f"fixture not found: {fixture}")
    result, digest = replay_fixture(fixture, target_date=args.target_date, as_of=cutoff)
    print(
        json.dumps(
            {
                "status": "completed",
                "target_date": args.target_date.isoformat(),
                "as_of": cutoff.astimezone(UTC).isoformat(),
                "fixture": str(fixture),
                "result": asdict(result),
                "output_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
