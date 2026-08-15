from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.exports import EXPORT_COLUMNS, validate_export, write_exports
from src.contracts.domain import GameSnapshot, Quote, Selection, stable_id
from src.models.manifest import quarantine_reason
from src.quality.contracts import load_contract, validate_contract

NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


def test_snapshot_rejects_future_watermark_and_quote_requires_aware_time():
    with pytest.raises(ValidationError, match="future observations"):
        GameSnapshot(
            snapshot_id="s",
            game_id="g",
            as_of_time=NOW,
            snapshot_type="morning",
            feature_set_version="v",
            features={},
            source_watermarks={"lineup": datetime(2026, 8, 10, 16, 0, 1, tzinfo=UTC)},
            row_hash="h",
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        Quote(
            quote_id="q",
            game_id="g",
            bookmaker_id="b",
            market_id="m",
            selection=Selection.HOME,
            price_decimal=Decimal("2"),
            observed_at=datetime(2026, 8, 10, 16),
        )


def test_stable_ids_and_missing_model_manifests_are_fail_closed(tmp_path):
    assert stable_id("game", "A", 1) == stable_id("game", " a ", 1)
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"opaque")
    assert quarantine_reason(artifact, artifact.with_suffix(".manifest.json")) == "missing_manifest"


def test_canonical_export_requires_book_price_date_snapshot_and_version(tmp_path):
    row = {column: "value" for column in EXPORT_COLUMNS}
    row.update(
        {
            "point": 1.5,
            "price_decimal": 2.0,
            "price_american": 100,
            "probability": 0.6,
            "probability_low": 0.55,
            "probability_high": 0.65,
            "break_even_probability": 0.5,
            "edge": 0.1,
            "expected_value": 0.2,
            "recommended_stake": 10.0,
        }
    )
    assert list(validate_export(pd.DataFrame([row]))) == EXPORT_COLUMNS
    json_path = tmp_path / "opportunities.json"
    csv_path = tmp_path / "opportunities.csv"
    write_exports(pd.DataFrame([row]), json_path=json_path, csv_path=csv_path)
    assert json_path.is_file() and csv_path.is_file()
    assert pd.read_csv(csv_path).loc[0, "prediction_id"] == row["prediction_id"]
    row["bookmaker_id"] = None
    with pytest.raises(ValueError, match="null required"):
        validate_export(pd.DataFrame([row]))


def test_odds_yaml_contract_checks_duplicate_keys():
    contract = load_contract(__import__("pathlib").Path("config/contracts/fact_odds_quote.yaml"))
    frame = pd.DataFrame(
        [
            {
                "quote_id": "q",
                "game_id": "g",
                "bookmaker_id": "b",
                "market_id": "m",
                "selection": "home",
                "point": None,
                "price_decimal": 2.0,
                "observed_at": pd.Timestamp(NOW),
                "ingested_at": pd.Timestamp(NOW),
            }
        ]
    )
    assert validate_contract(frame, contract)
