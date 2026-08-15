import pyarrow as pa
import pytest

from src.contracts import schemas


def test_canonical_arrow_schemas_have_stable_unique_fingerprints() -> None:
    canonical = [
        schemas.RAW_OBSERVATION_SCHEMA,
        schemas.GAME_SCHEMA,
        schemas.GAME_RESULT_SCHEMA,
        schemas.ODDS_QUOTE_SCHEMA,
        schemas.GAME_SNAPSHOT_SCHEMA,
        schemas.PREDICTION_SCHEMA,
        schemas.ELIGIBILITY_SCHEMA,
        schemas.DECISION_SCHEMA,
        schemas.EXECUTION_SCHEMA,
        schemas.SETTLEMENT_SCHEMA,
        schemas.CLOSING_QUOTE_SCHEMA,
    ]
    fingerprints = [schemas.schema_fingerprint(schema) for schema in canonical]
    assert len(set(fingerprints)) == len(fingerprints)


def test_arrow_boundary_rejects_extra_columns() -> None:
    table = pa.table({"game_id": ["g"], "unexpected": [1]})
    with pytest.raises(ValueError, match="schema columns differ"):
        schemas.enforce_schema(table, schemas.GAME_SCHEMA)
