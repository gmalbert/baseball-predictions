"""Canonical Arrow schemas for typed Parquet boundaries."""

from __future__ import annotations

import pyarrow as pa

UTC = pa.timestamp("us", tz="UTC")
DECIMAL_PRICE = pa.decimal128(12, 6)
DECIMAL_POINT = pa.decimal128(8, 3)
DECIMAL_MONEY = pa.decimal128(14, 2)

RAW_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string()),
        pa.field("event_time", UTC),
        pa.field("source_updated_at", UTC),
        pa.field("observed_at", UTC, nullable=False),
        pa.field("ingested_at", UTC, nullable=False),
        pa.field("request_params_json", pa.string(), nullable=False),
        pa.field("http_metadata_json", pa.string(), nullable=False),
        pa.field("payload_sha256", pa.string(), nullable=False),
        pa.field("payload_uri", pa.string(), nullable=False),
        pa.field("ingestion_run_id", pa.string(), nullable=False),
    ]
)

GAME_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("season", pa.int32(), nullable=False),
        pa.field("game_type", pa.string(), nullable=False),
        pa.field("scheduled_start_utc", UTC, nullable=False),
        pa.field("venue_id", pa.string(), nullable=False),
        pa.field("home_team_id", pa.string(), nullable=False),
        pa.field("away_team_id", pa.string(), nullable=False),
        pa.field("doubleheader_number", pa.int8()),
        pa.field("ruleset_version", pa.string(), nullable=False),
        pa.field("created_at", UTC, nullable=False),
    ]
)

GAME_RESULT_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("home_runs", pa.int16(), nullable=False),
        pa.field("away_runs", pa.int16(), nullable=False),
        pa.field("innings_played", pa.int16(), nullable=False),
        pa.field("completed_at", UTC, nullable=False),
        pa.field("official_at", UTC, nullable=False),
        pa.field("result_version", pa.int32(), nullable=False),
        pa.field("source_updated_at", UTC),
    ]
)

ODDS_QUOTE_SCHEMA = pa.schema(
    [
        pa.field("quote_id", pa.string(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("bookmaker_id", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("selection", pa.string(), nullable=False),
        pa.field("participant_id", pa.string()),
        pa.field("point", DECIMAL_POINT),
        pa.field("price_decimal", DECIMAL_PRICE, nullable=False),
        pa.field("price_american", pa.int32()),
        pa.field("observed_at", UTC, nullable=False),
        pa.field("source_updated_at", UTC),
        pa.field("is_live", pa.bool_(), nullable=False),
        pa.field("is_suspended", pa.bool_(), nullable=False),
        pa.field("is_actionable", pa.bool_(), nullable=False),
        pa.field("action_rule", pa.string()),
        pa.field("raw_payload_hash", pa.string()),
        pa.field("ingestion_run_id", pa.string()),
    ]
)

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("prediction_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("model_run_id", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("selection", pa.string(), nullable=False),
        pa.field("probability_raw", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
        pa.field("probability_low", pa.float64()),
        pa.field("probability_high", pa.float64()),
        pa.field("predicted_at", UTC, nullable=False),
        pa.field("feature_row_hash", pa.string(), nullable=False),
        pa.field("calibration_version", pa.string()),
    ]
)

GAME_SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("as_of_time", UTC, nullable=False),
        pa.field("snapshot_type", pa.string(), nullable=False),
        pa.field("feature_set_version", pa.string(), nullable=False),
        pa.field("features_json", pa.string(), nullable=False),
        pa.field("source_watermarks_json", pa.string(), nullable=False),
        pa.field("row_hash", pa.string(), nullable=False),
        pa.field("build_run_id", pa.string()),
        pa.field("quality_status", pa.string(), nullable=False),
    ]
)

ELIGIBILITY_SCHEMA = pa.schema(
    [
        pa.field("eligibility_id", pa.string(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("selection", pa.string(), nullable=False),
        pa.field("as_of_time", UTC, nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("quote_id", pa.string()),
        pa.field("reason_codes", pa.list_(pa.string()), nullable=False),
        pa.field("quality_status", pa.string(), nullable=False),
    ]
)

DECISION_SCHEMA = pa.schema(
    [
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("prediction_id", pa.string(), nullable=False),
        pa.field("quote_id", pa.string(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("market_id", pa.string(), nullable=False),
        pa.field("selection", pa.string(), nullable=False),
        pa.field("decided_at", UTC, nullable=False),
        pa.field("fair_probability", pa.float64(), nullable=False),
        pa.field("market_probability", pa.float64(), nullable=False),
        pa.field("edge", pa.float64(), nullable=False),
        pa.field("expected_value", pa.float64(), nullable=False),
        pa.field("recommended_stake", DECIMAL_MONEY, nullable=False),
        pa.field("policy_version", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
    ]
)

EXECUTION_SCHEMA = pa.schema(
    [
        pa.field("execution_id", pa.string(), nullable=False),
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("quote_id", pa.string(), nullable=False),
        pa.field("bookmaker_id", pa.string(), nullable=False),
        pa.field("placed_at", UTC, nullable=False),
        pa.field("accepted_price_decimal", DECIMAL_PRICE, nullable=False),
        pa.field("accepted_point", DECIMAL_POINT),
        pa.field("stake", DECIMAL_MONEY, nullable=False),
        pa.field("external_bet_id", pa.string()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("rejection_reason", pa.string()),
    ]
)

SETTLEMENT_SCHEMA = pa.schema(
    [
        pa.field("execution_id", pa.string(), nullable=False),
        pa.field("settled_at", UTC, nullable=False),
        pa.field("result", pa.string(), nullable=False),
        pa.field("profit_loss", DECIMAL_MONEY, nullable=False),
        pa.field("settlement_rule", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_reference", pa.string()),
        pa.field("settlement_version", pa.int32(), nullable=False),
    ]
)

CLOSING_QUOTE_SCHEMA = pa.schema(
    [
        pa.field("execution_id", pa.string(), nullable=False),
        pa.field("quote_id", pa.string(), nullable=False),
        pa.field("close_definition", pa.string(), nullable=False),
        pa.field("closing_probability_no_vig", pa.float64()),
        pa.field("clv_probability", pa.float64()),
        pa.field("clv_log_price", pa.float64()),
    ]
)


def schema_fingerprint(schema: pa.Schema) -> str:
    import hashlib

    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def enforce_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    if set(table.column_names) != set(schema.names):
        missing = sorted(set(schema.names) - set(table.column_names))
        extra = sorted(set(table.column_names) - set(schema.names))
        raise ValueError(f"Arrow schema columns differ: missing={missing}, extra={extra}")
    return table.select(schema.names).cast(schema, safe=True)
