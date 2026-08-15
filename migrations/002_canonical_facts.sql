CREATE TABLE IF NOT EXISTS dim_team (
    team_id VARCHAR PRIMARY KEY,
    mlb_team_id INTEGER,
    abbreviation VARCHAR NOT NULL,
    canonical_name VARCHAR NOT NULL,
    league VARCHAR,
    division VARCHAR,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    record_hash VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_player (
    player_id VARCHAR PRIMARY KEY,
    mlb_player_id INTEGER,
    full_name VARCHAR NOT NULL,
    bats VARCHAR,
    throws VARCHAR,
    birth_date DATE,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    record_hash VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_venue (
    venue_id VARCHAR NOT NULL,
    canonical_name VARCHAR NOT NULL,
    timezone VARCHAR NOT NULL,
    altitude_ft DOUBLE,
    latitude DOUBLE,
    longitude DOUBLE,
    roof_type VARCHAR,
    surface VARCHAR,
    field_orientation_deg DOUBLE,
    geometry_json JSON,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    record_hash VARCHAR NOT NULL,
    PRIMARY KEY (venue_id, valid_from)
);

CREATE TABLE IF NOT EXISTS dim_bookmaker (
    bookmaker_id VARCHAR PRIMARY KEY,
    canonical_name VARCHAR NOT NULL,
    jurisdiction VARCHAR,
    source VARCHAR NOT NULL,
    rules_json JSON NOT NULL,
    actionable BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_market (
    market_id VARCHAR PRIMARY KEY,
    innings_scope VARCHAR NOT NULL,
    includes_extra_innings BOOLEAN NOT NULL,
    pitcher_action_rule VARCHAR,
    settlement_source VARCHAR NOT NULL,
    rule_version VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS bridge_provider_entity (
    entity_type VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    provider_entity_id VARCHAR NOT NULL,
    canonical_entity_id VARCHAR NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    PRIMARY KEY (entity_type, provider, provider_entity_id, valid_from)
);

CREATE TABLE IF NOT EXISTS fact_game (
    game_id VARCHAR PRIMARY KEY,
    season INTEGER NOT NULL,
    game_type VARCHAR NOT NULL,
    scheduled_start_utc TIMESTAMPTZ NOT NULL,
    venue_id VARCHAR NOT NULL,
    home_team_id VARCHAR NOT NULL,
    away_team_id VARCHAR NOT NULL,
    doubleheader_number INTEGER,
    ruleset_version VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_game_status_observation (
    game_id VARCHAR NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    status VARCHAR NOT NULL,
    scheduled_start_utc TIMESTAMPTZ,
    source_updated_at TIMESTAMPTZ,
    raw_payload_hash VARCHAR NOT NULL,
    ingestion_run_id VARCHAR NOT NULL,
    PRIMARY KEY (game_id, observed_at)
);

CREATE TABLE IF NOT EXISTS fact_game_result (
    game_id VARCHAR NOT NULL,
    home_runs INTEGER NOT NULL,
    away_runs INTEGER NOT NULL,
    innings_played INTEGER NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    official_at TIMESTAMPTZ NOT NULL,
    result_version INTEGER NOT NULL,
    source_updated_at TIMESTAMPTZ,
    PRIMARY KEY (game_id, result_version)
);

CREATE TABLE IF NOT EXISTS fact_odds_quote (
    quote_id VARCHAR PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    bookmaker_id VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    selection_id VARCHAR NOT NULL,
    participant_id VARCHAR,
    point DECIMAL(8, 3),
    price_decimal DECIMAL(12, 6) NOT NULL,
    price_american INTEGER,
    observed_at TIMESTAMPTZ NOT NULL,
    source_updated_at TIMESTAMPTZ,
    is_live BOOLEAN NOT NULL,
    is_suspended BOOLEAN NOT NULL,
    is_actionable BOOLEAN NOT NULL,
    action_rule VARCHAR,
    raw_payload_hash VARCHAR NOT NULL,
    ingestion_run_id VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_pregame_observation (
    observation_id VARCHAR PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    entity_id VARCHAR,
    observation_type VARCHAR NOT NULL,
    payload_json JSON NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR NOT NULL,
    ingestion_run_id VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_game_snapshot (
    snapshot_id VARCHAR PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    as_of_time TIMESTAMPTZ NOT NULL,
    snapshot_type VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    source_watermark_json JSON NOT NULL,
    row_hash VARCHAR NOT NULL,
    build_run_id VARCHAR NOT NULL,
    quality_status VARCHAR NOT NULL,
    UNIQUE (game_id, as_of_time, snapshot_type, feature_set_version)
);

CREATE TABLE IF NOT EXISTS fact_prediction (
    prediction_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    game_id VARCHAR NOT NULL,
    model_run_id VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    selection_id VARCHAR NOT NULL,
    probability_raw DOUBLE NOT NULL,
    probability_calibrated DOUBLE NOT NULL,
    probability_low DOUBLE,
    probability_high DOUBLE,
    predicted_at TIMESTAMPTZ NOT NULL,
    feature_row_hash VARCHAR NOT NULL,
    UNIQUE (snapshot_id, model_run_id, market_id, selection_id)
);

CREATE TABLE IF NOT EXISTS fact_sample_eligibility (
    eligibility_id VARCHAR PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    selection_id VARCHAR NOT NULL,
    as_of_time TIMESTAMPTZ NOT NULL,
    eligible BOOLEAN NOT NULL,
    quote_id VARCHAR,
    reason_codes JSON NOT NULL,
    quality_status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_bet_decision (
    decision_id VARCHAR PRIMARY KEY,
    prediction_id VARCHAR NOT NULL,
    quote_id VARCHAR NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL,
    fair_probability DOUBLE NOT NULL,
    market_probability DOUBLE NOT NULL,
    edge DOUBLE NOT NULL,
    expected_value DOUBLE NOT NULL,
    stake_recommended DECIMAL(14, 2) NOT NULL,
    bankroll_before DECIMAL(14, 2),
    policy_version VARCHAR NOT NULL,
    decision VARCHAR NOT NULL,
    reason_codes JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_bet_execution (
    execution_id VARCHAR PRIMARY KEY,
    decision_id VARCHAR NOT NULL,
    placed_at TIMESTAMPTZ NOT NULL,
    accepted_price_decimal DECIMAL(12, 6) NOT NULL,
    accepted_point DECIMAL(8, 3),
    stake DECIMAL(14, 2) NOT NULL,
    external_bet_id VARCHAR,
    status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_bet_settlement (
    execution_id VARCHAR NOT NULL,
    settled_at TIMESTAMPTZ NOT NULL,
    result VARCHAR NOT NULL,
    profit_loss DECIMAL(14, 2) NOT NULL,
    settlement_rule VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    source_reference VARCHAR,
    settlement_version INTEGER NOT NULL,
    PRIMARY KEY (execution_id, settlement_version)
);

CREATE TABLE IF NOT EXISTS fact_closing_quote (
    execution_id VARCHAR NOT NULL,
    quote_id VARCHAR NOT NULL,
    close_definition VARCHAR NOT NULL,
    closing_probability_no_vig DOUBLE,
    clv_probability DOUBLE,
    clv_log_price DOUBLE,
    PRIMARY KEY (execution_id, close_definition)
);

