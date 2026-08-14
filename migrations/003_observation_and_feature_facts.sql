CREATE TABLE IF NOT EXISTS raw_observation (
    observation_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    source_record_id VARCHAR,
    event_time TIMESTAMPTZ,
    source_updated_at TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    request_params JSON NOT NULL,
    http_metadata JSON NOT NULL,
    payload_sha256 VARCHAR NOT NULL,
    payload_uri VARCHAR NOT NULL,
    ingestion_run_id VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_probable_pitcher_snapshot (
    game_id VARCHAR NOT NULL,
    team_id VARCHAR NOT NULL,
    player_id VARCHAR,
    status VARCHAR NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR,
    ingestion_run_id VARCHAR,
    PRIMARY KEY (game_id, team_id, observed_at, source)
);

CREATE TABLE IF NOT EXISTS fact_lineup_snapshot (
    game_id VARCHAR NOT NULL,
    team_id VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL,
    batting_order INTEGER,
    defensive_position VARCHAR,
    lineup_status VARCHAR NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR,
    ingestion_run_id VARCHAR,
    PRIMARY KEY (game_id, team_id, player_id, observed_at, source)
);

CREATE TABLE IF NOT EXISTS fact_weather_snapshot (
    game_id VARCHAR NOT NULL,
    forecast_for TIMESTAMPTZ NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    temperature_f DOUBLE,
    relative_humidity DOUBLE,
    pressure_hpa DOUBLE,
    wind_speed_mph DOUBLE,
    wind_direction_deg DOUBLE,
    precipitation_probability DOUBLE,
    roof_status VARCHAR,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR,
    ingestion_run_id VARCHAR,
    PRIMARY KEY (game_id, forecast_for, observed_at, source)
);

CREATE TABLE IF NOT EXISTS fact_roster_observation (
    team_id VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL,
    roster_status VARCHAR NOT NULL,
    availability_probability DOUBLE,
    observed_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR,
    PRIMARY KEY (team_id, player_id, observed_at, source)
);

CREATE TABLE IF NOT EXISTS fact_umpire_assignment (
    game_id VARCHAR NOT NULL,
    umpire_id VARCHAR NOT NULL,
    position VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    raw_payload_hash VARCHAR,
    PRIMARY KEY (game_id, umpire_id, position, observed_at, source)
);

CREATE TABLE IF NOT EXISTS feature_team_asof (
    team_id VARCHAR NOT NULL,
    as_of_time TIMESTAMPTZ NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    games_played INTEGER NOT NULL,
    offense_json JSON NOT NULL,
    defense_json JSON NOT NULL,
    baserunning_json JSON NOT NULL,
    uncertainty_json JSON NOT NULL,
    source_max_observed_at TIMESTAMPTZ,
    build_run_id VARCHAR NOT NULL,
    row_hash VARCHAR NOT NULL,
    PRIMARY KEY (team_id, as_of_time, feature_set_version)
);

CREATE TABLE IF NOT EXISTS feature_player_asof (
    player_id VARCHAR NOT NULL,
    as_of_time TIMESTAMPTZ NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    features_json JSON NOT NULL,
    posterior_json JSON NOT NULL,
    source_max_observed_at TIMESTAMPTZ,
    build_run_id VARCHAR NOT NULL,
    row_hash VARCHAR NOT NULL,
    PRIMARY KEY (player_id, as_of_time, feature_set_version, role)
);

CREATE TABLE IF NOT EXISTS model_run (
    model_run_id VARCHAR PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    training_start DATE NOT NULL,
    training_end DATE NOT NULL,
    validation_definition JSON NOT NULL,
    code_commit VARCHAR NOT NULL,
    environment_lock_hash VARCHAR NOT NULL,
    random_seed BIGINT NOT NULL,
    artifact_uri VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    calibration_uri VARCHAR,
    calibration_sha256 VARCHAR,
    metrics_json JSON NOT NULL,
    status VARCHAR NOT NULL
);

