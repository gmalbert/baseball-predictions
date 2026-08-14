CREATE TABLE IF NOT EXISTS run_manifest (
    run_id VARCHAR PRIMARY KEY,
    target_date DATE NOT NULL,
    as_of_time TIMESTAMPTZ NOT NULL,
    code_commit VARCHAR NOT NULL,
    environment_lock_hash VARCHAR NOT NULL,
    config_json JSON NOT NULL,
    status VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dataset_manifest (
    dataset VARCHAR NOT NULL,
    schema_version VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL,
    uri VARCHAR NOT NULL,
    output_sha256 VARCHAR NOT NULL,
    row_count BIGINT NOT NULL,
    quality_status VARCHAR NOT NULL,
    max_observed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset, schema_version, run_id)
);

CREATE TABLE IF NOT EXISTS quality_result (
    run_id VARCHAR NOT NULL,
    dataset VARCHAR NOT NULL,
    partition_key VARCHAR NOT NULL,
    check_name VARCHAR NOT NULL,
    passed BOOLEAN NOT NULL,
    severity VARCHAR NOT NULL,
    observed VARCHAR NOT NULL,
    expected VARCHAR NOT NULL,
    affected_rows BIGINT NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, dataset, partition_key, check_name)
);

CREATE TABLE IF NOT EXISTS data_revision (
    dataset VARCHAR NOT NULL,
    record_key VARCHAR NOT NULL,
    revision_number INTEGER NOT NULL,
    change_type VARCHAR NOT NULL,
    prior_hash VARCHAR,
    current_hash VARCHAR,
    observed_at TIMESTAMPTZ NOT NULL,
    run_id VARCHAR NOT NULL,
    PRIMARY KEY (dataset, record_key, revision_number)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_run_id VARCHAR PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    data_schema_version VARCHAR NOT NULL,
    artifact_uri VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ('development','shadow','challenger','champion','retired','quarantined')),
    promoted_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS experiment_registry (
    experiment_id VARCHAR PRIMARY KEY,
    hypothesis VARCHAR NOT NULL,
    code_commit VARCHAR NOT NULL,
    feature_version VARCHAR NOT NULL,
    folds_json JSON NOT NULL,
    search_space_json JSON NOT NULL,
    metrics_json JSON NOT NULL,
    decision VARCHAR NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL
);

