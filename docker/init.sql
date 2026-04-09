CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS data_points (
    id BIGSERIAL,
    series_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('data_points', 'timestamp', if_not_exists => TRUE);

CREATE INDEX idx_data_points_series ON data_points (series_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS anomalies (
    id BIGSERIAL PRIMARY KEY,
    series_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    detector VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'warning',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_anomalies_series ON anomalies (series_id, timestamp DESC);
CREATE INDEX idx_anomalies_severity ON anomalies (severity);
CREATE INDEX idx_anomalies_detector ON anomalies (detector);
