# Spectral

Real-time anomaly detection engine for time-series data. Statistical, spectral, and deep learning methods in a single configurable pipeline.

```
                                          ┌─────────────┐
                                     ┌───>│  WebSocket   │
┌──────────┐    ┌──────────────────┐ │    └─────────────┘
│  Kafka   │───>│                  │ │    ┌─────────────┐
│  Redis   │───>│    Pipeline      │─┼───>│  Storage    │
│  CSV     │───>│                  │ │    │  (Postgres) │
│  API     │───>│  Pre ─> Detect   │ │    └─────────────┘
└──────────┘    │       ─> Score   │ │    ┌─────────────┐
                └──────────────────┘ └───>│  Alerting   │
                                          │  (Webhook)  │
                                          └─────────────┘
```

### Features

- **Multiple detection algorithms** - Z-Score, MAD, Grubbs' test, Isolation Forest, Spectral Residual (FFT-based), LSTM Autoencoder
- **Ensemble detection** - combine detectors via voting, averaging, or max scoring
- **Streaming and batch** - process data points in real-time or run detection on files
- **Configurable pipelines** - YAML configs for preprocessing, detector params, and alerting
- **REST API** - FastAPI endpoints for ingestion, detection, querying, and monitoring
- **WebSocket** - real-time anomaly alerts pushed to connected clients
- **Storage** - PostgreSQL/TimescaleDB for persisting data points and anomaly records
- **Alerting** - webhook notifications with severity filtering and cooldown
- **CLI** - detect anomalies from the terminal, generate test data, manage the server

### Detectors

| Algorithm | Type | Best for | Speed |
|-----------|------|----------|-------|
| Z-Score | Statistical | Gaussian data, obvious spikes | Fast |
| MAD | Statistical | Data with existing outliers | Fast |
| Grubbs | Statistical | Small datasets, rigorous testing | Fast |
| Isolation Forest | ML | High-dimensional, mixed anomaly types | Medium |
| Spectral Residual | Signal processing | Periodic signals, subtle deviations | Fast |
| LSTM Autoencoder | Deep learning | Complex temporal patterns | Slow |

### Quick start

```bash
# with docker
docker compose up -d

# or install locally
pip install -e ".[dev]"

# generate test data and run detection
spectral generate --points 5000 --output data/test.csv
spectral detect data/test.csv --config configs/default.yml

# start the api server
spectral serve --port 8000
```

### API

```bash
# batch detection
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"values": [1.0, 1.1, 0.9, 1.0, 8.5, 1.0, 0.95]}'

# ingest a data point
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-01-01T00:00:00", "value": 42.5, "series_id": "cpu-load"}'

# query anomalies
curl "http://localhost:8000/anomalies?series_id=cpu-load&min_severity=warning"

# pipeline status
curl http://localhost:8000/status
```

### Configuration

Pipeline behavior is controlled through YAML config files. See `configs/default.yml` for the baseline and `configs/example_pipeline.yml` for a full example with all options.

```yaml
pipeline:
  name: my-pipeline
  window_size: 200

detectors:
  - name: zscore
    params:
      threshold: 3.0
  - name: spectral_residual
    params:
      window_size: 48

ensemble:
  method: voting
  min_detectors: 2

alerting:
  enabled: true
  webhook_url: "https://hooks.slack.com/..."
  cooldown_seconds: 600
```

### Development

```bash
pip install -e ".[dev]"

make test          # run tests (skips slow + integration)
make test-all      # run all tests including LSTM training
make lint          # ruff check
make format        # ruff format
```

### Project structure

```
spectral/
├── src/spectral/
│   ├── detectors/      # anomaly detection algorithms
│   ├── pipeline/       # orchestration, preprocessing, windowing
│   ├── ingestion/      # data sources (CSV, Kafka, Redis, synthetic)
│   ├── api/            # FastAPI REST + WebSocket
│   ├── storage/        # SQLAlchemy models, repository
│   ├── alerting/       # webhook dispatcher
│   ├── cli/            # click CLI
│   └── config/         # YAML config loading + pydantic schemas
├── tests/              # pytest suite
├── configs/            # pipeline config files
├── docker/             # Dockerfile + init.sql
└── notebooks/          # demo notebook
```
