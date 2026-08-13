# Spectral

An anomaly detection engine for time-series data. Statistical, signal-processing, and deep-learning
detectors sit behind one configurable pipeline that can score a CSV file in batch or a live stream
point by point, with results persisted to TimescaleDB and pushed to a webhook.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)

## Features

- **Six detectors** behind one registry: Z-Score, MAD, Grubbs' test, Isolation Forest, Spectral
  Residual (FFT-based), and an LSTM autoencoder.
- **Ensemble scoring** with three combination strategies — `voting` (require *k* of *n* detectors to
  agree), `average`, and `max`.
- **Batch and streaming modes** — `Pipeline.detect()` scores a full array; `detect_streaming()`
  buffers points and only scores once a full window has accumulated.
- **Preprocessing chain** — min-max, z-score, or robust normalisation, linear detrending, seasonal
  decomposition, and missing-value filling by interpolation, forward-fill, or mean.
- **Severity scoring** — each anomaly is graded low/warning/high/critical from its score, and
  alerting filters on that grade.
- **Declarative YAML pipelines** — detectors, their parameters, ensemble policy, alerting, and
  storage are all configuration, validated by Pydantic models.
- **REST API and WebSocket** — FastAPI endpoints for ingestion, batch detection, querying, and
  metrics, plus a `/ws/stream` endpoint with a connection manager and heartbeats.
- **TimescaleDB storage** — async SQLAlchemy repository over a hypertable for data points and an
  indexed anomalies table.
- **Webhook alerting** with per-series cooldown and a minimum severity threshold.
- **CLI** for file detection, synthetic data generation, serving, and status.

## Architecture

Detection runs on a *window*, not a single point. That is the reason for the buffer in streaming
mode: methods like Spectral Residual and the LSTM autoencoder are meaningless on one value, so a
point is scored only when it is the newest element of a full window, and the pipeline reports the
label for that last position.

```
  CSV file ─┐                ┌───────────────────────────────────┐
            │                │             Pipeline              │
  synthetic ┼──> DataPoint ─>│                                   │
            │                │  Preprocessor ─> detectors ─┐     │
  HTTP      ─┘               │   normalise      zscore     │     │
  /ingest                    │   detrend        mad        ├─> Ensemble
                             │   deseason       iforest    │   (voting/avg/max)
                             │   fill           spectral   │        │
                             │                  lstm_ae ───┘        │
                             └──────────────────────┬───────────────┘
                                                    │ Anomaly(score, severity)
                        ┌───────────────────────────┼──────────────────────┐
                        v                           v                      v
              ┌───────────────────┐     ┌────────────────────┐   ┌──────────────────┐
              │ AnomalyRepository │     │  AlertDispatcher   │   │  API responses   │
              │ TimescaleDB       │     │  severity filter + │   │  /detect /query  │
              │ hypertable        │     │  per-series        │   │  /metrics        │
              └───────────────────┘     │  cooldown          │   └──────────────────┘
                                        └─────────┬──────────┘
                                                  v
                                             webhook POST
```

| Module | Responsibility |
|---|---|
| `detectors/` | Detector implementations, `DETECTORS` registry, `EnsembleDetector` |
| `pipeline/` | `Pipeline` orchestration, preprocessing functions, stream windowing |
| `ingestion/` | `BaseSource` and `DataPoint`, CSV and synthetic sources, Kafka and Redis Stream sources |
| `storage/` | SQLAlchemy models and the async `AnomalyRepository` |
| `alerting/` | Severity dispatcher and the cooldown-aware webhook sender |
| `api/` | FastAPI app, routes, dependencies, WebSocket connection manager |
| `cli/`, `config/` | Click entry point; YAML loading and Pydantic schemas |

Note on sources: `CSVSource` and `SyntheticSource` are exported from `spectral.ingestion` and used by
the CLI. `KafkaSource` and `RedisStreamSource` are implemented against the same `BaseSource`
interface but are not yet wired into the served pipeline — the API ingests points over HTTP. Kafka
support is an optional extra (`pip install -e ".[kafka]"`) because `confluent-kafka` is a heavy
dependency that most users will not need.

## Quickstart

```bash
git clone https://github.com/peteroyce/spectral
cd spectral
make setup                       # pip install -e ".[dev]"
cp .env.example .env
```

Environment variables, from `.env.example`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async Postgres/TimescaleDB DSN, e.g. `postgresql+asyncpg://spectral:spectral@localhost:5432/spectral` |
| `REDIS_URL` | Redis connection string |
| `WEBHOOK_URL` | Destination for anomaly alerts; empty disables sending |
| `KAFKA_BOOTSTRAP_SERVERS` | Broker list for the optional Kafka source |

Run everything with Docker Compose — it brings up the app, a TimescaleDB instance seeded from
`docker/init.sql`, and Redis:

```bash
make docker-up
```

Or work locally:

```bash
spectral generate --points 5000 --anomaly-ratio 0.03 --output data/synthetic.csv
spectral detect data/synthetic.csv --config configs/default.yml
spectral serve --port 8000        # SPECTRAL_CONFIG is set from --config
spectral status                   # queries /status on a running server
```

## Usage

```python
import numpy as np
from spectral.config import get_default_config
from spectral.pipeline.runner import Pipeline

pipeline = Pipeline(get_default_config())
anomalies = pipeline.detect(values, timestamps)

for a in anomalies[:5]:
    print(a.index, a.value, round(a.score, 3), a.severity, a.detector)
```

Detectors can also be used on their own:

```python
from spectral.detectors import create_detector, EnsembleDetector, ZScoreDetector, MADDetector

ensemble = EnsembleDetector(
    detectors=[ZScoreDetector(threshold=3.0), MADDetector(threshold=3.5),
               create_detector("spectral_residual", window_size=24)],
    method="voting",
    min_detectors=2,
)
result = ensemble.fit_detect(values)
```

Requiring consensus is the point of the ensemble. On the synthetic series in `notebooks/demo.ipynb`
(2000 points, 58 injected anomalies), the individual detectors reach precision 0.741 (Z-Score),
0.710 (MAD), 0.684 (Spectral Residual), and 0.260 (Isolation Forest); voting with `min_detectors=2`
reaches precision 0.833 at recall 0.345. Higher precision, slightly lower recall — the right trade
when every alert costs someone's attention.

### Detectors

| Key | Class | Family | Suited to |
|---|---|---|---|
| `zscore` | `ZScoreDetector` | Statistical | Roughly Gaussian data, clear spikes |
| `mad` | `MADDetector` | Statistical | Data whose baseline already contains outliers |
| `grubbs` | `GrubbsDetector` | Statistical | Small samples, formal hypothesis testing |
| `isolation_forest` | `IsolationForestDetector` | Tree ensemble | Windowed features, mixed anomaly shapes |
| `spectral_residual` | `SpectralResidualDetector` | Signal processing | Periodic signals, subtle deviations |
| `lstm_autoencoder` | `LSTMAutoencoderDetector` | Deep learning | Complex temporal structure; needs training time |

### HTTP API

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Ingest one point or a batch; runs streaming detection and returns any anomalies |
| POST | `/detect` | Batch detection over a supplied array of values |
| GET | `/anomalies` | Query stored anomalies, filterable by series and severity |
| GET | `/status` | Pipeline name, state, points processed, anomalies detected, active detectors |
| GET | `/metrics` | Pipeline counters plus database statistics over a time window |
| WS | `/ws/stream` | WebSocket channel; the connection manager exposes an anomaly broadcast helper |

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"values": [1.0, 1.1, 0.9, 1.0, 8.5, 1.0, 0.95]}'

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-01-01T00:00:00", "value": 42.5, "series_id": "cpu-load"}'

curl "http://localhost:8000/anomalies?series_id=cpu-load"
```

## Configuration

`configs/default.yml` is the baseline; `configs/example_pipeline.yml` enables every detector with
comments explaining each choice.

```yaml
pipeline:
  name: my-pipeline
  window_size: 200
  step_size: 5

preprocessing:
  normalize: true
  method: robust          # robust scaling tolerates outliers in the fitting data
  detrend: true
  decompose_seasonality: true
  seasonal_period: 24

detectors:
  - name: zscore
    params: { threshold: 3.0 }
  - name: spectral_residual
    params: { window_size: 48, score_threshold: 3.0 }

ensemble:
  method: voting
  min_detectors: 2

alerting:
  enabled: true
  webhook_url: "https://example.com/hooks/spectral"
  cooldown_seconds: 600
  min_severity: warning
```

## Tech stack

Python 3.11+, NumPy, SciPy, scikit-learn, PyTorch, pandas, FastAPI, Uvicorn, WebSockets, async
SQLAlchemy 2 with asyncpg against TimescaleDB, Redis, Pydantic 2, Click, Rich, PyYAML. Optional:
confluent-kafka. Built with hatchling, linted with ruff.

## Testing

```bash
make test        # skips tests marked slow or integration
make test-all    # everything, including LSTM training
make lint
```

91 tests cover the detectors, pipeline, preprocessing, windowing, ingestion, storage, alerting, and
the API including the WebSocket. Slow and integration tests are marked so the default run stays fast;
CI runs ruff and the fast subset on Python 3.11.

## License

MIT
