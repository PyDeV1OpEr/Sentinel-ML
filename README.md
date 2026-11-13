# Sentinel ML

**Time-series anomaly detection and ML experimentation platform** — six detection algorithms, a
trainable and persistent LSTM autoencoder, honest held-out evaluation, a FastAPI service and a
Streamlit laboratory for training, testing and inspecting models.

[![CI](https://github.com/PyDeV1OpEr/Sentinel-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/PyDeV1OpEr/Sentinel-ML/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Built on the MIT-licensed [peteroyce/spectral](https://github.com/peteroyce/spectral) engine. The
upstream attribution and licence are preserved; [ASSESSMENT.md](ASSESSMENT.md) records the
original audit and every behavioural change is documented in the detector docstrings.

---

## The problem

Univariate sensor and business series are easy to plot and hard to trust: a threshold that catches
a spike also fires on seasonality, an autoencoder that reconstructs normal data can still flag an
entire unseen series, and a model that reports 99% accuracy can be useless when anomalies are 3% of
the data. Sentinel ML addresses the engineering problem around detection, not just the algorithm:

- every prediction is made **out of sample**;
- model weights, scaler and threshold are **persisted together** and reload without training;
- evaluation uses **one-to-one event matching**, so a smeared prediction cannot claim several hits;
- unlabelled data never receives fabricated precision or recall;
- a laboratory lets you reproduce, inspect and challenge the numbers rather than trusting a chart.

---

## Key features

- **Six algorithms + ensemble:** Z-Score, MAD, Grubbs' test, Isolation Forest, Spectral Residual
  (FFT) and an LSTM autoencoder, combined by voting, averaging or maximum score.
- **Train → save → restart → load → infer.** Artefacts are directories containing a CPU
  `state_dict` plus JSON metadata, normalisation and calibration; loading never retrains.
- **Reproducible training.** Seeded PyTorch training, chronological calibration holdout, early
  stopping on held-out reconstruction loss.
- **Corrected LSTM scoring.** Each point is scored by the *minimum* error among the windows that
  cover it, which localises events instead of smearing a spike across its window.
- **Interactive Streamlit workbench:** synthetic data, CSV upload, training curves, model registry,
  detection, evaluation, six-detector comparison, visual drill-down and experiment history.
- **Reproducible A–I benchmark** covering clean data, spikes, dense spikes, level shifts, drift,
  noise bursts, flatlines, mixed unseen data and an unlabelled CSV.
- **FastAPI service** with batch and streaming detection, persisted anomalies (optional
  PostgreSQL/TimescaleDB), webhook alerting and WebSocket fan-out on the real request path.
- **Local-first:** PostgreSQL, Redis and Kafka are optional. A fresh clone runs with the workbench
  and API without any infrastructure.

---

## Architecture

```mermaid
flowchart LR
    CSV[CSV upload] --> LAB[spectral.lab]
    SYN[Synthetic generator] --> LAB
    LAB --> DET[Six detectors + ensemble]
    LAB --> ART[(Versioned LSTM artefacts)]
    LAB --> HIST[(experiment JSONL)]
    WB[Streamlit workbench] --> LAB

    HTTP[FastAPI /ingest /detect] --> SVC[DetectionService]
    SVC --> PIPE[Pipeline + preprocessing]
    PIPE --> DET
    SVC --> DB[(PostgreSQL - optional)]
    SVC --> ALERT[Webhook dispatcher - optional]
    SVC --> WS[WebSocket /ws/stream]

    CLI[CLI scripts / spectral CLI] --> LAB
    CLI --> PIPE
```

Detection is **window based** for methods that need context. In streaming mode a point is scored
only when a full window has accumulated, and the pipeline reports the label for the newest point.

| Package | Responsibility |
|---|---|
| `src/spectral/detectors/` | Detector contract, six implementations, registry, ensemble |
| `src/spectral/pipeline/` | Preprocessing, orchestration, stream buffering |
| `src/spectral/lab/` | Dataset preparation, metrics, evaluation, artefact persistence, plots |
| `src/spectral/api/` | FastAPI app, routes, schemas, detection service, WebSocket |
| `src/spectral/storage/` | Async SQLAlchemy models and repository (optional PostgreSQL) |
| `src/spectral/alerting/` | Severity routing and cooldown-aware webhook delivery |
| `workbench/` | Streamlit UI; calls `spectral.lab`, never reimplements detection |
| `scripts/` | Reproducible train/evaluate/predict/benchmark entry points |
| `notebooks/` | Upstream matplotlib walkthrough, updated to the current API |

---

## The six detectors

| Detector | Family | Learns from data | Best suited to | Main limitation |
|---|---|---|---|---|
| **Z-Score** | Statistical | mean / std | roughly Gaussian data, clear point spikes | one outlier inflates its own denominator |
| **MAD** | Statistical | median / MAD | baselines that already contain outliers | ignores contextual events |
| **Grubbs' test** | Statistical | none (iterative test) | small samples, formal outlier testing | too conservative on long series with many events |
| **Isolation Forest** | Tree ensemble | lagged windows | mixed point/contextual anomalies | contamination is a prior, not a calibrated rate |
| **Spectral Residual (FFT)** | Signal processing | none | periodic signals, spectral shifts | misses slow drift and flatlines |
| **LSTM Autoencoder** | Deep learning | sequence reconstruction | sustained or complex temporal events | dense isolated spikes can flood the window context |
| **Ensemble** | Meta | constituent detectors | precision-first consensus | inherits its members' blind spots |

### LSTM training and inference

The autoencoder is a two-layer-free design: an encoder LSTM compresses a window, its final
hidden/cell state seeds a decoder LSTM, and a linear head reconstructs the window. The wrapper
implements:

1. **Chronological split.** The last `calibration_fraction` (default 20%) of *windows* is held out
   of optimisation. A random split would leak overlapping windows into training.
2. **Train-only normalisation.** Mean/std are computed from the training portion only and stored
   with the model.
3. **Early stopping** on calibration-holdout reconstruction MSE, restoring the best state.
4. **Robust threshold:** `median + threshold_sigma × (IQR / 1.349)` on held-out window errors. For
   Gaussian errors `IQR / 1.349` estimates the standard deviation while resisting a minority of
   outliers, so `threshold_sigma` remains "standard deviations above typical".
5. **Point scoring:** a point's score is the minimum window MSE among the windows covering it. A
   single spike raises every window that contains it; its neighbours still have a window that
   excludes it, so the minimum localises the event. Sustained events have no clean covering window
   and remain flagged.

`random_seed`/`seed` (default 42) controls weight initialisation and DataLoader shuffling. Training
is CPU-first; on an Apple M4 the model is small and CPU is faster than MPS for this workload. MPS is
neither required nor claimed as an optimisation.

### Model persistence

```text
artifacts/lstm_autoencoder/model_001/
├── model.pt            # CPU state_dict, loaded with weights_only=True
├── metadata.json       # version, timestamps, architecture, hyperparameters, dataset fingerprint,
│                       # training history, evaluation metrics, SHA-256 of model.pt
├── preprocessing.json  # train_mean, train_std
└── calibration.json    # threshold, robust statistics, source, window count
```

Loading validates the schema, required files, checksum, architecture and finite parameters, then
rebuilds the detector on CPU and restores weights, scaler and threshold. `fit()` is never called.
Round-trip predictions are tested to be identical (`tests/test_lab/test_artifacts.py`). Artefact
versions are auto-incremented; the registry supports save/load/inspect/delete with path validation.

---

## Evaluation methodology

- **Chronological train/test split**, default 70/30. The LSTM's calibration holdout lives inside the
  training half, so the test half is untouched by optimisation and threshold estimation.
- **One-to-one event matching.** Each predicted point can match at most one ground-truth point within
  ±`tolerance` points (default 2). A 20-point smear cannot inflate recall.
- **Full metric set:** precision, recall, F1, accuracy, false-positive rate, TP/FP/FN/TN, per-anomaly-
  type recall, PR curve, threshold sweep.
- **No ground truth, no metrics.** An uploaded/unlabelled CSV shows predictions and scores only.
- **Fixtures are fixed seeds.** The benchmark is generated from explicit seeds; no tuning against
  held-out scenarios.

### Verified benchmark

Protocol: one clean seasonal baseline of 1,200 points; every detector is fitted on its first 70%.
Scenario A scores the clean held-out 30%. B–H are separate, unseen synthetic datasets. The LSTM is
trained once on the clean baseline (100 epochs, hidden 32, sequence 30, seed 42; fit ≈3.4 s on an
M4 CPU) and reused for every scenario. Scenario I scores an unlabelled CSV.

| Scenario | Best detector (F1) | LSTM precision | LSTM recall | LSTM F1 | LSTM FP |
|---|---|---|---|---|---|
| A · Clean seasonal (0 anomalies) | — (FPR: LSTM 0.000, IF 0.042) | — | — | — | **0** |
| B · Isolated spikes | Z-Score / Spectral / Ensemble 0.986 | 0.105 | 1.000 | 0.190 | 306 |
| C · Dense spikes and drops | Z-Score / Ensemble 0.962 | 0.092 | 1.000 | 0.169 | 946 |
| D · Level shifts | **LSTM 0.932** (Ensemble 0.927) | 0.873 | 1.000 | **0.932** | 15 |
| E · Gradual drift | **LSTM 0.960** (IF 0.644) | 1.000 | 0.924 | **0.960** | 0 |
| F · Noise bursts | **LSTM 0.970** (IF 0.769) | 1.000 | 0.941 | **0.970** | 0 |
| G · Flatline sensor | **LSTM 0.827** (IF 0.343) | 0.860 | 0.796 | **0.827** | 7 |
| H · Mixed unseen | **LSTM 0.962** (IF 0.635) | 0.971 | 0.953 | **0.962** | 3 |
| I · Unlabelled CSV | no metrics | 47 / 600 flagged (7.83%) | — | — | — |

Full per-detector tables, per-type recall and timings: [`reports/scenario_results.md`](reports/scenario_results.md)
(regenerate with the command below). The LSTM wins on sustained and mixed events and loses badly on
dense point spikes. That is a measured limitation of 30-point window context, not a tuning failure:
scenarios B and C put many spikes closer together than the window length. Z-Score, Spectral
Residual, or a shorter `sequence_length` are the right tools for that regime.

> These are synthetic benchmarks. They demonstrate the evaluation machinery and comparative
> behaviour; they are not a claim of production accuracy on your data.

---

## Screenshots

| Overview | Dataset laboratory |
|---|---|
| ![Workbench overview](screenshots/01-workbench-overview.png) | ![Dataset laboratory](screenshots/02-dataset-laboratory.png) |

| Training | Model registry |
|---|---|
| ![Training](screenshots/03-model-training.png) | ![Model registry](screenshots/04-model-registry.png) |

| Evaluation | Detector comparison |
|---|---|
| ![Evaluation](screenshots/05-model-evaluation.png) | ![Detector comparison](screenshots/06-detector-comparison.png) |

| Anomaly visualisation | Detection and export |
|---|---|
| ![Anomaly visualisation](screenshots/07-anomaly-visualisation.png) | ![Detection](screenshots/08-anomaly-detection.png) |

---

## Installation

Requires **Python 3.11+** (verified with 3.12 on macOS arm64). No CUDA, Docker or database is needed
for the first experiment.

```bash
git clone https://github.com/PyDeV1OpEr/Sentinel-ML.git
cd Sentinel-ML

python3.12 -m venv .venv
source .venv/bin/activate

# Workbench + lab + test tooling
pip install -e ".[dev,workbench]"

# Or only the runtime/workbench (no pytest/ruff/jupyter)
# pip install -e ".[workbench]"
```

Verify the install:

```bash
python -c "import torch, spectral, streamlit, pandas, matplotlib; print(torch.__version__)"
```

The detector chooses CUDA when available, otherwise CPU. Apple MPS is available in PyTorch but is
intentionally not selected: at hidden 32 / sequence 30 / batch 32, Metal launch overhead dominates.
Peak process memory for a 1,200-point training run is roughly 500–600 MiB.

---

## Quick start

```bash
streamlit run workbench/app.py
```

Open the URL Streamlit prints (normally <http://localhost:8501>; this checkout was tested on 8507).
Then:

1. **Dataset laboratory → Scenario presets → A · Clean seasonal.**
2. **Train → Train LSTM.** Defaults: 100 epochs, hidden 32, sequence 30, seed 42, holdout 0.20.
3. **Save to artefacts.** A `model_00N` directory is created.
4. **Dataset laboratory → H · Mixed unseen.**
5. **Evaluate → Saved LSTM artefact → Load and evaluate.** Inspect metrics, confusion matrix,
   PR curve, reconstruction error and individual false positives/negatives.
6. **Compare → Run detector comparison.** Compare all six detectors and the ensemble on the held-out
   30% of the current dataset.
7. **Detect** exports predictions and scores to CSV. **Experiments** records every run locally.

Training never starts when the app is merely opened. Loading an artefact never retrains it. The
first experiment takes about two minutes on CPU. The step-by-step version is
[TESTING_GUIDE.md](TESTING_GUIDE.md).

---

## CLI examples

Generate a labelled baseline and an unseen test set:

```bash
python scripts/generate_dataset.py \
  --output data/lab_clean.csv --points 1200 --anomaly-ratio 0 --seed 42
python scripts/generate_dataset.py \
  --output data/lab_mixed.csv --points 1200 --anomaly-ratio 0.06 --seed 148
```

Train, save, reload and predict:

```bash
python scripts/train_lstm.py --input data/lab_clean.csv --epochs 100
python scripts/predict.py \
  --input data/lab_mixed.csv \
  --model artifacts/lstm_autoencoder/model_001
python scripts/evaluate_lstm.py \
  --input data/lab_mixed.csv \
  --model artifacts/lstm_autoencoder/model_001 \
  --export-csv data/predictions.csv
```

Reproduce the A–I benchmark (writes JSON, Markdown and unlabelled predictions):

```bash
python scripts/run_test_scenarios.py --points 1200 --epochs 100
```

The upstream CLI remains available:

```bash
spectral generate --points 5000 --anomaly-ratio 0.03 --output data/synthetic.csv
spectral detect data/synthetic.csv --config configs/default.yml
spectral detect data/synthetic.csv --format json > anomalies.json  # stdout stays parseable
spectral serve --host 127.0.0.1 --port 8000
```

### Library use

```python
import numpy as np

from spectral.detectors import (
    EnsembleDetector,
    MADDetector,
    SpectralResidualDetector,
    ZScoreDetector,
)

rng = np.random.default_rng(42)
values = np.sin(np.arange(600) * 0.1) + rng.standard_normal(600) * 0.05
values[[120, 300, 480]] += 6.0

ensemble = EnsembleDetector(
    [
        ZScoreDetector(threshold=3.0),
        MADDetector(threshold=3.5),
        SpectralResidualDetector(window_size=24, score_threshold=3.0),
    ],
    method="voting",
    min_detectors=2,
)
result = ensemble.fit_detect(values)

print(result.anomaly_count, result.threshold)
print(np.flatnonzero(result.labels))
```

---

## FastAPI service

```bash
uvicorn spectral.api.app:app --reload --port 8000
```

Interactive OpenAPI documentation: <http://localhost:8000/docs>.

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Ingest one point or a batch; run streaming detection, persist, alert and broadcast |
| POST | `/detect` | Batch detection over `values` and optional matching `timestamps` |
| GET | `/anomalies` | Query persisted anomalies by series, time, severity or detector |
| GET | `/status` | Pipeline name, real uptime, counters and detector list |
| GET | `/metrics` | Pipeline counters plus storage aggregates when enabled |
| WS | `/ws/stream` | Anomaly broadcasts, `ping`/`pong` and 30-second heartbeats |

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-01-01T00:00:00", "value": 42.5, "series_id": "cpu-load"}'

curl "http://localhost:8000/status"
```

Batch detection needs a real window rather than a handful of points (the first call fits the
pipeline). This example sends 120 seasonal points with one spike at index 60:

```bash
python - <<'PY'
import json
import urllib.request

import numpy as np

values = list(np.sin(np.arange(120) * 0.1))
values[60] = 8.0
request = urllib.request.Request(
    "http://localhost:8000/detect",
    data=json.dumps({"values": values}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(json.load(response)["anomaly_count"], "anomalies")
PY
```

**Storage is off by default** (`configs/default.yml`, `StorageConfig.enabled=False`), so the API
runs without PostgreSQL. With storage disabled, `/anomalies` returns HTTP 503 with an actionable
message and `/metrics` returns live pipeline statistics plus `"storage_enabled": false`; it does not
pretend an empty database was queried. To enable persistence:

```bash
export DATABASE_URL="postgresql+asyncpg://spectral:spectral@localhost:5432/spectral"
# set storage.enabled: true in the config, then start the service
docker compose up -d timescaledb   # optional local database
```

Alerting is also opt-in (`alerting.enabled`, `webhook_url`, `min_severity`). Detection results are
never dependent on an optional integration: persistence, webhook and WebSocket failures are logged
and degrade to detection-only.

CORS is closed by default. Set `SENTINEL_CORS_ORIGINS` to a comma-separated allow-list if a browser
client on another origin needs access. The service has no authentication or rate limiting; do not
expose it publicly without adding those controls.

---

## Dataset format

The lab accepts any CSV that can be mapped explicitly:

| Column | Requirement |
|---|---|
| `timestamp` | Parseable by `pandas.to_datetime`; sorted automatically; duplicate timestamps keep the last |
| `value` | Numeric; missing values are interpolated **after** the split, never across it |
| `is_anomaly` | Optional; `0`/`1` only. Without it, no precision/recall/F1 is reported |
| `anomaly_type` | Optional; carried through to evaluation and exports |

A training series needs at least `sequence_length + 2` points; inference needs at least
`sequence_length`. The workbench's upload tab exposes column mapping; `scripts/*.py` accept
`--timestamp-col`, `--value-col` and `--label-col`. A small unlabelled example is provided at
[`examples/example_unlabelled.csv`](examples/example_unlabelled.csv).

Train the autoencoder on a **clean** baseline. If the training half contains labelled anomalies the
model can learn to reconstruct them; the workbench warns, and the benchmark never does this.

---

## Tests, lint and CI

```bash
pytest -q -m "not integration"    # fast + slow, no database
pytest -q -m "slow"               # LSTM and artefact lifecycle
pytest -q -m integration          # needs PostgreSQL/TimescaleDB
ruff check src/ tests/ scripts/ workbench/
```

The suite covers detectors and validation errors, the ensemble contract, preprocessing and
windowing, ingestion, storage, alerting, the API (including the detection service and WebSocket),
the lab metrics with one-to-one matching, artefact lifecycle and corruption handling, the seeded
LSTM behaviour, and the CLI/lab integration.

GitHub Actions installs the project on Python 3.12, runs Ruff, compiles source/scripts/workbench,
and runs `pytest -m "not integration"`. Database tests require a PostgreSQL service and are not
silently skipped: run them explicitly with `-m integration` when a database is configured. The three
storage tests fail with `role "spectral" does not exist` on a machine without that database; that is
an environment prerequisite, not a code pass.

---

## Limitations

- **Dense point spikes remain the LSTM's weak case.** With `sequence_length=30`, spikes closer than
  the window contaminate nearly every window; precision falls to 0.09–0.11 on scenarios B/C. The
  benchmark reports this rather than tuning it away. Use Z-Score/Spectral Residual or a shorter
  window for that regime.
- **Univariate only.** Every detector consumes a single `value` series; there is no multivariate
  feature support.
- **Synthetic evidence.** The benchmark is generated, reproducible and seeded, but it is not a
  substitute for evaluation on your own labelled data. No claim of universal superiority is made.
- **Threshold stationarity.** Calibration assumes the held-out training tail represents normal
  behaviour. A regime change after training can raise or lower the false-positive rate.
- **Optional integrations are local-first.** PostgreSQL/TimescaleDB and webhooks are implemented and
  unit-tested but are not required for a demo; no authentication, rate limiting or multi-tenant
  isolation is included.
- **No public deployment.** The workbench is intended to run locally. Streamlit Community Cloud
  deployment is documented as an option but has not been performed; no live demo URL is claimed.
- **LSTM training is CPU-first.** MPS is not used; very large datasets should be downsampled or
  windowed before training.

---

## Project structure

```text
Sentinel-ML/
├── configs/                 default and example pipeline YAML
├── docker/                  Dockerfile, TimescaleDB init
├── examples/                small unlabelled CSV
├── notebooks/demo.ipynb     upstream walkthrough, updated
├── reports/                 benchmark JSON/Markdown and unlabelled predictions
├── screenshots/             real captures from the running workbench
├── scripts/                 generate, train, predict, evaluate, benchmark
├── src/spectral/            detectors, lab, pipeline, api, storage, alerting, CLI
├── tests/                   detector, lab, API, pipeline, storage tests
├── workbench/app.py         Streamlit experimentation UI
├── ASSESSMENT.md            Phase 1 audit and change log
├── TESTING_GUIDE.md         step-by-step recruiter walkthrough
├── LICENSE                  MIT licence (upstream + project attribution)
└── pyproject.toml
```

---

## Attribution and licence

Sentinel ML is built on **[peteroyce/spectral](https://github.com/peteroyce/spectral)**, MIT
licensed, © 2026 Pete Royce. The upstream copyright notice is retained; Sentinel ML's own
modifications are additionally attributed to Mohammed Unais. Upstream modules remain recognisable;
modifications and additions are documented in code docstrings, `ASSESSMENT.md` and the Phase 1/Phase
2 work:

- `EnsembleDetector.fit_detect()` implemented consistently with `BaseDetector`;
- LSTM calibration moved to a chronological held-out split with a robust threshold, seeded training
  and minimum-window point scoring;
- input validation and error handling for every detector;
- versioned LSTM artefact persistence, the `spectral.lab` evaluation layer, synthetic generator,
  Streamlit workbench, benchmark scripts and regression tests;
- FastAPI persistence/alert/WebSocket wiring and explicit optional-storage behaviour.

No upstream history is rewritten. See [LICENSE](LICENSE) for the full MIT text.
