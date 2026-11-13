# Sentinel ML — Phase 1 Technical Assessment

> **Historical document.** This is the audit written before any changes to the upstream code. It
> describes the original `peteroyce/spectral` state, not the current release. The temporary
> `verification/` scripts and single-detector configs it references lived in the development
> workspace and are not shipped; their conclusions are superseded by `reports/scenario_results.md`
> and the regression suite. Kept for provenance and attribution.

**Subject:** `peteroyce/spectral` v0.3.1 (MIT, © 2026 Pete Royce)
**Commit inspected:** `217cae0` — "Rewrite README" (2026-08-13)
**Local clone:** `Sentinel-ML/` (unmodified, `git status` clean)
**Environment:** macOS 26.6, Apple M4 (arm64), 24 GB RAM, Python 3.12.15, PyTorch 2.15.0
**Status:** understand → install → run → validate. **No upstream code was changed.**

---

## 1. Original repository architecture

2,895 lines of source across 8 packages, 1,087 lines of tests, 91 tests. Everything below was read in full.

```
Sentinel-ML/
├── pyproject.toml            hatchling, requires-python >=3.11, 17 deps + [dev] + [kafka]
├── Makefile                  setup / test / test-all / lint / format / serve / docker-up / generate / detect
├── configs/                  default.yml (3 detectors), example_pipeline.yml (all 5 classical+DL)
├── docker/                   Dockerfile (python:3.11-slim), init.sql (TimescaleDB hypertable + indexes)
├── docker-compose.yml        app + timescaledb + redis
├── notebooks/demo.ipynb      matplotlib walkthrough (the only "visual" artifact)
├── src/spectral/
│   ├── detectors/            base.py (contract), statistical.py, isolation_forest.py,
│   │                         spectral_residual.py, lstm_autoencoder.py, ensemble.py, __init__.py (registry)
│   ├── pipeline/             runner.py (orchestrator), preprocessing.py, window.py
│   ├── ingestion/            base.py, csv_loader.py (csv/json/parquet), synthetic.py, redis_stream.py, kafka_consumer.py
│   ├── config/               schema.py (Pydantic), loader.py (YAML → deep-merge with defaults)
│   ├── api/                  app.py, routes.py, dependencies.py (singletons), schemas.py, websocket.py
│   ├── storage/              database.py (async engine), models.py (SQLAlchemy 2), repository.py
│   ├── alerting/             dispatcher.py (severity gate), webhook.py (per-series cooldown)
│   └── cli/                  main.py (Click + Rich: detect / generate / serve / status)
└── tests/                    9 packages mirroring the source layout
```

**Data flow (as designed):** `CSVSource | SyntheticSource | HTTP /ingest` → `DataPoint` → `Preprocessor`
(fill → deseasonalise → detrend → normalise) → detectors → `EnsembleDetector` → `Anomaly(score, severity)`
→ repository / alert dispatcher / API response.

**Key architectural facts**

- Detectors sit behind a single ABC (`base.py:43`) with a `DetectionResult` dataclass (`base.py:13`) carrying
  `scores`, `labels`, `threshold`, `metadata`. A string registry maps config keys → classes (`detectors/__init__.py:8`).
- `Pipeline` (`runner.py:42`) is the only orchestrator. It builds detectors from config, owns the preprocessor,
  a `StreamBuffer`, and counters. Two entry points: `detect()` (`:117`, batch) and `detect_streaming()` (`:173`, one point at a time).
- Configuration is declarative YAML validated by Pydantic, deep-merged over a code default (`loader.py:38`).
- The API holds **one mutable `Pipeline` singleton** for the whole process (`dependencies.py:13`), created in the
  FastAPI lifespan (`app.py:34`).

---

## 2. The six algorithms and their exact locations

| # | Algorithm | Class / file:line | Fit | Detect | Method | Default params |
|---|---|---|---|---|---|---|
| 1 | Z-Score | `ZScoreDetector` — `detectors/statistical.py:13` | `:26` | `:36` | \|x−μ\|/σ from fit stats; label if > threshold | `threshold=3.0` |
| 2 | MAD | `MADDetector` — `statistical.py:60` | `:74` | `:84` | Modified z-score `0.6745·\|x−median\|/MAD` | `threshold=3.5` |
| 3 | Grubbs' test | `GrubbsDetector` — `statistical.py:108` | `:121` (no-op) | `:125` | Iterative single-outlier removal using the t-distribution critical value G_crit; `alpha` per test, N−2 dof | `alpha=0.05`, `max_outliers=None` |
| 4 | Isolation Forest | `IsolationForestDetector` — `detectors/isolation_forest.py:13` | `:59` | `:77` | Lag features `[x(t)…x(t−k+1)]` (`_build_features:46`) → sklearn `IsolationForest`; score from `decision_function` flipped and min-max scaled | `contamination=0.05`, `n_estimators=100`, `window_features=5` |
| 5 | Spectral Residual (FFT) | `SpectralResidualDetector` — `detectors/spectral_residual.py:13` | `:40` (no-op) | `:76` | FFT → log amplitude → subtract moving-average-smoothed log spectrum (`_spectral_residual:54`) → recombine with phase → IFFT saliency (`_saliency_map:70`) → z-score the saliency | `window_size=24`, `score_threshold=3.0` |
| 6 | LSTM Autoencoder | `_LSTMAutoencoder` — `detectors/lstm_autoencoder.py:15`, wrapper `LSTMAutoencoderDetector:56` | `:107` | `:174` | Encoder LSTM → decoder LSTM seeded with final hidden/cell state → linear head; per-window MSE | `hidden_dim=32`, `n_layers=1`, `sequence_length=30`, `epochs=50`, `lr=1e-3`, `batch_size=32`, `threshold_sigma=2.0`, `patience=5` |

### Ensemble and voting

`EnsembleDetector` (`detectors/ensemble.py:12`) is **not** a `BaseDetector` subclass (stated deliberately in its docstring).
It runs every child, stacks labels/scores, and combines:

- `voting` (`:60`): label = at least `min_detectors` children flagged; score = `vote_count / n_detectors`.
- `average` (`:66`): mean score, label = `score > 0.5` (hard-coded, unrelated to child thresholds).
- `max` (`:71`): max score, label = any child flagged.

`Pipeline` instantiates it automatically whenever more than one detector is enabled (`runner.py:73`).

### Preprocessing / scoring / evaluation

`Preprocessor.transform` (`pipeline/preprocessing.py:148`) chains `fill_missing` → optional seasonal decomposition
(residual returned) → optional linear detrend → normalise (`minmax` / `zscore` / `robust`).
**Normalisation is per call**, i.e. computed from the batch being scored, not from a stored training distribution.
Severity is a pure function of the score (`runner.py:31`): ≥0.9 critical, ≥0.7 high, ≥0.5 warning, else low.

There is **no evaluation framework in the repo** — no metric functions, no train/test split, no benchmark harness.
The only precision/recall numbers are printed inside the notebook via sklearn, and the README quotes 0.741/0.710/0.684/0.260.
I built `verification/evaluate_detectors.py` to fill that gap (results in §7).

---

## 3. Training vs inference (and the LSTM specifically)

**Only two of six detectors actually learn anything.** Z-Score and MAD store two scalars; Grubbs, Spectral Residual
and the ensemble are stateless (`fit()` just sets `_fitted = True`). The LSTM autoencoder is the only trained model.

LSTM training (`lstm_autoencoder.py:107`):

1. Global z-normalisation of the series using its own mean/std (including anomalies).
2. `_create_sequences` builds `len(data) − sequence_length + 1` overlapping windows, reshaped to `(N, 30, 1)`.
3. Adam + MSE, batch 32, shuffle on, up to `epochs=50`, early stopping after 5 epochs without improvement.
4. Threshold = `mean(train_reconstruction_error) + 2·std(train_reconstruction_error)`.

Inference (`:174`): recompute window errors, then project each window error onto all points it covers using
**max** pooling, min-max the point scores by the batch maximum, and label `point_error > self._threshold`.

Critical observations about the training lifecycle:

- **Weights are never saved or loaded.** There is no `torch.save`, `torch.load`, `state_dict`, checkpoint,
  joblib or MLflow call anywhere in the repository (verified by search). A fitted model exists only in RAM.
  Every `spectral serve` start, every API request that triggers a fit, and every CLI invocation retrains from scratch.
  The `.gitignore` even lists `*.pt`, `*.pth`, `checkpoints/`, `mlruns/` — paths nothing writes to.
- **No seeding.** `random_state=42` is passed to Isolation Forest, but PyTorch gets no `manual_seed`, so the LSTM is
  non-reproducible. Two identical 10-epoch fits on the same 800 points flagged **65 vs 62** points with different
  thresholds (1.7166 vs 1.7048). The CLI gave 795 vs 825 anomalies for the same file across runs.
- **Device selection ignores Apple GPUs**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` (`:98`).
  On this M4 it always resolves to CPU.
- **Threshold calibration is weak.** On training data the detector flags 16–17% of points at `threshold_sigma=2.0`
  against a nominal ~2.3%. Overlapping windows plus max-pooling inflate the error distribution's right tail, and the
  threshold is never rescaled to the score space actually used for labelling.
- **No train/serve separation in the pipeline.** `Pipeline.detect()` fits on the array it is about to score if the
  pipeline is not already fitted (`runner.py:137`). Batch mode is therefore always in-sample.

---

## 4. Dataset requirements and CSV format

`CSVSource` (`ingestion/csv_loader.py:14`) accepts `.csv`, `.json`, `.parquet`.

Required columns, configurable via `--timestamp-col` / `--value-col` (defaults `timestamp`, `value`):

```csv
timestamp,value,is_anomaly
2026-01-01T00:00:00,0.030471707975443137,0
2026-01-01T00:01:00,0.08820163145652417,0
```

- `timestamp` — anything `pandas.to_datetime` parses; rows are sorted by it. Timestamps are **not** used for
  resampling or gap detection, only carried through to output.
- `value` — numeric, cast with `float()`.
- third column optional. `is_anomaly` is only consumed by `spectral generate` output; the loader ignores it.
  A `series_id` column is supported through the `series_id_col` constructor argument, which the CLI does **not** expose.
- NaNs survive loading and are repaired later by `fill_missing` (`interpolate` by default).
- No maximum length is enforced; the whole file is materialised in memory (`list(source.read())` in `cli/main.py:41`).

`spectral generate` produces a labelled synthetic series from `sine ± seasonal + trend + random_walk` with three
injected anomaly families (point spikes 3–6σ, contextual shifts, collective flatline/reversal runs).

---

## 5. API surface and frontend capabilities

**There is no frontend of any kind.** No HTML/CSS/JS, no `templates/`, no `static/`, no Node/React/Vue/Svelte
project, no `index.html`. `/` returns 404. The only browser-facing output is FastAPI's autogenerated Swagger at `/docs`
and ReDoc, plus a matplotlib notebook. Any "existing frontend" assumed by the brief does not exist and cannot be
preserved — this is greenfield work for Phase 2.

| Method | Path | Handler | Verified behaviour |
|---|---|---|---|
| POST | `/ingest` | `routes.py:26` | Accepts one point or `{points:[…]}`; feeds `detect_streaming`; returns anomalies found. Works. |
| POST | `/detect` | `routes.py:63` | Batch scoring of `{values, timestamps}`. Works. `config` in the request schema (`schemas.py:38`) is **ignored**. |
| GET | `/anomalies` | `routes.py:94` | Filterable query, requires Postgres. **500** without DB. |
| GET | `/status` | `routes.py:129` | Works. `uptime_seconds` is hard-coded `0`. |
| GET | `/metrics` | `routes.py:143` | Pipeline counters + DB aggregates. **500** without DB. |
| WS | `/ws/stream` | `websocket.py:67` | `ping`→`pong` and 30 s `heartbeat` confirmed. No anomaly is ever pushed: `broadcast_anomaly` (`:30`) has **no caller**. |

Streaming semantics: `window_size=100`, `step_size=1`. A point is only scored once 100 have accumulated, so the first
99 points of any connection produce nothing and `points_processed` counts only scored windows (130 ingested → 31 processed).
Measured: 130 points with one +9σ spike at index 120 produced 6 alerts — the spike plus 5 false positives from ordinary
sine values. The spike scored 0.667 and was labelled `warning`, not `critical`, because severity is derived from the
fraction of detectors that agreed (2/3), not from the magnitude of the deviation.

**Wiring gap.** `AnomalyRepository.save_*`, `AlertDispatcher`, `WebsocketConnectionManager.broadcast_anomaly`,
`KafkaSource` and `RedisStreamSource` are implemented and unit-tested but **never constructed or called by the running
application**. `/ingest` neither persists nor broadcasts, so `/anomalies` and `/metrics` can only ever read an empty
table. The README's claim that results are "persisted to TimescaleDB and pushed to a webhook" describes the class
diagram, not the code path. (The README does honestly flag Kafka/Redis as unwired.)

---

## 6. MacBook Air M4 compatibility

| Check | Result |
|---|---|
| Python 3.12.15 on arm64 | ✅ native; `requires-python >=3.11` satisfied |
| Dependency install (`pip install -e ".[dev]"`) | ✅ all wheels arm64-native, **no CUDA/nvidia packages pulled**; torch 2.15.0, numpy 2.5.4, scipy 1.18.1, sklearn 1.9.1, pandas 3.1.1 |
| `torch.backends.mps.is_available()` | ✅ True (`is_built` True) |
| CUDA | not present, not needed |
| LSTM device actually used | ❌ `cpu` — hard-coded `cuda else cpu`, MPS never selected |
| MPS vs CPU on the real workload | **MPS is 34× slower**: 20 forward+backward passes of the repo's LSTM (batch 64, seq 30, hidden 16) took 1.147 s on MPS vs 0.034 s on CPU |
| Peak RSS, LSTM fit on 5,000 points / 20 epochs / hidden 32 | 584 MB total process (~245 MB attributable), 11.6 s |
| Classical detectors | negligible: <0.05 s fit, <0.02 s detect for 5,000 points; whole 3-detector pipeline 0.067 s |

**Recommendation: stay on native CPU.** This is not a compromise — at sequence length 30 and hidden size 32 the model is
tiny, and per-kernel launch overhead on Metal dominates (confirmed above). MPS should only be revisited for
larger hidden sizes, multi-feature inputs, or batch ≫ 1,024, and only after `_device` is fixed to honour MPS explicitly.

Sizing guidance for this architecture: 5k points ≈ 0.6 s/epoch; time is linear in points × epochs and memory is
dominated by the ~330 MB PyTorch runtime, so 50k–100k point series fit comfortably in 24 GB. Keep `hidden_dim ≤ 64`,
`sequence_length ≤ 100`, `batch_size 32–128`, and rely on early stopping. Anything beyond ~500k windows should move to
subsampled training or batched NumPy/NumPy-backed inference.

---

## 7. Verification results

### Test suite (91 collected)

| Run | Command | Result |
|---|---|---|
| Fast (Makefile `test` target) | `pytest -m "not slow and not integration"` | **82 passed, 3 failed**, 16.3 s |
| Slow | `pytest -m slow` | **2 passed, 1 failed**, 3.2 s |
| Integration | `pytest -m integration` | **3 failed** (no Postgres), 2.0 s |
| Full | `pytest` | **84 passed, 7 failed**, 5.0 s |
| Lint (CI job) | `ruff check src/ tests/` | **29 errors, exit 1** (unsorted import blocks, unused imports across tests and source, 1 unused local, 2 long lines) |
| Notebook | `jupyter nbconvert --execute notebooks/demo.ipynb` | **fails at the ensemble cell** |

Failures and their causes — none are environment problems:

1–3. `test_ensemble.py::test_average_method`, `::test_max_method`, `::test_metadata_contains_per_detector` —
`AttributeError: 'EnsembleDetector' object has no attribute 'fit_detect'`. The class deliberately does not inherit
`BaseDetector`, so it never receives the inherited convenience method that its own tests, the README usage example, and
the demo notebook all call. **The notebook stops here**, so the README's quoted ensemble numbers cannot be reproduced
from the shipped code.
4. `test_lstm_autoencoder.py::test_model_reconstructs_normal_data` — asserts <20% of *training* data is flagged;
  observed 26.7%. The threshold rule is miscalibrated (see §3), not a flaky environment issue.
5–7. `test_storage.py` — `role "spectral" does not exist`. Needs TimescaleDB; **Docker is not installed on this machine**,
  so the DB/Redis path is *unverified*, not proven broken.

### Six detectors on labelled data (5,000 points, 141 injected anomalies = 2.82%)

Strict point-wise scoring; tolerant = hit within ±2 points. Raw data: `verification/detector_results.csv`.

| Detector | Flagged | Precision | Recall | F1 | Fit | Detect |
|---|---|---|---|---|---|---|
| Z-Score | 58 | 1.000 | 0.411 | 0.583 | 0.0001 s | <0.0001 s |
| MAD | 42 | 1.000 | 0.298 | 0.459 | 0.0001 s | <0.0001 s |
| Grubbs | 32 | 1.000 | 0.227 | 0.370 | <0.0001 s | 0.0026 s |
| Isolation Forest | 250 | 0.216 | 0.383 | 0.276 | 0.0506 s | 0.0184 s |
| Spectral Residual | 72 | 1.000 | 0.511 | 0.676 | <0.0001 s | 0.0006 s |
| LSTM Autoencoder | 825 | 0.051 | 0.298 | 0.087 | 27.6 s | 0.0749 s |
| Ensemble (`default.yml`, voting ≥2) | 67 | 1.000 | 0.475 | 0.644 | — | 0.067 s |

All six run through the CLI with single-detector configs in `verification/configs/`. Overlap between detectors is low:
Z-Score vs Isolation Forest Jaccard 0.137, Z-Score vs LSTM 0.039, Isolation Forest vs Spectral Residual 0.167 —
they are genuinely complementary, which is the ensemble's justification.

Precision of exactly 1.000 for the statistical detectors is a consequence of easy synthetic spikes (3–6σ) and
max-normalised scoring, not a sign of production readiness. The recall ceiling (~0.51) is structural: roughly half the
injected labels are contextual or collective anomalies that a per-point amplitude test cannot see.

### End-to-end runs (all successful)

- `spectral generate --points 5000 --anomaly-ratio 0.03` → `data/synthetic.csv`, 5,001 lines, 141 labels.
- `spectral detect data/synthetic.csv -c configs/default.yml` → 67 anomalies in a Rich table (also verified `--format json` and `--format csv`).
- `spectral serve` → `/status` 200 with the 3 configured detectors; `/detect` correctly flags an obvious spike.
- All six per-detector CLI configs exit 0 (58 / 42 / 32 / 250 / 71 / 795 anomalies).

### Concrete defects found while running it

1. **5-point spike is invisible.** `POST /detect {"values":[1,2,3,4,50]}` returns **no anomaly**. The outlier inflates
   its own z-score denominator (z ≈ 1.6 < 3), and Isolation Forest is the only detector to flag it — below the 2-of-3
   voting bar. Scores still normalise to 1.0, so "score = 1.0 but label = 0" is possible.
2. **Request history changes results.** The pipeline is a process-wide singleton fitted on whichever request arrived first;
   `stats.points_processed` accumulates across unrelated callers. `/detect` with 5 values after a streaming session
   returned `points_processed: 36`.
3. **`min_detectors` is validated against the detector list at construction** (`ensemble.py:29`). A config that enables
   two detectors but keeps `min_detectors: 3` raises `ValueError` at startup; disabling detectors is not a safe operation.
4. **`config.preprocessing.fill_method` is silently dropped** — `Pipeline.__init__` (`runner.py:51`) never forwards it.
5. **`StorageConfig.enabled=False` is not honoured by the routes**, which depend on the repository unconditionally.
6. **Startup swallows DB failure** with the message "Database init failed (storage disabled)" even when storage is enabled,
   then serves 500s later.
7. **`allow_origins=["*"]` together with `allow_credentials=True`** (`app.py:60-62`) is rejected by browsers; the API also has
   no authentication, rate limiting or request size limits.
8. **`window_size` / `step_size` do not affect batch detection** — `SlidingWindow` is imported (`runner.py:13`) and never used.
9. **Severity is unreachable for `high` in a 3-detector vote**: possible scores are 0, 0.333, 0.667, 1.0, while the
   `high` band starts at 0.7.

---

## 8. Limitations and potential improvements (upstream, not yet acted on)

**Correctness / ML**
- Train/serve separation: fit on a clean baseline window, persist scaler + threshold + weights, score out-of-sample.
- Replace per-batch min-max with a stored scaler; otherwise scores are not comparable across requests or windows.
- Recalibrate the LSTM threshold (per-point error distribution, quantile-based instead of mean+2σ) and verify against a
  labelled hold-out. Better still, evaluate with range-aware metrics (precision/recall over ±k) rather than point equality.
- Seed everything and record the seed in run metadata.
- Add a proper evaluation module: train/val/test split, per-detector and per-anomaly-type metrics, PR curves,
  and a regression baseline so improvements are measurable.
- Batch mode should window (or explicitly document that it is whole-series), and streaming should support `step_size > 1`
  with a documented scoring cadence.

**Architecture / platform**
- Wire the pieces that already exist: persist anomalies in `/ingest`, broadcast over the WebSocket, invoke
  `AlertDispatcher`, expose Kafka/Redis sources behind a worker process.
- Replace the global mutable pipeline with a per-request/per-series context, or a model registry keyed by `series_id`.
- Add model artefacts (versioned weights + scaler + threshold) and a model registry; without it, retraining on every
  boot is both slow and non-reproducible.
- Fix the ensemble contract: subclass `BaseDetector`, implement `fit_detect`, and make `average`'s 0.5 cut and the
  `max` threshold derive from child thresholds.
- Take test integrity seriously: the suite is green only if the three ensemble tests and the notebook are ignored.
  CI currently fails on lint and cannot pass on HEAD.

**Operations**
- Auth, CORS allow-list, request limits, structured logging with request IDs, readiness that distinguishes "API up"
  from "storage reachable".
- TimescaleDB continuous aggregates/retention instead of ad-hoc `GROUP BY` over the raw anomalies table.
- Docker is not installed here; the compose path (app + TimescaleDB + Redis) is untested and must be validated on
  a machine with Docker before we rely on it.

---

## 9. Proposed architecture for Sentinel ML

Goal: keep the upstream engine as the proven detection core, and build the production platform the brief describes
around it — without rewriting the algorithms first.

```
                     ┌──────────────────────────────────────────────────────────┐
   CSV / HTTP / WS   │                     ingestion gateway                      │
   Kafka / Redis ───►│  validation · schema inference · dedupe · tenant/series   │
                     └───────────────┬──────────────────────────────────────────┘
                                     ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │                  feature / window service                │
                     │  resampling · gap fill · stored scaler · SlidingWindow    │
                     └───────────────┬──────────────────────────────────────────┘
                                     ▼
      ┌──────────────────────────────┴───────────────────────────────┐
      ▼                                                              ▼
 ┌──────────────────────┐                                  ┌──────────────────────┐
 │   training service   │  artefacts: weights+scaler+      │   inference service  │
 │  LSTM / IF / stats   │  threshold+metrics (versioned)   │  batch + streaming   │
 │  eval harness, PR    │─────────────────────────────────►│  ensemble policy     │
 │  curves, backtests   │        model registry            │  severity calibration│
 └──────────────────────┘                                  └──────────┬───────────┘
                                                                      ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  results store (anomalies, runs, artefacts) + alert bus  │
                     └───────────────┬──────────────────────────┬───────────────┘
                                     ▼                          ▼
                     ┌──────────────────────────┐   ┌───────────────────────────┐
                     │  React/TS dashboard      │   │  webhook / email / Slack  │
                     │  live stream, series,    │   │  severity routing,        │
                     │  anomaly drill-down,     │   │  cooldown, dedupe         │
                     │  model + pipeline admin  │   └───────────────────────────┘
                     └──────────────────────────┘
```

**Phased delivery** (each phase independently reviewable; nothing from the upstream repo is deleted)

| Phase | Scope | Why first |
|---|---|---|
| 1 (done) | This assessment, reproducible verification harness in `verification/` | Baseline before any change |
| 2 | Repackage upstream as an importable engine (`spectral` stays a dependency/vendored fork), fix the two real bugs (ensemble contract, threshold calibration), add a seeded + persisted artefact pipeline and a held-out evaluation harness | Without trusted metrics and reproducible models, a UI only visualises noise |
| 3 | Service layer: per-series pipeline contexts, Postgres/Timescale persistence actually wired into ingest, WebSocket broadcasting, alert dispatch; tests for each | Turns the existing dead code paths into a working product |
| 4 | Frontend: React + TypeScript + Vite, live WebSocket stream, series browser, anomaly drill-down with score/severity/detector attribution, model registry page | The brief's "new UI" |
| 5 | Ops: auth (OIDC), CORS allow-list, model/version admin, retention, load test on 1M points, Docker/compose validation | Production hardening |

**Deliberate choices proposed for your review**

1. **Upstream stays vendored, untouched** in `Sentinel-ML/`; Sentinel-specific work lands in new top-level packages
   (`Sentinel-ML/src/sentinel/…` or a sibling package) so the origin remains diffable and MIT attribution is intact.
2. **Fix-forward, not rewrite.** The six detectors and the `BaseDetector`/`DetectionResult` contract are good; the gaps
   are calibration, persistence, wiring and evaluation.
3. **CPU-first on Apple Silicon.** MPS stays behind an opt-in flag, enabled only for configurations where it is measured
   to help.
4. **A model artefact is a first-class object** (weights + scaler + threshold + training window + metrics + seed), because
   today none of it exists and every restart retrains.

**Open questions before Phase 2 starts**

1. Is Sentinel ML intended for *your own* time series (single-tenant, local-first) or as a multi-tenant SaaS?
2. Which data domains matter first — infrastructure metrics, IoT/sensors, finance? This determines whether the LSTM
   window model or the statistical fast path is the primary detector.
3. Do you want to keep TimescaleDB (requires Docker/Postgres) or use SQLite/DuckDB for a frictionless local demo?
4. Should the Phase 4 UI be a dashboard-style console (ops monitoring) or an ML-workbench style app (dataset → train →
   evaluate → deploy)? The brief says "advanced ML engineering", which suggests the latter.

---

### Reproduction

Everything in this document is reproducible from the workspace:

```bash
cd Sentinel-ML && .venv/bin/python -m pytest -q -m "not slow and not integration"   # fast suite
Sentinel-ML/.venv/bin/spectral generate --points 5000 --anomaly-ratio 0.03 --output data/synthetic.csv
Sentinel-ML/.venv/bin/python verification/evaluate_detectors.py                     # six-detector table
Sentinel-ML/.venv/bin/python verification/holdout_eval.py                           # hold-out + memory
Sentinel-ML/.venv/bin/python verification/stream_and_ws.py                          # needs a server on :8124
```

Raw outputs: `verification/detector_results.csv`, `verification/api_smoke_output.txt`,
`verification/stream_and_ws_output.txt`, `verification/holdout_output.txt`.

**Not verified:** TimescaleDB/Redis paths (Docker absent), and the README's quoted notebook numbers (notebook cannot execute).
