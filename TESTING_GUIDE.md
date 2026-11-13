# Sentinel ML — step-by-step testing guide

This guide takes a fresh clone to a completed experiment without an external dataset, then shows how
to verify the same behaviour from the terminal and the API. Every command below was executed on
macOS 26 with an Apple M4, 24 GB RAM and Python 3.12.15; PyTorch runs on CPU by design.

The original engine is [peteroyce/spectral](https://github.com/peteroyce/spectral), MIT licensed.
The lab, workbench, persistence layer, evaluation harness and the two fixes (ensemble
`fit_detect()` and LSTM calibration/scoring) are documented in [README.md](README.md) and the
detector docstrings.

---

## 0. Install

```bash
git clone https://github.com/PyDeV1OpEr/Sentinel-ML.git
cd Sentinel-ML

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,workbench]"
```

`[dev]` adds pytest, Ruff and matplotlib; `[workbench]` adds Streamlit. No CUDA, Docker or database
is needed. Check the interpreter and PyTorch:

```bash
python -c "import sys, torch, spectral; print(sys.version.split()[0], torch.__version__, torch.backends.mps.is_available())"
```

MPS may report `True`; Sentinel ML still runs the LSTM on CPU because it is faster for this model
size. CUDA is used automatically when present.

---

## 1. Start the workbench

```bash
streamlit run workbench/app.py
```

Streamlit prints a local URL (normally <http://localhost:8501>). If the port is busy:

```bash
streamlit run workbench/app.py --server.port 8507
```

The sidebar has nine sections:

| Section | Purpose |
|---|---|
| **Overview** | Project scope, algorithm table, model/dataset status, latest benchmark |
| **Dataset laboratory** | Generate synthetic data, upload/map a CSV, load A–H presets |
| **Train** | Configure the LSTM, train with a live progress bar, inspect losses, save an artefact |
| **Detect** | Run inference with a saved artefact, a session model, a classical detector or the ensemble; export CSV |
| **Evaluate** | Precision/recall/F1/FPR/confusion matrix/PR curve on labelled data; inspect individual FP/FN |
| **Compare** | Fit all five classical detectors + ensemble on the 70% split; compare with the supplied LSTM on the held-out 30% |
| **Model registry** | List artefacts, metadata, threshold, training curve; load or delete with confirmation |
| **Experiments** | Local JSONL history; compare two runs; export or clear |
| **Guide** | Short in-app version of this walkthrough |

Training is never triggered by opening a page. Loading a model reads weights, scaler and threshold;
it never calls `fit()`.

---

## 2. First experiment (about two minutes)

1. **Dataset laboratory → Scenario presets → A · Clean seasonal.**
   The overview should show **1,200 points, 0 anomalies, source A · Clean seasonal**.
2. **Train.** Keep the defaults: train fraction 0.70, hidden 32, 1 layer, sequence 30, 100 epochs,
   learning rate 0.001, batch 32, `threshold_sigma` 2.0, holdout 0.20, patience 5, seed 42.
   Click **Train LSTM**.
   - Expected: training completes in roughly 3–5 s on an M4; early stopping typically lands around
     40–60 epochs.
   - Inspect **Training result**: fit time, inference time, threshold (~0.03), calibration source
     `holdout`, peak RSS, epochs run, and the loss curve.
   - The **Training half (in-sample)** should show very few or no flags. The **Held-out test** tab
     is the clean 30% that was never trained on; its labels are all zero, so precision/recall are
     meaningless there — the number to read is the false-positive count.
3. **Save artefact.** Accept the proposed `model_00N` and click **Save to artefacts**.
4. **Dataset laboratory → Scenario presets → H · Mixed unseen.**
   This is a separate seeded dataset with level shifts, seasonal events, noise bursts, missing
   values, spikes and drops.
5. **Evaluate.**
   - Choose **Saved LSTM artefact**, select `model_001`, click **Load and evaluate**.
   - The caption says the artefact was trained on a different dataset, so all points are scored as
     unseen. That is intentional.
   - Read precision, recall, F1, accuracy, FPR, TP/FP/FN, the confusion matrix, PR curve and
     threshold sweep.
   - Under **Where the model is wrong**, use **Inspect an individual error** to zoom into a false
     positive or false negative; download the per-point CSV or metrics JSON.
6. **Compare → Run detector comparison.**
   The classical detectors are fitted on the first 70% of the current (H) dataset; the LSTM artefact
   is evaluated as-is on the held-out 30%. The table and bar chart cover Z-Score, MAD, Grubbs,
   Isolation Forest, Spectral Residual, the voting ensemble and the LSTM. The per-type table shows
   which detector finds level shifts, drifts, bursts, flatlines and spikes.
7. **Detect.** Choose **Saved LSTM artefact**, click **Run detection**, inspect the score/error
   table and download the predictions CSV. **Experiments** now contains the run history.

> **Honest result:** on H · Mixed unseen the saved LSTM scores about F1 0.96. On B/C (dense point
> spikes) it scores 0.17–0.19 because a 30-point window cannot isolate spikes spaced closer than the
> window. Do not conclude that the LSTM is universally best. The benchmark and per-type table exist
> to make that visible.

---

## 3. Train, save, reload and predict from the terminal

The same workflow without the browser:

```bash
# 1. Clean baseline for training
python scripts/generate_dataset.py \
  --output data/lab_clean.csv --points 1200 --anomaly-ratio 0 --seed 42

# 2. Train + save (prints split, timing, threshold, train/held-out metrics)
python scripts/train_lstm.py --input data/lab_clean.csv --epochs 100

# 3. Unseen labelled data for evaluation
python scripts/generate_dataset.py \
  --output data/lab_mixed.csv --points 1200 --anomaly-ratio 0.06 --seed 148

# 4. Reload the artefact and score unseen data
python scripts/predict.py \
  --input data/lab_mixed.csv \
  --model artifacts/lstm_autoencoder/model_001 \
  --output data/predictions.csv

# 5. Evaluate against labels, using the artefact's recorded split when the file matches
python scripts/evaluate_lstm.py \
  --input data/lab_clean.csv \
  --model artifacts/lstm_autoencoder/model_001 \
  --output reports/lstm_evaluation.json \
  --export-csv data/lstm_evaluation_points.csv
```

Artefact layout:

```text
artifacts/lstm_autoencoder/model_001/
├── model.pt            # CPU state_dict, torch.load(weights_only=True)
├── metadata.json       # version, architecture, hyperparameters, seed, fingerprint, history
├── preprocessing.json  # train_mean, train_std
└── calibration.json    # threshold, robust stats, calibration source
```

`predict.py` prints the flagged count and score range. If the CSV has labels it also prints
precision/recall/F1; if not, it reports predictions only. `evaluate_lstm.py` records
`split_matched: false` when the input differs from the training dataset.

To verify persistence, stop the process (or restart the shell), reload the artefact and predict the
same rows again:

```bash
python scripts/predict.py \
  --input examples/example_unlabelled.csv \
  --model artifacts/lstm_autoencoder/model_001 \
  --output /tmp/reload_a.csv
python scripts/predict.py \
  --input examples/example_unlabelled.csv \
  --model artifacts/lstm_autoencoder/model_001 \
  --output /tmp/reload_b.csv
diff /tmp/reload_a.csv /tmp/reload_b.csv   # identical predictions and scores
```

The automated version is `tests/test_lab/test_artifacts.py::test_round_trip_restores_everything` and
`::test_reloaded_predictions_match_within_tolerance`.

---

## 4. Upload your own CSV

Required columns:

| Column | Requirement |
|---|---|
| `timestamp` | Anything `pandas.to_datetime` parses; sorted automatically; duplicate timestamps keep the last |
| `value` | Numeric; missing values are interpolated after the chronological split |
| `is_anomaly` | Optional 0/1; without it, no precision/recall/F1 is shown |
| `anomaly_type` | Optional; carried through to evaluation/export |

The **Upload CSV** tab previews the raw file and asks you to map timestamp, value and optional label
columns explicitly. It never guesses. Invalid timestamps, non-numeric values, empty files, fewer
than 10 rows, or labels other than 0/1 produce a visible error.

Series length constraints:

- training: at least `sequence_length + 2` points (32 with the default window of 30);
- inference: at least `sequence_length` points.

`examples/example_unlabelled.csv` (600 points) is a small file for testing the unlabelled path.

---

## 5. Reading the metrics

Metrics compare predicted labels with `is_anomaly` using **one-to-one matching**: each prediction can
match at most one ground-truth point, within ±2 points by default. This prevents a smeared cluster
from claiming many true positives.

| Metric | Meaning | What a low value usually means |
|---|---|---|
| Precision | TP / (TP + FP) | too many false alarms; inspect the reconstruction-error plot |
| Recall | TP / (TP + FN) | missed events; the event may be inside the model's normal range |
| F1 | harmonic mean | single-number trade-off |
| Accuracy | (TP + TN) / total | dominated by normal points; never use alone |
| FPR | FP / (FP + TN) | share of normal points flagged |
| TP / FP / FN / TN | counts after matching | always check counts before percentages |

`threshold_sigma` is the number of robust standard deviations above the **median window error**.
A point is anomalous only when every window covering it exceeds that threshold. Raising it reduces
both false positives and recall; it is not a quality dial. Tune on a validation series, never on the
held-out test series.

### The LSTM's weak case

With `sequence_length=30`, 36 spikes in 1,200 points means a spike roughly every 33 points. Almost
every 30-point window contains one, so the "every covering window is abnormal" rule flags many normal
neighbours. The benchmark reports recall 1.00 but precision 0.11 on this scenario. Shorter windows
localise spikes better but learn less temporal context. Z-Score, Spectral Residual or the voting
ensemble are the appropriate detectors for isolated point spikes.

---

## 6. Reproduce the A–I benchmark

```bash
python scripts/run_test_scenarios.py --points 1200 --epochs 100
```

It writes:

```text
reports/scenario_results.json          # machine-readable, all detectors and per-type recall
reports/scenario_results.md            # tables quoted in the README
reports/unlabelled_predictions.csv     # scenario I output, no metrics
```

Protocol:

- one clean seasonal baseline; all detectors fitted on its first 70%;
- scenario A = clean held-out 30%;
- scenarios B–H = separate unseen datasets with fixed seeds;
- scenario I = unlabelled CSV: predictions and scores only, never precision/recall/F1;
- one-to-one matching with ±2 points for every detector;
- the LSTM is trained once on the clean baseline and reused; it is never refitted per scenario.

Current held-out results on the committed seed (LSTM F1):

| Scenario | LSTM F1 | Classically best |
|---|---|---|
| A · Clean | — (0 false positives) | Isolation Forest FPR 0.042 |
| B · Isolated spikes | 0.190 | Z-Score/Spectral/Ensemble 0.986 |
| C · Dense spikes | 0.169 | Z-Score/Ensemble 0.962 |
| D · Level shift | **0.932** | Ensemble 0.927 |
| E · Drift | **0.960** | Isolation Forest 0.644 |
| F · Noise burst | **0.970** | Isolation Forest 0.769 |
| G · Flatline | **0.827** | Isolation Forest 0.343 |
| H · Mixed unseen | **0.962** | Isolation Forest 0.635 |
| I · Unlabelled | no metrics | 47/600 flagged (7.83%) |

To use a different unlabelled file:

```bash
python scripts/run_test_scenarios.py --unlabelled-csv path/to/file.csv
# or skip scenario I entirely
python scripts/run_test_scenarios.py --skip-unlabelled
```

---

## 7. Run and inspect the FastAPI service

```bash
uvicorn spectral.api.app:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the generated OpenAPI page, then:

```bash
# Batch detection (a real window with one injected spike)
python - <<'PY'
import json, urllib.request
import numpy as np

values = list(np.sin(np.arange(120) * 0.1))
values[60] = 8.0
request = urllib.request.Request(
    "http://localhost:8000/detect",
    data=json.dumps({"values": values}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    body = json.load(response)
print(body["anomaly_count"], "anomalies")
PY

# Single-point ingestion (streaming path)
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-01-01T00:00:00", "value": 42.5, "series_id": "cpu-load"}' \
  | python -m json.tool

# Status and metrics
curl -s http://localhost:8000/status | python -m json.tool
curl -s http://localhost:8000/metrics | python -m json.tool
```

Storage is disabled by default. `/status` and `/metrics` work; `/anomalies` returns HTTP 503 with a
clear message instead of attempting a database connection. With storage enabled and PostgreSQL
running, `/ingest` persists raw points and anomalies, dispatches alerts, and broadcasts over
`/ws/stream`.

WebSocket smoke test (two terminals, or use a Python client):

```python
import asyncio, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send("ping")
        print(await ws.recv())          # {"type": "pong"}

asyncio.run(main())
```

Validation covers mismatched timestamp/value lengths, non-finite values, oversized batches and empty
requests (HTTP 422). CORS is closed by default; set `SENTINEL_CORS_ORIGINS` to an explicit
comma-separated list if needed. There is no authentication or rate limiting in this local build.

---

## 8. Tests, lint and CI

```bash
pytest -q -m "not integration"      # 187 tests: fast + slow, no database required
pytest -q -m "slow"                 # LSTM + artefact lifecycle
pytest -q -m integration            # 3 storage tests; needs PostgreSQL/TimescaleDB
ruff check src/ tests/ scripts/ workbench/
```

`pytest` with no marker filter also collects the three integration tests; without a database they
fail with `role "spectral" does not exist`. That is the documented environment prerequisite. CI runs
`pytest -m "not integration"` plus Ruff and a compile check on Python 3.12.

To run the database tests:

```bash
docker compose up -d timescaledb
export DATABASE_URL="postgresql+asyncpg://spectral:spectral@localhost:5432/spectral"
pytest -q -m integration
```

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `need at least 32 points` | Series shorter than `sequence_length + 2`; add points or lower the sequence length |
| Threshold looks huge (e.g. 1.9) | The training half contains strong anomalies; retrain on a clean baseline |
| Recall 1.0 with hundreds of flags on spikes | Window-context contamination; use Z-Score/Spectral/ensemble or a shorter window |
| `ArtefactError: checksum` | `model.pt` changed after saving; retrain rather than editing it |
| `split_matched: false` | Evaluation CSV differs from the training file; all points are scored |
| Metrics missing | No `is_anomaly` column, or values outside 0/1 |
| `/anomalies` returns 503 | Storage disabled by default; enable it and set `DATABASE_URL` |
| Port already in use | `streamlit run workbench/app.py --server.port 8507` |
| `role "spectral" does not exist` | Integration tests need the compose PostgreSQL service |

---

## 10. What to look for in a code review

- `src/spectral/detectors/lstm_autoencoder.py` — chronological calibration split, robust threshold,
  minimum-window scoring, seeded training, CPU device choice.
- `src/spectral/detectors/ensemble.py` — `fit_detect()` and voting semantics.
- `src/spectral/lab/artifacts.py` — schema/checksum validation, path-scoped delete, CPU reload.
- `src/spectral/lab/metrics.py` — strict one-to-one matching and per-type recall.
- `src/spectral/api/service.py` — persistence/alert/WebSocket fan-out with explicit optional
  integration failure handling.
- `workbench/app.py` — the UI calls `spectral.lab`; it does not reimplement detection.
- `tests/` — regression tests for each fix, including corruption, repeatability and edge cases.

---

## Licence and attribution

Upstream: [peteroyce/spectral](https://github.com/peteroyce/spectral), MIT, © 2026 Pete Royce.
The [LICENSE](LICENSE) retains the upstream notice and adds the Sentinel ML modification
attribution. Project-specific additions and modifications are described in [README.md](README.md)
and [ASSESSMENT.md](ASSESSMENT.md).
