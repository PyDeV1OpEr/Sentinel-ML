"""Sentinel ML experimental workbench.

Run from the repository root:

    streamlit run workbench/app.py

The UI orchestrates ``spectral.lab``; it does not reimplement detection or
evaluation. It is a local experimentation surface, deliberately separate from
the FastAPI service and the production pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from spectral.lab.artifacts import (
    ArtifactError,
    delete_lstm_artifact,
    list_artifacts,
    load_lstm_artifact,
    next_model_version,
    save_lstm_artifact,
)
from spectral.lab.data import (
    fingerprint_frame,
    impute_values,
    labels_from_frame,
    predictions_frame,
    prepare_frame,
)
from spectral.lab.evaluation import (
    classical_detectors,
    default_ensemble,
    evaluate_loaded_model,
    predict_with_detector,
    run_detector_comparison,
    train_lstm_on_frame,
)
from spectral.lab.history import clear_history, load_history, record_experiment
from spectral.lab.metrics import (
    confusion_matrix,
    false_negative_indices,
    false_positive_indices,
    precision_recall_curve_data,
    recall_by_type,
    threshold_sweep,
)
from spectral.lab.plots import (
    plot_confusion,
    plot_detector_comparison,
    plot_pr_curve,
    plot_reconstruction_errors,
    plot_score_distribution,
    plot_series,
    plot_threshold_sweep,
    plot_training_history,
    plot_type_recall,
)
from spectral.lab.synthetic import (
    ANOMALY_TYPES,
    SyntheticConfig,
    generate_synthetic_timeseries,
)

ARTIFACT_ROOT = Path(
    os.environ.get("SENTINEL_ARTIFACT_ROOT", "artifacts/lstm_autoencoder")
)
BENCHMARK_REPORT = Path("reports/scenario_results.json")

st.set_page_config(
    page_title="Sentinel ML Workbench",
    page_icon="🧪",
    layout="wide",
)

for key, default in (
    ("data_frame", None),
    ("data_info", None),
    ("data_source", None),
    ("data_events", None),
    ("train_result", None),
    ("loaded_model", None),
    ("prediction", None),
    ("prediction_part", None),
    ("comparison", None),
    ("comparison_frame", None),
):
    st.session_state.setdefault(key, default)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _show(fig) -> None:
    st.pyplot(fig, clear_figure=True)


def _store_dataset(
    frame: pd.DataFrame,
    source: str,
    events: pd.DataFrame | None = None,
    info: dict | None = None,
) -> None:
    st.session_state.data_frame = frame
    st.session_state.data_info = info or {}
    st.session_state.data_source = source
    st.session_state.data_events = events
    st.session_state.train_result = None
    st.session_state.prediction = None
    st.session_state.comparison = None


def _metrics_cards(metrics: dict | None, prefix: str = "") -> None:
    if not metrics:
        st.info("No ground-truth labels: precision, recall and F1 are unavailable.")
        return
    columns = st.columns(6)
    columns[0].metric(f"{prefix}Precision", f"{metrics['precision']:.3f}")
    columns[1].metric(f"{prefix}Recall", f"{metrics['recall']:.3f}")
    columns[2].metric(f"{prefix}F1", f"{metrics['f1']:.3f}")
    columns[3].metric(f"{prefix}Accuracy", f"{metrics['accuracy']:.3f}")
    columns[4].metric(f"{prefix}FPR", f"{metrics['false_positive_rate']:.3f}")
    columns[5].metric(
        f"{prefix}TP / FP / FN",
        f"{metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
    )


def _dataset_overview() -> None:
    frame = st.session_state.data_frame
    if frame is None:
        return
    labels = labels_from_frame(frame)
    info = st.session_state.data_info or {}
    columns = st.columns(5)
    columns[0].metric("Points", f"{len(frame):,}")
    columns[1].metric(
        "Anomalies",
        f"{int(labels.sum()):,}" if labels is not None else "unlabelled",
    )
    columns[2].metric("Missing", f"{int(frame['is_missing'].sum()):,}")
    columns[3].metric("Duplicates removed", f"{int(info.get('duplicates', 0)):,}")
    columns[4].metric("Source", st.session_state.data_source or "—")
    _show(plot_series(frame, title=f"Dataset — {st.session_state.data_source}"))
    st.dataframe(frame.head(200), hide_index=True)
    if st.session_state.data_events is not None and len(st.session_state.data_events):
        with st.expander("Ground-truth events (exact spans)"):
            st.dataframe(st.session_state.data_events, hide_index=True)


def _artifact_rows() -> pd.DataFrame:
    rows = []
    for path in list_artifacts(ARTIFACT_ROOT):
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        architecture = metadata.get("architecture", {})
        training = metadata.get("training", {})
        rows.append(
            {
                "version": metadata.get("model_version", path.name),
                "created": str(metadata.get("created_at", ""))[:19],
                "sequence": architecture.get("sequence_length"),
                "hidden": architecture.get("hidden_dim"),
                "epochs": training.get("epochs"),
                "seed": training.get("seed"),
                "train_points": training.get("train_points"),
                "threshold": metadata.get("threshold")
                or _read_calibration(path, "threshold"),
                "dataset": str(metadata.get("dataset", {}).get("fingerprint_sha256", ""))[:12],
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _read_calibration(path: Path, key: str):
    try:
        payload = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
        return payload.get(key)
    except (OSError, json.JSONDecodeError):
        return None


def _load_artifact_into_session(path: Path) -> None:
    loaded = load_lstm_artifact(path)
    st.session_state.loaded_model = loaded
    st.session_state.prediction = None


def _session_detector():
    loaded = st.session_state.loaded_model
    return loaded.detector if loaded is not None else None


def _model_source(label: str = "Model source"):
    """Radio selecting a fitted LSTM, with an explicit load step."""
    train_result = st.session_state.train_result
    loaded = st.session_state.loaded_model
    options = []
    if train_result is not None:
        options.append("Session-trained LSTM")
    if loaded is not None:
        options.append(f"Loaded artefact ({loaded.model_version})")
    options.append("Load artefact from disk")
    choice = st.radio(label, options, horizontal=True)

    if choice == "Session-trained LSTM":
        return train_result.detector if train_result else None
    if choice.startswith("Loaded artefact") and loaded is not None:
        return loaded.detector

    artifacts = list_artifacts(ARTIFACT_ROOT)
    if not artifacts:
        st.warning(f"No artefacts under {ARTIFACT_ROOT}. Train and save one first.")
        return None
    default_index = 0
    if loaded is not None and loaded.path in artifacts:
        default_index = artifacts.index(loaded.path)
    selected = st.selectbox(
        "Artefact",
        artifacts,
        index=default_index,
        format_func=lambda path: path.name,
    )
    if loaded is None or loaded.path != selected:
        if st.button("Load artefact", type="primary"):
            try:
                _load_artifact_into_session(selected)
                st.success(f"Loaded {selected.name}; loading never retrains.")
                st.rerun()
            except ArtifactError as exc:
                st.error(str(exc))
        return None
    return loaded.detector


def _prediction_from_detector(detector, frame: pd.DataFrame):
    if detector is None:
        return None
    values = impute_values(frame["value"])
    if len(values) < getattr(detector, "sequence_length", 2):
        raise ValueError("series is shorter than the detector's minimum length")
    result = detector.detect(values)
    return {
        "frame": frame,
        "values": values,
        "result": result,
        "errors": result.scores,
        "metrics": None,
    }


def _record(kind: str, *, detector: str, metrics: dict | None, duration=None, notes=""):
    frame = st.session_state.data_frame
    try:
        record_experiment(
            kind=kind,
            dataset_source=st.session_state.data_source,
            dataset_fingerprint=fingerprint_frame(frame) if frame is not None else None,
            detector=detector,
            model_version=(
                st.session_state.loaded_model.model_version
                if st.session_state.loaded_model is not None
                else None
            ),
            metrics=metrics or {},
            duration_seconds=duration,
            notes=notes,
        )
    except OSError as exc:  # history must never break an experiment
        st.warning(f"Could not write experiment history: {exc}")


# --------------------------------------------------------------------------- #
# 1. overview
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    st.title("Sentinel ML")
    st.caption(
        "Time-series anomaly detection and ML experimentation platform — "
        "six algorithms, a trainable LSTM autoencoder, model persistence and "
        "honest held-out evaluation."
    )
    frame = st.session_state.data_frame
    columns = st.columns(4)
    columns[0].metric("Algorithms", "6 + ensemble")
    columns[1].metric("Dataset", f"{len(frame):,}" if frame is not None else "none")
    columns[2].metric(
        "Model",
        st.session_state.loaded_model.model_version
        if st.session_state.loaded_model is not None
        else ("session-trained" if st.session_state.train_result else "not trained"),
    )
    columns[3].metric("Saved artefacts", len(list_artifacts(ARTIFACT_ROOT)))

    st.subheader("Detection engine")
    st.markdown(
        """
| Algorithm | Family | Learns from data | Best suited to |
|---|---|---|---|
| Z-Score | Statistical | mean / std | roughly Gaussian data, clear spikes |
| MAD | Statistical | median / MAD | baselines that already contain outliers |
| Grubbs' test | Statistical | none (iterative test) | small samples, formal outlier testing |
| Isolation Forest | Tree ensemble | lagged windows | mixed point/contextual anomalies |
| Spectral Residual (FFT) | Signal processing | none | periodic signals, spectral shifts |
| LSTM Autoencoder | Deep learning | sequence reconstruction | complex temporal events |
| Ensemble | Meta | constituent detectors | precision-first consensus |
        """
    )

    st.subheader("Quick start")
    st.markdown(
        """
1. **Data laboratory** → load preset **A · Clean seasonal**.
2. **Train** → keep the defaults, train, inspect the loss curve and held-out tab.
3. **Model registry** → save the artefact, then reload it in a later session.
4. **Data laboratory** → load preset **H · Mixed unseen**.
5. **Evaluate** → inspect precision, recall, the confusion matrix and individual FP/FN points.
6. **Compare** → run all six detectors and the ensemble on the same split.
        """
    )

    if BENCHMARK_REPORT.is_file():
        st.subheader("Latest reproducible benchmark")
        try:
            payload = json.loads(BENCHMARK_REPORT.read_text(encoding="utf-8"))
            rows = []
            for scenario in payload.get("scenarios", []):
                for row in scenario.get("rows", []):
                    if row.get("detector") == "LSTM Autoencoder":
                        rows.append(
                            {
                                "scenario": scenario["key"].replace("_", " "),
                                "precision": row.get("test_precision"),
                                "recall": row.get("test_recall"),
                                "f1": row.get("test_f1"),
                                "fpr": row.get("test_fpr"),
                                "fp": row.get("test_fp"),
                                "fn": row.get("test_fn"),
                            }
                        )
            if rows:
                table = pd.DataFrame(rows)
                for column in ("precision", "recall", "f1", "fpr"):
                    table[column] = pd.to_numeric(
                        table[column], errors="coerce"
                    ).round(3)
                st.dataframe(table, hide_index=True)
                st.caption(
                    "LSTM Autoencoder on scenarios A–H. Generated by "
                    "`python scripts/run_test_scenarios.py`; the full report, "
                    "including every detector, is `reports/scenario_results.md`."
                )
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Benchmark report could not be read: {exc}")
    else:
        st.info(
            "No benchmark report yet. Run `python scripts/run_test_scenarios.py` "
            "to generate `reports/scenario_results.json`."
        )


# --------------------------------------------------------------------------- #
# 2. dataset laboratory
# --------------------------------------------------------------------------- #
def render_data() -> None:
    st.header("2 · Dataset laboratory")
    st.caption(
        "Generate a labelled synthetic series, upload a time-series CSV, or load "
        "a reproducible scenario preset. Column mapping is explicit."
    )

    generate_tab, upload_tab, presets_tab = st.tabs(
        ["Generate synthetic", "Upload CSV", "Scenario presets"]
    )

    with generate_tab:
        with st.form("generate_form"):
            row1 = st.columns(4)
            n_points = row1[0].number_input("Points", 200, 20000, 1200, step=100)
            seed = row1[1].number_input("Seed", 0, 100000, 42)
            freq = row1[2].selectbox("Frequency", ["1h", "15min", "1min", "1D"], index=0)
            start = row1[3].text_input("Start", "2026-01-01T00:00:00")
            row2 = st.columns(4)
            trend = row2[0].number_input("Trend (total change)", value=0.0, step=0.5)
            period = row2[1].number_input("Seasonal period (points)", 2, 1000, 24)
            amplitude = row2[2].number_input("Seasonal amplitude", 0.0, 10.0, 1.0, step=0.1)
            noise = row2[3].number_input("Noise σ", 0.0, 5.0, 0.1, step=0.05)
            row3 = st.columns(2)
            ratio = row3[0].slider("Anomaly fraction", 0.0, 0.4, 0.05, step=0.01)
            types = row3[1].multiselect(
                "Anomaly types",
                list(ANOMALY_TYPES),
                default=list(ANOMALY_TYPES),
            )
            submitted = st.form_submit_button("Generate dataset", type="primary")
        if submitted:
            try:
                config = SyntheticConfig(
                    n_points=int(n_points),
                    start=start,
                    freq=freq,
                    trend=float(trend),
                    seasonal_period=int(period),
                    seasonal_amplitude=float(amplitude),
                    noise=float(noise),
                    anomaly_ratio=float(ratio),
                    anomaly_types=tuple(types),
                    seed=int(seed),
                )
                dataset = generate_synthetic_timeseries(config)
                frame, _ = prepare_frame(dataset.frame)
                _store_dataset(
                    frame,
                    f"synthetic seed={seed}",
                    dataset.events_frame(),
                    {"duplicates": 0, "raw_rows": len(dataset.frame)},
                )
                st.success(
                    f"Generated {len(frame)} points with "
                    f"{int(dataset.labels.sum())} anomalous points."
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    with upload_tab:
        uploaded = st.file_uploader("CSV file", type=["csv"], key="data_upload")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Cannot parse CSV: {exc}")
                raw = None
            if raw is not None and len(raw):
                st.dataframe(raw.head(50), hide_index=True)
                columns = list(raw.columns)
                mapping = st.columns(3)
                timestamp_col = mapping[0].selectbox("Timestamp column", columns, key="ts_col")
                value_col = mapping[1].selectbox("Value column", columns, key="val_col")
                label_col = mapping[2].selectbox(
                    "Label column (optional)", ["(none)"] + columns, key="label_col"
                )
                if st.button("Use this CSV", type="primary", key="use_upload"):
                    try:
                        frame, has_labels = prepare_frame(
                            raw,
                            timestamp_col=timestamp_col,
                            value_col=value_col,
                            label_col=None if label_col == "(none)" else label_col,
                        )
                        duplicates = int(
                            raw.duplicated(subset=[timestamp_col]).sum()
                        )
                        _store_dataset(
                            frame,
                            f"upload:{uploaded.name}",
                            None,
                            {
                                "duplicates": duplicates,
                                "raw_rows": len(raw),
                                "has_labels": has_labels,
                            },
                        )
                        st.success(
                            f"Loaded {len(frame)} points; "
                            f"labels: {'yes' if has_labels else 'no'}."
                        )
                    except Exception as exc:
                        st.error(f"Cannot use this CSV: {exc}")
        st.caption(
            "Required: `timestamp` and numeric `value`. Optional: `is_anomaly` "
            "(0/1). A CSV without labels is supported: predictions and scores are "
            "shown, but precision/recall/F1 are deliberately withheld."
        )

    with presets_tab:
        st.caption("Fixed, reproducible configurations matching the benchmark scenarios.")
        presets = {
            "A · Clean seasonal": dict(anomaly_ratio=0.0),
            "B · Isolated spikes": dict(anomaly_ratio=0.03, anomaly_types=("spike",)),
            "C · Dense spikes": dict(anomaly_ratio=0.08, anomaly_types=("spike", "drop")),
            "D · Level shift": dict(anomaly_ratio=0.06, anomaly_types=("level_shift",)),
            "E · Gradual drift": dict(anomaly_ratio=0.06, anomaly_types=("drift",)),
            "F · Noise burst": dict(anomaly_ratio=0.06, anomaly_types=("noise_burst",)),
            "G · Flatline": dict(anomaly_ratio=0.03, anomaly_types=("flatline",)),
            "H · Mixed unseen": dict(anomaly_ratio=0.06),
        }
        names = list(presets)
        for start in range(0, len(names), 4):
            columns = st.columns(4)
            for offset, name in enumerate(names[start : start + 4]):
                index = start + offset
                if columns[offset].button(name, key=f"preset_{index}"):
                    try:
                        config = SyntheticConfig(
                            n_points=1200, seed=42 + index, **presets[name]
                        )
                        dataset = generate_synthetic_timeseries(config)
                        frame, _ = prepare_frame(dataset.frame)
                        _store_dataset(
                            frame,
                            name,
                            dataset.events_frame(),
                            {"duplicates": 0, "raw_rows": len(dataset.frame)},
                        )
                        st.success(f"Loaded preset {name}.")
                    except Exception as exc:
                        st.error(f"Preset failed: {exc}")

    st.divider()
    _dataset_overview()


# --------------------------------------------------------------------------- #
# 3. training
# --------------------------------------------------------------------------- #
def render_train() -> None:
    st.header("3 · Model training")
    frame = st.session_state.data_frame
    if frame is None:
        st.warning("Load or generate a dataset on the Dataset laboratory page first.")
        return
    st.caption(
        f"Dataset: {len(frame)} points from {st.session_state.data_source}. "
        "The split is chronological. The LSTM trains on the training half; its "
        "calibration holdout lives inside that half. The test half is untouched."
    )

    with st.form("train_form"):
        columns = st.columns(4)
        train_fraction = columns[0].slider("Training fraction", 0.5, 0.9, 0.7, 0.05)
        hidden_dim = columns[1].number_input("Hidden dim", 4, 128, 32, step=4)
        n_layers = columns[2].number_input("LSTM layers", 1, 3, 1)
        sequence_length = columns[3].number_input("Sequence length", 5, 120, 30)
        columns = st.columns(4)
        epochs = columns[0].number_input("Epochs", 1, 300, 100)
        lr = columns[1].selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
        batch_size = columns[2].number_input("Batch size", 8, 256, 32, step=8)
        threshold_sigma = columns[3].number_input(
            "threshold_sigma", 0.5, 6.0, 2.0, step=0.25
        )
        columns = st.columns(4)
        calibration_fraction = columns[0].slider(
            "Calibration holdout", 0.1, 0.4, 0.2, 0.05
        )
        patience = columns[1].number_input("Early-stop patience", 1, 50, 5)
        seed = columns[2].number_input("Seed", 0, 100000, 42)
        submitted = st.form_submit_button("Train LSTM", type="primary")

    labels = labels_from_frame(frame)
    train_cut = int(len(frame) * float(train_fraction))
    if labels is not None and labels[:train_cut].sum() > 0:
        st.warning(
            "The training half contains labelled anomalies. An autoencoder can "
            "learn to reconstruct them, which inflates apparent quality. Train on "
            "a clean baseline (preset A), save the artefact, then evaluate on an "
            "anomalous dataset."
        )

    if submitted:
        params = {
            "hidden_dim": int(hidden_dim),
            "n_layers": int(n_layers),
            "sequence_length": int(sequence_length),
            "epochs": int(epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "threshold_sigma": float(threshold_sigma),
            "patience": int(patience),
            "seed": int(seed),
            "calibration_fraction": float(calibration_fraction),
        }
        progress = st.progress(0.0, text="Starting training…")

        def _progress(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
            progress.progress(
                min(epoch / total, 1.0),
                text=(
                    f"Epoch {epoch}/{total} — train {train_loss:.5f} / "
                    f"validation {val_loss:.5f}"
                ),
            )

        try:
            result = train_lstm_on_frame(
                frame,
                train_fraction=float(train_fraction),
                params=params,
                progress_callback=_progress,
            )
            st.session_state.train_result = result
            st.session_state.prediction = None
            st.session_state.comparison = None
            progress.progress(1.0, text="Training complete")
            st.success("Training complete.")
            _record(
                "train",
                detector="LSTM Autoencoder",
                metrics=result.test_metrics,
                duration=result.fit_seconds,
                notes=f"epochs={params['epochs']}, seq={params['sequence_length']}",
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Training failed: {exc}")

    result = st.session_state.train_result
    if result is None:
        return

    st.divider()
    st.subheader("Training result")
    columns = st.columns(6)
    columns[0].metric("Fit time", f"{result.fit_seconds:.2f}s")
    columns[1].metric("Inference", f"{result.inference_seconds:.3f}s")
    columns[2].metric("Threshold", f"{result.detector._threshold:.4f}")
    columns[3].metric("Calibration", result.detector._calibration_source or "—")
    columns[4].metric("Peak RSS", f"{result.peak_memory_mb:.0f} MiB")
    columns[5].metric("Epochs run", len(result.detector._training_history))
    _show(
        plot_training_history(
            result.detector._training_history,
            title="Training and calibration-holdout loss",
        )
    )

    train_tab, test_tab = st.tabs(["Training half (in-sample)", "Held-out test"])
    with train_tab:
        _metrics_cards(result.train_metrics, prefix="Train ")
        _show(
            plot_series(
                result.train_frame,
                predicted=result.train_result.labels,
                title="Training half — actual vs predicted",
            )
        )
    with test_tab:
        _metrics_cards(result.test_metrics, prefix="Held-out ")
        _show(
            plot_series(
                result.test_frame,
                predicted=result.test_result.labels,
                title="Held-out half — actual vs predicted",
            )
        )
        _show(
            plot_reconstruction_errors(
                result.test_frame,
                result.test_errors,
                threshold=result.detector._threshold,
                predicted=result.test_result.labels,
                title="Held-out reconstruction error (best covering window)",
            )
        )

    st.subheader("Save artefact")
    with st.form("save_form"):
        default_version = next_model_version(ARTIFACT_ROOT)
        version = st.text_input("Model version", value=default_version)
        save = st.form_submit_button("Save to artefacts")
    if save:
        try:
            path = save_lstm_artifact(
                result.detector,
                ARTIFACT_ROOT,
                model_version=version.strip() or None,
                dataset_frame=frame,
                train_points=len(result.train_frame),
                evaluation={"train": result.train_metrics, "test": result.test_metrics},
                fit_seconds=result.fit_seconds,
            )
            st.success(f"Saved {path}")
            st.code(
                f"python scripts/evaluate_lstm.py --input <data.csv> --model {path}"
            )
            _record(
                "save",
                detector="LSTM Autoencoder",
                metrics=result.test_metrics,
                notes=f"artefact={path.name}",
            )
        except ArtifactError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Save failed: {exc}")


# --------------------------------------------------------------------------- #
# 4. detection
# --------------------------------------------------------------------------- #
def render_detect() -> None:
    st.header("4 · Anomaly detection")
    st.caption(
        "Inference only. Training and threshold calibration happen on the Train "
        "page; loading an artefact never retrains it."
    )
    frame = st.session_state.data_frame
    if frame is None:
        st.warning("Load a dataset on the Dataset laboratory page first.")
        return

    source = st.radio(
        "Detector",
        ["Saved LSTM artefact", "Session-trained LSTM", "Classical detector", "Ensemble"],
        horizontal=True,
    )
    detector = None
    label = "detector"
    try:
        if source == "Saved LSTM artefact":
            detector = _model_source("Artefact source")
            label = "LSTM Autoencoder"
        elif source == "Session-trained LSTM":
            result = st.session_state.train_result
            if result is None:
                st.info("Train an LSTM on the Train page first.")
                return
            detector = result.detector
            label = "LSTM Autoencoder (session)"
        else:
            names = list(classical_detectors())
            if source == "Ensemble":
                detector = default_ensemble()
                label = "Ensemble"
                st.caption(
                    "Voting ensemble of Z-Score, Isolation Forest and Spectral "
                    "Residual; two of three must agree."
                )
            else:
                name = st.selectbox("Classical detector", names)
                detector = classical_detectors()[name]
                label = name
            st.caption(
                "The detector is fitted on the first 70% of this dataset and "
                "scored on all of it. The first 70% is therefore in-sample; use "
                "Compare for a held-out comparison."
            )
    except ArtifactError as exc:
        st.error(str(exc))
        return
    if detector is None:
        return

    if st.button("Run detection", type="primary"):
        try:
            if label.startswith("LSTM") or source == "Saved LSTM artefact":
                prediction = predict_with_detector(detector, frame)
                st.session_state.prediction = prediction
                st.session_state.prediction_part = "all"
            else:
                cut = int(round(len(frame) * 0.7))
                detector.fit(frame["value"].iloc[:cut].interpolate().to_numpy(dtype=float))
                prediction = _prediction_from_detector(detector, frame)
                st.session_state.prediction = prediction
                st.session_state.prediction_part = "all"
            _record(
                "detect",
                detector=label,
                metrics=st.session_state.prediction.metrics,
                notes=f"points={len(frame)}",
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

    prediction = st.session_state.prediction
    if prediction is None:
        return
    result = prediction.result
    st.divider()
    columns = st.columns(4)
    columns[0].metric("Points", f"{len(prediction.frame):,}")
    columns[1].metric("Flagged", f"{result.anomaly_count:,}")
    columns[2].metric("Flag rate", f"{result.anomaly_ratio:.2%}")
    columns[3].metric("Threshold", f"{result.threshold:.4f}")

    _show(
        plot_series(
            prediction.frame,
            predicted=result.labels,
            title=f"{label} — predictions on the loaded dataset",
        )
    )
    errors = prediction.errors if label.startswith("LSTM") else None
    table = predictions_frame(
        prediction.frame,
        result.labels,
        result.scores,
        errors,
    ).sort_values("score", ascending=False)
    st.dataframe(table.head(200), hide_index=True)
    st.download_button(
        "Download predictions CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="sentinel_predictions.csv",
        mime="text/csv",
    )
    if prediction.metrics:
        _metrics_cards(prediction.metrics)
    else:
        st.info(
            "No ground-truth labels on this dataset: scores and predictions are "
            "shown, but precision, recall and F1 are not reported."
        )


# --------------------------------------------------------------------------- #
# 5. evaluation
# --------------------------------------------------------------------------- #
def _eval_parts(frame: pd.DataFrame):
    source = st.radio(
        "Evaluation source",
        ["Session-trained LSTM", "Saved LSTM artefact", "Classical detector", "Ensemble"],
        horizontal=True,
    )
    tolerance = st.slider("Matching tolerance (points)", 0, 10, 2)

    if source == "Session-trained LSTM":
        result = st.session_state.train_result
        if result is None:
            st.info("Train an LSTM on the Train page first.")
            return None
        trained_frame = pd.concat(
            [result.train_frame, result.test_frame], ignore_index=True
        )
        same_dataset = fingerprint_frame(frame) == fingerprint_frame(trained_frame)
        if same_dataset:
            split = st.radio(
                "Data to evaluate",
                ["Held-out test", "Training half (in-sample)", "All data"],
                horizontal=True,
            )
            chosen = {
                "Held-out test": result.test_frame,
                "Training half (in-sample)": result.train_frame,
                "All data": frame,
            }[split]
        else:
            split = "All data"
            chosen = frame
            st.info(
                "The session-trained model was fitted on a different dataset. "
                "It is scored on the currently loaded dataset as unseen data."
            )
        return {
            "name": "LSTM Autoencoder",
            "parts": {
                split: predict_with_detector(result.detector, chosen, tolerance=tolerance)
            },
            "detector": result.detector,
            "tolerance": tolerance,
        }

    if source == "Saved LSTM artefact":
        loaded = st.session_state.loaded_model
        artifacts = list_artifacts(ARTIFACT_ROOT)
        if not artifacts:
            st.warning(f"No artefacts under {ARTIFACT_ROOT}.")
            return None
        default_index = 0
        if loaded is not None and loaded.path in artifacts:
            default_index = artifacts.index(loaded.path)
        selected = st.selectbox(
            "Artefact",
            artifacts,
            index=default_index,
            format_func=lambda p: p.name,
        )
        if loaded is None or loaded.path != selected:
            if st.button("Load and evaluate", type="primary"):
                try:
                    _load_artifact_into_session(selected)
                    st.rerun()
                except ArtifactError as exc:
                    st.error(str(exc))
            return None
        parts = evaluate_loaded_model(loaded, frame, tolerance=tolerance)
        split_note = (
            "Artefact fingerprint matched this dataset; train/test split restored."
            if parts.get("split_matched")
            else "Artefact was trained on a different dataset; scoring all points as unseen."
        )
        st.caption(split_note)
        return {
            "name": "LSTM Autoencoder",
            "parts": {
                key: value
                for key, value in parts.items()
                if key in {"train", "test", "all"}
            },
            "detector": loaded.detector,
            "tolerance": tolerance,
        }

    names = list(classical_detectors())
    if source == "Ensemble":
        detector = default_ensemble()
        name = "Ensemble"
    else:
        name = st.selectbox("Classical detector", names)
        detector = classical_detectors()[name]
    cut = int(round(len(frame) * 0.7))
    train = frame.iloc[:cut].reset_index(drop=True)
    test = frame.iloc[cut:].reset_index(drop=True)
    detector.fit(train["value"].interpolate().to_numpy(dtype=float))
    return {
        "name": name,
        "parts": {"test": predict_with_detector(detector, test, tolerance=tolerance)},
        "detector": detector,
        "tolerance": tolerance,
    }


def render_evaluate() -> None:
    st.header("5 · Model evaluation")
    frame = st.session_state.data_frame
    if frame is None:
        st.warning("Load a dataset on the Dataset laboratory page first.")
        return

    selection = _eval_parts(frame)
    if selection is None:
        return
    name = selection["name"]
    parts = selection["parts"]
    tolerance = selection["tolerance"]

    part_name = "test" if "test" in parts else next(iter(parts))
    prediction = parts[part_name]
    st.caption(
        f"{name}; part: **{part_name}**; one-to-one matching within ±{tolerance} points."
    )
    _metrics_cards(prediction.metrics)
    labels = labels_from_frame(prediction.frame)
    if labels is None:
        st.info(
            "This dataset has no ground-truth labels. Predictions and scores are "
            "shown below, but no precision/recall/F1 is invented."
        )
        _show(
            plot_series(
                prediction.frame,
                predicted=prediction.result.labels,
                title=f"{name} — predictions",
            )
        )
        return

    columns = st.columns(3)
    with columns[0]:
        _show(plot_confusion(confusion_matrix(labels, prediction.result.labels)))
    with columns[1]:
        pr = precision_recall_curve_data(labels, prediction.result.scores)
        _show(plot_pr_curve(pr["precision"], pr["recall"], pr["average_precision"]))
    with columns[2]:
        # The LSTM's calibrated threshold is in raw MSE units, while its
        # DetectionResult scores are normalised. Sweep the same raw errors the
        # threshold applies to so the marker is meaningful.
        if name.startswith("LSTM"):
            sweep_values = prediction.errors
            selected_threshold = prediction.result.threshold
        elif name == "Ensemble":
            sweep_values = prediction.result.scores
            selected_threshold = prediction.result.threshold
        else:
            sweep_values = prediction.result.scores
            selected_threshold = None
        rows = threshold_sweep(
            labels, sweep_values, tolerance=tolerance, n_thresholds=40
        )
        _show(
            plot_threshold_sweep(rows, selected_threshold=selected_threshold)
        )

    st.subheader("Where the model is wrong")
    _show(
        plot_series(
            prediction.frame,
            predicted=prediction.result.labels,
            title=f"{name} — actual vs predicted",
        )
    )
    if not name.startswith("LSTM") and name != "Ensemble":
        _show(
            plot_score_distribution(
                prediction.result.scores,
                threshold=None,
                labels=labels,
            )
        )
    else:
        _show(
            plot_reconstruction_errors(
                prediction.frame,
                prediction.errors,
                threshold=prediction.result.threshold,
                predicted=prediction.result.labels,
                title="Reconstruction error (best covering window)",
            )
        )
        _show(
            plot_score_distribution(
                prediction.errors,
                prediction.result.threshold,
                labels=labels,
            )
        )

    fp = false_positive_indices(
        labels, prediction.result.labels, tolerance=tolerance
    )
    fn = false_negative_indices(
        labels, prediction.result.labels, tolerance=tolerance
    )
    options = [f"False positive @ {i}" for i in fp[:100]] + [
        f"False negative @ {i}" for i in fn[:100]
    ]
    if options:
        choice = st.selectbox("Inspect an individual error", options)
        index = int(choice.rsplit("@", 1)[1])
        low, high = max(0, index - 40), min(len(prediction.frame), index + 41)
        window = prediction.frame.iloc[low:high].reset_index(drop=True)
        _show(
            plot_series(
                window,
                predicted=prediction.result.labels[low:high],
                title=choice,
            )
        )
        st.caption(
            f"Window [{low}, {high}) around index {index}. "
            "Red circles are ground truth; orange crosses are predictions."
        )
    elif labels.sum() == 0:
        st.success("No labelled anomalies in this part; nothing to review.")
    else:
        st.success("No false positives or false negatives within the tolerance.")

    table = predictions_frame(
        prediction.frame,
        prediction.result.labels,
        prediction.result.scores,
        prediction.errors if name.startswith("LSTM") else None,
    ).sort_values("score", ascending=False)
    st.dataframe(table.head(200), hide_index=True)
    columns = st.columns(2)
    columns[0].download_button(
        "Download per-point CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="sentinel_predictions.csv",
        mime="text/csv",
    )
    if prediction.metrics:
        columns[1].download_button(
            "Download metrics JSON",
            json.dumps(
                {k: v for k, v in prediction.metrics.items() if k != "matched_pairs"},
                indent=2,
                default=str,
            ).encode("utf-8"),
            file_name="sentinel_metrics.json",
            mime="application/json",
        )
    _record(
        "evaluate",
        detector=name,
        metrics=prediction.metrics,
        notes=f"part={part_name}, tolerance={tolerance}",
    )


# --------------------------------------------------------------------------- #
# 6. comparison
# --------------------------------------------------------------------------- #
def render_compare() -> None:
    st.header("6 · Detector comparison")
    frame = st.session_state.data_frame
    if frame is None:
        st.warning("Load a dataset on the Dataset laboratory page first.")
        return

    result = st.session_state.train_result
    loaded = st.session_state.loaded_model
    if result is not None:
        lstm_detector = result.detector
        lstm_source = "session-trained LSTM"
    elif loaded is not None:
        lstm_detector = loaded.detector
        lstm_source = f"loaded artefact {loaded.model_version}"
    else:
        lstm_detector = None
        lstm_source = None

    st.caption(
        "All five classical detectors and the ensemble are fitted on the first "
        "70% of the currently loaded dataset; the remaining 30% is held out. "
        "The LSTM is not refitted here — it is used as supplied. Each predicted "
        "point can match at most one ground-truth anomaly."
    )
    if lstm_source:
        st.caption(f"LSTM source: {lstm_source}.")
    else:
        st.info(
            "No LSTM is available yet. Run the comparison below to train the "
            "default LSTM on the current training split; it is then evaluated "
            "on the held-out 30%."
        )

    tolerance = st.slider("Matching tolerance (points)", 0, 10, 2)
    if st.button("Run detector comparison", type="primary"):
        try:
            if lstm_detector is None:
                with st.spinner("Training the default LSTM on the training split…"):
                    trained = train_lstm_on_frame(
                        frame, tolerance=tolerance
                    )
                    st.session_state.train_result = trained
                    lstm_detector = trained.detector
            cut = int(round(len(frame) * 0.7))
            train_frame = frame.iloc[:cut].reset_index(drop=True)
            test_frame = frame.iloc[cut:].reset_index(drop=True)
            with st.spinner("Fitting five detectors and the ensemble…"):
                comparison = run_detector_comparison(
                    train_frame,
                    test_frame,
                    lstm_detector=lstm_detector,
                    tolerance=tolerance,
                )
            st.session_state.comparison = comparison
            st.session_state.comparison_frame = test_frame
            _record(
                "compare",
                detector="all",
                metrics={
                    row["detector"]: row["test_f1"]
                    for _, row in comparison.rows.iterrows()
                },
                notes=f"tolerance={tolerance}",
            )
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    comparison = st.session_state.comparison
    if comparison is None:
        return

    st.subheader("Held-out metrics")
    display = comparison.rows.copy()
    for column in display.columns:
        if display[column].dtype.kind == "f":
            display[column] = display[column].round(4)
    st.dataframe(display, hide_index=True)
    _show(plot_detector_comparison(comparison.rows, metric="test_f1"))
    st.caption(
        "One-to-one matching: one predicted point can satisfy at most one actual "
        "anomaly. `fit_seconds` is `NaN` for the LSTM because the supplied "
        "session-trained or loaded model is evaluated as-is, never refitted."
    )

    test_frame = st.session_state.comparison_frame
    labels = labels_from_frame(test_frame)
    if labels is not None and "anomaly_type" in test_frame.columns:
        rows = {}
        for name, det_result in comparison.results.items():
            per_type = recall_by_type(
                labels,
                det_result.labels,
                test_frame["anomaly_type"].to_numpy(),
                tolerance=tolerance,
            )
            rows[name] = {key: value["recall"] for key, value in per_type.items()}
        type_table = pd.DataFrame(rows).T
        if not type_table.empty:
            type_table.index.name = "detector"

            st.subheader("Which detector finds which anomaly type")
            st.dataframe(type_table.round(3))
            _show(plot_type_recall(type_table))

    selected = st.selectbox("Inspect detector on held-out data", list(comparison.results))
    _show(
        plot_series(
            test_frame,
            predicted=comparison.results[selected].labels,
            title=f"{selected} — held-out predictions",
        )
    )


# --------------------------------------------------------------------------- #
# 7. model registry
# --------------------------------------------------------------------------- #
def render_models() -> None:
    st.header("7 · Model registry")
    st.caption(f"Artefact root: {ARTIFACT_ROOT.resolve()}")
    table = _artifact_rows()
    if table.empty:
        st.info("No artefacts yet. Train a model and save it on the Train page.")
        return

    st.dataframe(table.drop(columns=["path"]), hide_index=True)
    selected_version = st.selectbox("Artefact", list(table["version"]))
    path = Path(table.loc[table["version"] == selected_version, "path"].iloc[0])

    try:
        loaded = load_lstm_artifact(path)
    except ArtifactError as exc:
        st.error(str(exc))
        return

    columns = st.columns(4)
    columns[0].metric("Version", loaded.model_version)
    columns[1].metric("Threshold", f"{loaded.detector._threshold:.4f}")
    columns[2].metric("Sequence", loaded.detector.sequence_length)
    columns[3].metric("Created", str(loaded.metadata.get("created_at", "—"))[:19])
    with st.expander("Full metadata", expanded=False):
        st.json(loaded.metadata)
    history = loaded.detector._training_history
    if history:
        _show(plot_training_history(history, title="Saved model training loss"))

    columns = st.columns(2)
    if columns[0].button("Load into this session", type="primary"):
        st.session_state.loaded_model = loaded
        st.success(f"Loaded {loaded.model_version}. Go to Detect or Evaluate.")
    with columns[1].expander("Delete this artefact"):
        st.warning(
            f"This permanently removes `{path}`. It cannot be undone."
        )
        confirmed = st.checkbox(
            f"I understand that {loaded.model_version} will be deleted",
            key=f"confirm_delete_{loaded.model_version}",
        )
        if st.button(
            "Delete artefact",
            disabled=not confirmed,
            key=f"delete_{loaded.model_version}",
        ):
            try:
                delete_lstm_artifact(ARTIFACT_ROOT, loaded.model_version)
                if (
                    st.session_state.loaded_model is not None
                    and st.session_state.loaded_model.model_version
                    == loaded.model_version
                ):
                    st.session_state.loaded_model = None
                st.success(f"Deleted {loaded.model_version}.")
                st.rerun()
            except ArtifactError as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- #
# 8. experiment history
# --------------------------------------------------------------------------- #
def render_history() -> None:
    st.header("8 · Experiment history")
    st.caption(
        "Local-first JSONL log (`reports/experiments.jsonl`). It records "
        "dataset fingerprints, parameters, metrics and durations; no database."
    )
    history = load_history()
    if history.empty:
        st.info("No experiments recorded yet. Train, detect or evaluate first.")
        return

    columns = [
        column
        for column in (
            "timestamp",
            "kind",
            "dataset_source",
            "detector",
            "model_version",
            "duration_seconds",
            "notes",
        )
        if column in history.columns
    ]
    st.dataframe(history[columns].iloc[::-1], hide_index=True)
    st.download_button(
        "Download history CSV",
        history.to_csv(index=False).encode("utf-8"),
        file_name="sentinel_experiments.csv",
        mime="text/csv",
    )

    st.subheader("Compare two runs")
    labels = [
        f"{index}: {row['kind']} / {row.get('detector', '?')} / {row['timestamp']}"
        for index, row in history.iterrows()
    ]
    columns = st.columns(2)
    left = columns[0].selectbox("Run A", labels, index=max(0, len(labels) - 2))
    right = columns[1].selectbox("Run B", labels, index=len(labels) - 1)
    left_record = history.loc[int(left.split(":", 1)[0])]
    right_record = history.loc[int(right.split(":", 1)[0])]
    columns = st.columns(2)
    columns[0].json(
        {
            "kind": left_record["kind"],
            "detector": left_record.get("detector"),
            "metrics": left_record.get("metrics"),
            "params": left_record.get("params"),
        }
    )
    columns[1].json(
        {
            "kind": right_record["kind"],
            "detector": right_record.get("detector"),
            "metrics": right_record.get("metrics"),
            "params": right_record.get("params"),
        }
    )

    with st.expander("Clear history"):
        confirmed = st.checkbox("I understand this deletes the local history file")
        if st.button("Clear history", disabled=not confirmed):
            removed = clear_history()
            st.success(f"Removed {removed} records.")
            st.rerun()


# --------------------------------------------------------------------------- #
# 9. guide
# --------------------------------------------------------------------------- #
def render_guide() -> None:
    st.header("9 · Guide")
    st.markdown(
        """
### First experiment (about two minutes)

1. **Dataset laboratory** → preset **A · Clean seasonal** (zero anomalies).
2. **Train** → keep the defaults, click **Train LSTM**; watch the loss curve and held-out tab.
3. **Model registry** → save the proposed `model_00N`.
4. **Dataset laboratory** → preset **H · Mixed unseen**.
5. **Evaluate** → *Saved LSTM artefact* → load and evaluate; inspect the confusion
   matrix and pick a false positive or false negative to zoom into.
6. **Compare** → run all six detectors and the ensemble on the held-out split.

The full walkthrough, commands and troubleshooting are in
`TESTING_GUIDE.md` at the repository root.

### Reading the metrics

* **Precision** — of the points flagged, how many matched a real anomaly.
* **Recall** — of the real anomalies, how many were found.
* **F1** — harmonic mean; useful because anomalies are rare.
* **FPR** — false flags among normal points; the number to watch on clean data.
* **TP / FP / FN / TN** — counts after one-to-one matching within ±k points.

### Honest scope

* Training and calibration use the training half only; the test half is untouched.
* A CSV without `is_anomaly` never receives precision, recall or F1.
* The LSTM's weak case is densely packed isolated spikes: overlapping windows
  leave almost no clean context. Scenario C measures this rather than hiding it.
* This is a testing laboratory, not a production dashboard. The FastAPI service
  is separate and remains the integration surface.
        """
    )


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
st.sidebar.title("Sentinel ML")
st.sidebar.caption("Anomaly detection & ML experimentation workbench")
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Dataset laboratory",
        "Train",
        "Detect",
        "Evaluate",
        "Compare",
        "Model registry",
        "Experiments",
        "Guide",
    ],
    index=0,
)
st.sidebar.divider()
if st.session_state.data_frame is not None:
    st.sidebar.write(f"**Data:** {st.session_state.data_source}")
    st.sidebar.write(f"{len(st.session_state.data_frame):,} points")
if st.session_state.train_result is not None:
    st.sidebar.write("**Model:** session-trained")
if st.session_state.loaded_model is not None:
    st.sidebar.write(f"**Artefact:** {st.session_state.loaded_model.model_version}")

if section == "Overview":
    render_overview()
elif section == "Dataset laboratory":
    render_data()
elif section == "Train":
    render_train()
elif section == "Detect":
    render_detect()
elif section == "Evaluate":
    render_evaluate()
elif section == "Compare":
    render_compare()
elif section == "Model registry":
    render_models()
elif section == "Experiments":
    render_history()
else:
    render_guide()
