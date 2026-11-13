"""Training, prediction and all-detector comparison for the lab."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spectral.detectors import (
    DetectionResult,
    EnsembleDetector,
    GrubbsDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    MADDetector,
    SpectralResidualDetector,
    ZScoreDetector,
)
from spectral.lab.data import (
    chronological_split,
    fingerprint_frame,
    impute_values,
    labels_from_frame,
)
from spectral.lab.metrics import event_metrics

DEFAULT_LSTM_PARAMS = {
    "hidden_dim": 32,
    "n_layers": 1,
    "sequence_length": 30,
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 32,
    "threshold_sigma": 2.0,
    "patience": 5,
    "seed": 42,
    "calibration_fraction": 0.2,
}


def peak_memory_mb() -> float:
    """Process peak RSS in MiB (macOS reports bytes, Linux KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024 * 1024)
    return float(usage) / 1024


@dataclass
class LSTMTrainResult:
    detector: LSTMAutoencoderDetector
    params: dict
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_values: np.ndarray
    test_values: np.ndarray
    train_result: DetectionResult
    test_result: DetectionResult
    train_errors: np.ndarray
    test_errors: np.ndarray
    train_metrics: dict | None
    test_metrics: dict | None
    fit_seconds: float
    inference_seconds: float
    peak_memory_mb: float


@dataclass
class PredictionResult:
    frame: pd.DataFrame
    values: np.ndarray
    result: DetectionResult
    errors: np.ndarray
    metrics: dict | None


@dataclass
class DetectorComparison:
    rows: pd.DataFrame
    results: dict[str, DetectionResult] = field(default_factory=dict)
    errors: dict[str, np.ndarray] = field(default_factory=dict)
    lstm_detector: LSTMAutoencoderDetector | None = None


def _metrics_or_none(frame: pd.DataFrame, result: DetectionResult, tolerance: int):
    labels = labels_from_frame(frame)
    if labels is None:
        return None
    return event_metrics(labels, result.labels, tolerance=tolerance)


def train_lstm_on_frame(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    params: dict | None = None,
    tolerance: int = 2,
    progress_callback=None,
) -> LSTMTrainResult:
    """Chronologically split, fit only on train, then score train and test.

    The detector's own calibration holdout lives inside the training half, so
    the test half is untouched by both optimisation and threshold estimation.
    """
    resolved = {**DEFAULT_LSTM_PARAMS, **(params or {})}
    train_frame, test_frame = chronological_split(frame, train_fraction)

    train_values = impute_values(train_frame["value"])
    test_values = impute_values(test_frame["value"])
    sequence_length = int(resolved["sequence_length"])
    if len(train_values) < sequence_length + 2:
        raise ValueError(
            f"training split has {len(train_values)} points; need at least "
            f"{sequence_length + 2} for sequence_length={sequence_length}"
        )
    if len(test_values) < sequence_length:
        raise ValueError(
            f"test split has {len(test_values)} points; need at least "
            f"{sequence_length} for inference"
        )

    detector = LSTMAutoencoderDetector(**resolved)

    started = time.perf_counter()
    detector.fit(train_values, progress_callback=progress_callback)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    train_result = detector.detect(train_values)
    test_result = detector.detect(test_values)
    inference_seconds = time.perf_counter() - started

    return LSTMTrainResult(
        detector=detector,
        params=resolved,
        train_frame=train_frame,
        test_frame=test_frame,
        train_values=train_values,
        test_values=test_values,
        train_result=train_result,
        test_result=test_result,
        train_errors=detector.reconstruction_errors(train_values),
        test_errors=detector.reconstruction_errors(test_values),
        train_metrics=_metrics_or_none(train_frame, train_result, tolerance),
        test_metrics=_metrics_or_none(test_frame, test_result, tolerance),
        fit_seconds=fit_seconds,
        inference_seconds=inference_seconds,
        peak_memory_mb=peak_memory_mb(),
    )


def predict_with_detector(
    detector: LSTMAutoencoderDetector,
    frame: pd.DataFrame,
    *,
    tolerance: int = 2,
) -> PredictionResult:
    """Run a fitted detector on a frame without training."""
    if not detector._fitted:
        raise RuntimeError("detector must be fitted or loaded before predicting")
    values = impute_values(frame["value"])
    if len(values) < detector.sequence_length:
        raise ValueError(
            f"need at least {detector.sequence_length} points for inference"
        )
    result = detector.detect(values)
    return PredictionResult(
        frame=frame,
        values=values,
        result=result,
        errors=detector.reconstruction_errors(values),
        metrics=_metrics_or_none(frame, result, tolerance),
    )


def evaluate_loaded_model(
    loaded,
    frame: pd.DataFrame,
    *,
    tolerance: int = 2,
    allow_unseen: bool = True,
) -> dict:
    """Evaluate a loaded artefact, using its recorded split when possible.

    Returns ``{"train": PredictionResult, "test": PredictionResult,
    "split_matched": bool}`` when the dataset fingerprint matches the artefact
    and both halves are available, otherwise ``{"all": PredictionResult,
    "split_matched": False}``.
    """
    metadata = loaded.metadata
    recorded = metadata.get("dataset", {}).get("fingerprint_sha256")
    train_points = metadata.get("training", {}).get("train_points")
    matched = bool(
        recorded
        and train_points
        and fingerprint_frame(frame) == recorded
        and 0 < int(train_points) < len(frame)
    )

    if matched:
        cut = int(train_points)
        train_frame = frame.iloc[:cut].reset_index(drop=True)
        test_frame = frame.iloc[cut:].reset_index(drop=True)
        return {
            "train": predict_with_detector(loaded.detector, train_frame, tolerance=tolerance),
            "test": predict_with_detector(loaded.detector, test_frame, tolerance=tolerance),
            "split_matched": True,
        }

    if not allow_unseen:
        raise ValueError(
            "dataset does not match the artefact fingerprint and no split is available"
        )
    return {
        "all": predict_with_detector(loaded.detector, frame, tolerance=tolerance),
        "split_matched": False,
    }


def classical_detectors() -> dict:
    """The five non-LSTM detectors with the upstream defaults."""
    return {
        "Z-Score": ZScoreDetector(threshold=3.0),
        "MAD": MADDetector(threshold=3.5),
        "Grubbs": GrubbsDetector(alpha=0.05),
        "Isolation Forest": IsolationForestDetector(
            contamination=0.05, random_state=42
        ),
        "Spectral Residual": SpectralResidualDetector(
            window_size=24, score_threshold=3.0
        ),
    }


def default_ensemble() -> EnsembleDetector:
    """The ensemble built by ``configs/default.yml``."""
    return EnsembleDetector(
        [
            ZScoreDetector(threshold=3.0),
            IsolationForestDetector(contamination=0.05, random_state=42),
            SpectralResidualDetector(window_size=24, score_threshold=3.0),
        ],
        method="voting",
        min_detectors=2,
    )


def _row(
    name: str,
    train_result: DetectionResult,
    test_result: DetectionResult,
    train_metrics: dict | None,
    test_metrics: dict | None,
    fit_seconds: float,
    inference_seconds: float,
) -> dict:
    train_metrics = train_metrics or {}
    test_metrics = test_metrics or {}
    return {
        "detector": name,
        "train_precision": train_metrics.get("precision", np.nan),
        "train_recall": train_metrics.get("recall", np.nan),
        "train_f1": train_metrics.get("f1", np.nan),
        "test_precision": test_metrics.get("precision", np.nan),
        "test_recall": test_metrics.get("recall", np.nan),
        "test_f1": test_metrics.get("f1", np.nan),
        "test_accuracy": test_metrics.get("accuracy", np.nan),
        "test_fpr": test_metrics.get("false_positive_rate", np.nan),
        "test_tp": test_metrics.get("tp", np.nan),
        "test_fp": test_metrics.get("fp", np.nan),
        "test_fn": test_metrics.get("fn", np.nan),
        "test_tn": test_metrics.get("tn", np.nan),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "train_anomalies": train_result.anomaly_count,
        "test_anomalies": test_result.anomaly_count,
    }


def run_detector_comparison(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    lstm_detector: LSTMAutoencoderDetector | None = None,
    lstm_params: dict | None = None,
    tolerance: int = 2,
    include_lstm: bool = True,
) -> DetectorComparison:
    """Fit every detector on train and score the held-out test frame.

    A supplied ``lstm_detector`` is assumed to have been fitted on this train
    half. When it is omitted, the LSTM is trained here with ``lstm_params``.
    """
    train_values = impute_values(train_frame["value"])
    test_values = impute_values(test_frame["value"])
    train_labels = labels_from_frame(train_frame)
    test_labels = labels_from_frame(test_frame)

    detectors = classical_detectors()
    detectors["Ensemble"] = default_ensemble()

    rows = []
    results: dict[str, DetectionResult] = {}
    errors: dict[str, np.ndarray] = {}

    for name, detector in detectors.items():
        started = time.perf_counter()
        detector.fit(train_values)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        train_result = detector.detect(train_values)
        test_result = detector.detect(test_values)
        inference_seconds = time.perf_counter() - started

        train_metrics = (
            event_metrics(train_labels, train_result.labels, tolerance)
            if train_labels is not None
            else None
        )
        test_metrics = (
            event_metrics(test_labels, test_result.labels, tolerance)
            if test_labels is not None
            else None
        )
        rows.append(
            _row(
                name,
                train_result,
                test_result,
                train_metrics,
                test_metrics,
                fit_seconds,
                inference_seconds,
            )
        )
        results[name] = test_result

    if include_lstm:
        if lstm_detector is None:
            started = time.perf_counter()
            lstm_detector = LSTMAutoencoderDetector(
                **{**DEFAULT_LSTM_PARAMS, **(lstm_params or {})}
            )
            lstm_detector.fit(train_values)
            fit_seconds = time.perf_counter() - started
        else:
            fit_seconds = float("nan")

        started = time.perf_counter()
        train_result = lstm_detector.detect(train_values)
        test_result = lstm_detector.detect(test_values)
        inference_seconds = time.perf_counter() - started

        train_metrics = (
            event_metrics(train_labels, train_result.labels, tolerance)
            if train_labels is not None
            else None
        )
        test_metrics = (
            event_metrics(test_labels, test_result.labels, tolerance)
            if test_labels is not None
            else None
        )
        rows.append(
            _row(
                "LSTM Autoencoder",
                train_result,
                test_result,
                train_metrics,
                test_metrics,
                fit_seconds,
                inference_seconds,
            )
        )
        results["LSTM Autoencoder"] = test_result
        errors["LSTM Autoencoder"] = lstm_detector.reconstruction_errors(test_values)

    return DetectorComparison(
        rows=pd.DataFrame(rows),
        results=results,
        errors=errors,
        lstm_detector=lstm_detector,
    )
