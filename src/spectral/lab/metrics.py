"""One-to-one anomaly matching and evaluation metrics.

Point-wise labels are useful for scoring, but several detectors smear a single
event across a window. The matcher below therefore allows a tolerance while
enforcing a strict one-to-one rule: one predicted point can satisfy at most one
ground-truth point, and vice versa. This prevents a cluster of predictions
around one anomaly from inflating recall or precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class MatchResult:
    matched_true: tuple[int, ...]
    matched_pred: tuple[int, ...]
    tp: int
    fp: int
    fn: int


def match_anomalies(
    true_indices: np.ndarray | list[int],
    pred_indices: np.ndarray | list[int],
    tolerance: int = 2,
) -> MatchResult:
    """Greedy nearest one-to-one matching within ``tolerance`` points."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    true_sorted = sorted({int(i) for i in np.asarray(true_indices).tolist()})
    pred_sorted = sorted({int(i) for i in np.asarray(pred_indices).tolist()})
    used_pred: set[int] = set()
    matched_true: list[int] = []
    matched_pred: list[int] = []

    for truth in true_sorted:
        best: tuple[int, int] | None = None
        for pred in pred_sorted:
            if pred in used_pred:
                continue
            distance = abs(pred - truth)
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, pred)
        if best is not None:
            used_pred.add(best[1])
            matched_true.append(truth)
            matched_pred.append(best[1])

    tp = len(matched_true)
    return MatchResult(
        matched_true=tuple(matched_true),
        matched_pred=tuple(matched_pred),
        tp=tp,
        fp=len(pred_sorted) - tp,
        fn=len(true_sorted) - tp,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def event_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    tolerance: int = 2,
) -> dict:
    """Precision/recall/F1 plus confusion counts with one-to-one matching."""
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)
    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape")

    true_indices = np.where(labels_true == 1)[0]
    pred_indices = np.where(labels_pred == 1)[0]
    match = match_anomalies(true_indices, pred_indices, tolerance)

    tp = match.tp
    fp = match.fp
    fn = match.fn
    tn = len(labels_true) - tp - fp - fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_true": int(len(true_indices)),
        "n_pred": int(len(pred_indices)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, len(labels_true)),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "tolerance": int(tolerance),
        "matched_pairs": list(zip(match.matched_true, match.matched_pred)),
    }


def false_positive_indices(
    labels_true: np.ndarray, labels_pred: np.ndarray, tolerance: int = 2
) -> list[int]:
    """Predicted indices that did not match any ground-truth point."""
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)
    match = match_anomalies(
        np.where(labels_true == 1)[0], np.where(labels_pred == 1)[0], tolerance
    )
    matched = set(match.matched_pred)
    return [int(i) for i in np.where(labels_pred == 1)[0] if int(i) not in matched]


def false_negative_indices(
    labels_true: np.ndarray, labels_pred: np.ndarray, tolerance: int = 2
) -> list[int]:
    """Ground-truth indices that no prediction matched."""
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)
    match = match_anomalies(
        np.where(labels_true == 1)[0], np.where(labels_pred == 1)[0], tolerance
    )
    matched = set(match.matched_true)
    return [int(i) for i in np.where(labels_true == 1)[0] if int(i) not in matched]


def recall_by_type(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    anomaly_types: np.ndarray,
    tolerance: int = 2,
) -> dict[str, dict]:
    """Per-anomaly-type recall under one-to-one matching.

    Precision per type is not well defined when a prediction may match a
    different type; recall is, and it is the useful question for a detector
    comparison ("which events does this detector miss?").
    """
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)
    types = np.asarray(anomaly_types, dtype=object)
    pred_indices = np.where(labels_pred == 1)[0]
    results: dict[str, dict] = {}

    for name in sorted({str(value) for value in types if str(value) != "none"}):
        truth = [
            int(i)
            for i in np.where((types == name) & (labels_true == 1))[0]
        ]
        if not truth:
            continue
        match = match_anomalies(truth, pred_indices, tolerance)
        results[name] = {
            "n_true": len(truth),
            "matched": int(match.tp),
            "recall": _safe_div(match.tp, len(truth)),
        }
    return results


def confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """2x2 matrix in ``[[tn, fp], [fn, tp]]`` order."""
    labels_true = np.asarray(labels_true).astype(int)
    labels_pred = np.asarray(labels_pred).astype(int)
    tn = int(((labels_true == 0) & (labels_pred == 0)).sum())
    fp = int(((labels_true == 0) & (labels_pred == 1)).sum())
    fn = int(((labels_true == 1) & (labels_pred == 0)).sum())
    tp = int(((labels_true == 1) & (labels_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_curve_data(
    labels_true: np.ndarray, scores: np.ndarray
) -> dict:
    """Point-wise PR curve and average precision, or empty arrays."""
    labels_true = np.asarray(labels_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels_true)) < 2:
        return {
            "precision": np.array([]),
            "recall": np.array([]),
            "thresholds": np.array([]),
            "average_precision": 0.0,
        }
    precision, recall, thresholds = precision_recall_curve(labels_true, scores)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": float(average_precision_score(labels_true, scores)),
    }


def threshold_sweep(
    labels_true: np.ndarray,
    scores: np.ndarray,
    tolerance: int = 2,
    n_thresholds: int = 50,
) -> list[dict]:
    """Score-threshold sensitivity using event-level one-to-one metrics.

    Higher scores mean more anomalous. This sweep is independent of the
    detector's own calibrated threshold and is meant to show how precision and
    recall trade off on the evaluated split.
    """
    labels_true = np.asarray(labels_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return []
    quantiles = np.linspace(0.02, 0.98, n_thresholds)
    thresholds = np.unique(np.quantile(scores, quantiles))
    rows = []
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        metrics = event_metrics(labels_true, predicted, tolerance)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows
