import numpy as np
import pytest

from spectral.lab.metrics import (
    confusion_matrix,
    event_metrics,
    false_negative_indices,
    false_positive_indices,
    match_anomalies,
    precision_recall_curve_data,
    recall_by_type,
    threshold_sweep,
)


class TestOneToOneMatching:
    def test_one_prediction_cannot_match_two_truths(self):
        match = match_anomalies([10, 11], [10], tolerance=1)
        assert match.tp == 1
        assert match.fn == 1
        assert match.fp == 0

    def test_two_predictions_cannot_both_match_one_truth(self):
        match = match_anomalies([10], [9, 10], tolerance=1)
        assert match.tp == 1
        assert match.fp == 1
        assert match.fn == 0

    def test_nearest_prediction_is_chosen(self):
        match = match_anomalies([10], [13, 9], tolerance=3)
        assert match.matched_pred == (9,)

    def test_outside_tolerance_is_not_a_match(self):
        match = match_anomalies([10], [13], tolerance=2)
        assert match.tp == 0
        assert match.fp == 1
        assert match.fn == 1

    def test_duplicate_indices_are_deduplicated(self):
        match = match_anomalies([5, 5], [5, 5], tolerance=0)
        assert match.tp == 1
        assert match.fp == 0
        assert match.fn == 0


class TestEventMetrics:
    def test_metrics_and_confusion_counts(self):
        labels_true = np.zeros(100, dtype=int)
        labels_pred = np.zeros(100, dtype=int)
        labels_true[[10, 20, 30]] = 1
        labels_pred[[10, 11, 20]] = 1

        metrics = event_metrics(labels_true, labels_pred, tolerance=2)
        assert metrics["tp"] == 2
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["tn"] == 96
        assert metrics["precision"] == 2 / 3
        assert metrics["recall"] == 2 / 3
        assert metrics["false_positive_rate"] == pytest.approx(1 / 97)

        matrix = confusion_matrix(labels_true, labels_pred)
        assert matrix.tolist() == [[96, 1], [1, 2]]

    def test_perfect_prediction(self):
        labels = np.zeros(50, dtype=int)
        labels[[5, 20]] = 1
        metrics = event_metrics(labels, labels, tolerance=0)
        assert (metrics["precision"], metrics["recall"], metrics["f1"]) == (1.0, 1.0, 1.0)
        assert metrics["tp"] == 2 and metrics["fp"] == 0 and metrics["fn"] == 0

    def test_no_predictions(self):
        labels_true = np.zeros(30, dtype=int)
        labels_true[[4, 8]] = 1
        metrics = event_metrics(labels_true, np.zeros(30, dtype=int), tolerance=2)
        assert metrics["tp"] == 0
        assert metrics["fn"] == 2
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestErrorReview:
    def test_false_positive_and_negative_indices(self):
        labels_true = np.zeros(40, dtype=int)
        labels_pred = np.zeros(40, dtype=int)
        labels_true[[10, 20]] = 1
        labels_pred[[10, 11, 30]] = 1

        assert false_positive_indices(labels_true, labels_pred, tolerance=2) == [11, 30]
        assert false_negative_indices(labels_true, labels_pred, tolerance=2) == [20]

    def test_recall_by_type_is_one_to_one(self):
        labels_true = np.zeros(20, dtype=int)
        labels_pred = np.zeros(20, dtype=int)
        labels_true[[2, 8, 14]] = 1
        labels_pred[[2, 9]] = 1
        types = np.array(
            ["none"] * 20, dtype=object
        )
        types[[2, 8]] = "spike"
        types[14] = "drift"

        result = recall_by_type(labels_true, labels_pred, types, tolerance=1)
        assert result["spike"] == {"n_true": 2, "matched": 2, "recall": 1.0}
        assert result["drift"] == {"n_true": 1, "matched": 0, "recall": 0.0}


class TestCurves:
    def test_pr_curve_and_average_precision(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
        curve = precision_recall_curve_data(labels, scores)
        assert len(curve["precision"]) > 0
        assert 0 <= curve["average_precision"] <= 1

    def test_pr_curve_without_positives_is_empty(self):
        curve = precision_recall_curve_data(np.zeros(10), np.zeros(10))
        assert len(curve["precision"]) == 0
        assert curve["average_precision"] == 0.0

    def test_threshold_sweep_is_monotonic_and_bounded(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(200) < 0.05).astype(int)
        scores = rng.random(200)
        rows = threshold_sweep(labels, scores, tolerance=0, n_thresholds=20)

        assert len(rows) > 1
        thresholds = [row["threshold"] for row in rows]
        assert thresholds == sorted(thresholds)
        assert all(0 <= row["precision"] <= 1 for row in rows)
        assert all(0 <= row["recall"] <= 1 for row in rows)
