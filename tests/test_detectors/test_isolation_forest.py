import numpy as np

from spectral.detectors.isolation_forest import IsolationForestDetector


class TestIsolationForestDetector:
    def test_detects_injected_anomalies(self, sample_with_anomalies):
        data, expected_indices = sample_with_anomalies
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_detect(data)

        # should detect at least 3 out of 5 injected spikes
        detected = sum(1 for idx in expected_indices if result.labels[idx] == 1)
        assert detected >= 3

    def test_contamination_param_affects_results(self, sample_normal_data):
        low = IsolationForestDetector(contamination=0.01).fit_detect(sample_normal_data)
        high = IsolationForestDetector(contamination=0.1).fit_detect(sample_normal_data)

        assert high.anomaly_count > low.anomaly_count

    def test_scores_between_zero_and_one(self, sample_normal_data):
        detector = IsolationForestDetector()
        result = detector.fit_detect(sample_normal_data)

        assert result.scores.min() >= 0
        assert result.scores.max() <= 1

    def test_fit_then_detect_separately(self, sample_normal_data, sample_with_anomalies):
        data_with, expected = sample_with_anomalies
        detector = IsolationForestDetector(contamination=0.05)

        # fit on clean data
        detector.fit(sample_normal_data)
        # detect on data with anomalies
        result = detector.detect(data_with)

        # should find some of the injected anomalies
        detected = sum(1 for idx in expected if result.labels[idx] == 1)
        assert detected >= 2

    def test_feature_construction(self):
        detector = IsolationForestDetector(window_features=3)
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        features = detector._build_features(data)

        assert features.shape == (5, 3)
