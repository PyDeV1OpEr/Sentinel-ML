import numpy as np
import pytest

from spectral.detectors.statistical import GrubbsDetector, MADDetector, ZScoreDetector


class TestZScoreDetector:
    def test_detects_spike(self, sample_with_anomalies):
        data, expected_indices = sample_with_anomalies
        detector = ZScoreDetector(threshold=3.0)
        result = detector.fit_detect(data)

        # all injected spikes should be detected
        for idx in expected_indices:
            assert result.labels[idx] == 1, f"Missed spike at index {idx}"

    def test_no_false_positives_on_clean_data(self, sample_normal_data):
        detector = ZScoreDetector(threshold=3.0)
        result = detector.fit_detect(sample_normal_data)

        # with threshold=3.0 on normally distributed data, very few false positives
        assert result.anomaly_ratio < 0.02

    def test_respects_threshold_param(self, sample_with_anomalies):
        data, _ = sample_with_anomalies

        loose = ZScoreDetector(threshold=5.0).fit_detect(data)
        strict = ZScoreDetector(threshold=2.0).fit_detect(data)

        # stricter threshold should detect more (or equal)
        assert strict.anomaly_count >= loose.anomaly_count

    def test_detection_result_shape(self, sample_normal_data):
        detector = ZScoreDetector()
        result = detector.fit_detect(sample_normal_data)

        assert result.scores.shape == sample_normal_data.shape
        assert result.labels.shape == sample_normal_data.shape
        assert result.scores.min() >= 0
        assert result.scores.max() <= 1

    def test_fit_stores_statistics(self, sample_normal_data):
        detector = ZScoreDetector()
        detector.fit(sample_normal_data)

        assert detector._mean is not None
        assert detector._std is not None
        assert detector._fitted is True


class TestMADDetector:
    def test_detects_spike(self, sample_with_anomalies):
        data, expected_indices = sample_with_anomalies
        detector = MADDetector(threshold=3.5)
        result = detector.fit_detect(data)

        for idx in expected_indices:
            assert result.labels[idx] == 1

    def test_more_robust_than_zscore(self):
        """MAD should handle data with pre-existing outliers better."""
        rng = np.random.default_rng(42)
        # data with a few large values in training set
        data = rng.standard_normal(500)
        data[0] = 50.0  # extreme outlier in training
        data[1] = -50.0

        # inject a moderate anomaly
        data[250] = 5.0

        zscore = ZScoreDetector(threshold=3.0).fit_detect(data)
        mad = MADDetector(threshold=3.5).fit_detect(data)

        # MAD should still detect the moderate anomaly even with extreme training outliers
        assert mad.labels[250] == 1

    def test_scores_normalized(self, sample_normal_data):
        detector = MADDetector()
        result = detector.fit_detect(sample_normal_data)
        assert result.scores.min() >= 0
        assert result.scores.max() <= 1


class TestGrubbsDetector:
    def test_single_outlier(self):
        data = np.array([1.0, 1.1, 0.9, 1.0, 0.95, 1.05, 10.0, 1.0, 0.9])
        detector = GrubbsDetector(alpha=0.05)
        result = detector.detect(data)

        assert result.labels[6] == 1  # the 10.0 should be detected

    def test_no_outliers_in_uniform_data(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        detector = GrubbsDetector(alpha=0.05)
        result = detector.detect(data)

        # very few or no outliers in standard normal data
        assert result.anomaly_count <= 3

    def test_max_outliers_limit(self, sample_with_anomalies):
        data, _ = sample_with_anomalies
        detector = GrubbsDetector(alpha=0.05, max_outliers=2)
        result = detector.detect(data)

        assert result.anomaly_count <= 2
