import numpy as np
import pytest

from spectral.detectors.spectral_residual import SpectralResidualDetector


class TestSpectralResidualDetector:
    def test_detects_point_anomaly_in_periodic_signal(self):
        """Spike in a clean sine wave should be detected."""
        t = np.arange(500)
        signal = np.sin(2 * np.pi * t / 50)
        signal[250] += 5.0  # inject spike

        detector = SpectralResidualDetector(window_size=24, score_threshold=3.0)
        result = detector.detect(signal)

        assert result.labels[250] == 1

    def test_saliency_map_shape(self, sample_normal_data):
        detector = SpectralResidualDetector()
        result = detector.detect(sample_normal_data)

        assert result.scores.shape == sample_normal_data.shape
        assert result.labels.shape == sample_normal_data.shape

    def test_no_anomalies_in_clean_periodic_signal(self):
        """Pure sine wave should produce few or no detections."""
        t = np.arange(1000)
        signal = np.sin(2 * np.pi * t / 100)

        detector = SpectralResidualDetector(score_threshold=3.0)
        result = detector.detect(signal)

        # clean periodic signal should have very few false positives
        assert result.anomaly_ratio < 0.05

    def test_window_size_param(self, sample_with_anomalies):
        data, _ = sample_with_anomalies

        small_window = SpectralResidualDetector(window_size=5).detect(data)
        large_window = SpectralResidualDetector(window_size=48).detect(data)

        # both should detect something, results will differ
        assert small_window.anomaly_count > 0 or large_window.anomaly_count > 0

    def test_fit_is_noop(self):
        detector = SpectralResidualDetector()
        detector.fit(np.array([1, 2, 3]))
        assert detector._fitted is True
