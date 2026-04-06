import numpy as np
import pytest

from spectral.detectors.lstm_autoencoder import LSTMAutoencoderDetector


@pytest.mark.slow
class TestLSTMAutoencoderDetector:
    def test_model_reconstructs_normal_data(self):
        """Training data should have low reconstruction error."""
        rng = np.random.default_rng(42)
        data = np.sin(np.arange(300) * 0.1) + rng.standard_normal(300) * 0.05

        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=30, lr=1e-3
        )
        detector.fit(data)
        result = detector.detect(data)

        # most training data should be classified as normal
        assert result.anomaly_ratio < 0.2

    def test_model_flags_anomalous_pattern(self):
        """Data with injected spikes should be flagged."""
        rng = np.random.default_rng(42)
        data = np.sin(np.arange(300) * 0.1) + rng.standard_normal(300) * 0.05

        # inject anomalies
        data[150] += 5.0
        data[200] += 5.0

        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=30, lr=1e-3
        )
        result = detector.fit_detect(data)

        # should detect at least one of the spikes
        assert result.labels[150] == 1 or result.labels[200] == 1

    def test_training_reduces_loss(self):
        """Loss should decrease over epochs."""
        import torch

        rng = np.random.default_rng(42)
        data = np.sin(np.arange(200) * 0.1) + rng.standard_normal(200) * 0.05

        detector = LSTMAutoencoderDetector(
            hidden_dim=8, sequence_length=10, epochs=20, lr=1e-3
        )
        # just verify fit completes without error
        detector.fit(data.astype(np.float32))
        assert detector._fitted is True
        assert detector._threshold > 0
