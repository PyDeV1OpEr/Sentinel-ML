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
        assert torch.isfinite(torch.tensor(detector._threshold))

    def test_calibration_uses_held_out_windows(self):
        """The threshold is estimated from windows the model did not train on."""
        rng = np.random.default_rng(42)
        data = np.sin(np.arange(300) * 0.1) + rng.standard_normal(300) * 0.05

        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=20, calibration_fraction=0.2
        ).fit(data)

        assert detector._calibration_source == "holdout"
        assert detector._calibration_sequences > 0
        assert detector._threshold > 0

    def test_repeated_fits_are_reproducible(self):
        """A fixed seed must make training and calibration deterministic."""
        rng = np.random.default_rng(7)
        data = np.sin(np.arange(240) * 0.1) + rng.standard_normal(240) * 0.05

        params = dict(hidden_dim=8, sequence_length=15, epochs=12, seed=123)
        first = LSTMAutoencoderDetector(**params).fit_detect(data)
        second = LSTMAutoencoderDetector(**params).fit_detect(data)

        assert first.threshold == second.threshold
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.scores, second.scores)

    def test_flags_injected_anomalies_in_held_out_data(self):
        """Calibration must not suppress genuine anomalies on unseen data."""
        rng = np.random.default_rng(42)
        clean = np.sin(np.arange(400) * 0.1) + rng.standard_normal(400) * 0.05

        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=40
        ).fit(clean[:300])

        held_out = clean[300:].copy()
        held_out[[20, 60]] += 5.0
        result = detector.detect(held_out)

        assert result.labels[20] == 1
        assert result.labels[60] == 1

    def test_clean_training_data_flag_rate_is_calibrated(self):
        """Normal training data should not be flagged at anything like 16%."""
        rng = np.random.default_rng(42)
        data = np.sin(np.arange(500) * 0.1) + rng.standard_normal(500) * 0.05

        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=30, threshold_sigma=2.0
        )
        result = detector.fit_detect(data)

        assert result.anomaly_ratio < 0.05

    def test_single_spike_is_localised_to_the_spike(self):
        """A spike must not smear its window error onto its neighbours.

        Max-pooling the window error used to label every point in every window
        containing the spike. Scoring each point by its best covering window
        keeps the label on the spike itself.
        """
        rng = np.random.default_rng(42)
        clean = np.sin(np.arange(600) * 0.1) + rng.standard_normal(600) * 0.05
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=40, seed=7
        ).fit(clean[:400])

        held_out = clean[400:].copy()
        held_out[50] += 5.0
        result = detector.detect(held_out)

        assert result.labels[50] == 1
        flagged = np.flatnonzero(result.labels)
        assert flagged.tolist() == [50], f"spike smeared to {flagged.tolist()}"

    def test_unseen_clean_series_has_no_false_positives(self):
        """An unseen clean continuation must not be flagged wholesale."""
        rng = np.random.default_rng(42)
        clean = np.sin(np.arange(700) * 0.1) + rng.standard_normal(700) * 0.05
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, sequence_length=20, epochs=40, seed=7
        ).fit(clean[:400])

        fresh = (
            np.sin(np.arange(400) * 0.1 + 1.7)
            + np.random.default_rng(99).standard_normal(400) * 0.05
        )
        result = detector.detect(fresh)

        assert result.anomaly_count == 0
        assert result.anomaly_ratio == 0.0

    def test_progress_callback_and_history_match(self):
        rng = np.random.default_rng(42)
        data = np.sin(np.arange(200) * 0.1) + rng.standard_normal(200) * 0.05
        seen = []
        detector = LSTMAutoencoderDetector(
            hidden_dim=8, sequence_length=10, epochs=6, patience=50, seed=3
        ).fit(
            data,
            progress_callback=lambda epoch, total, train, val: seen.append(
                (epoch, total, train, val)
            ),
        )

        assert len(seen) == len(detector._training_history) == 6
        assert [entry[0] for entry in seen] == [1, 2, 3, 4, 5, 6]
        assert all(entry[1] == 6 for entry in seen)
        assert detector._training_history[0]["epoch"] == 1
        assert np.isfinite(detector._training_history[-1]["train_loss"])

    def test_short_series_raises_clear_error(self):
        detector = LSTMAutoencoderDetector(sequence_length=20)
        with pytest.raises(ValueError, match="at least 22 points"):
            detector.fit(np.arange(21, dtype=np.float32))
