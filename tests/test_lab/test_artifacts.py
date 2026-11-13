import json

import numpy as np
import pandas as pd
import pytest

from spectral.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from spectral.lab.artifacts import (
    ArtifactError,
    delete_lstm_artifact,
    load_lstm_artifact,
    next_model_version,
    save_lstm_artifact,
)
from spectral.lab.data import fingerprint_frame


@pytest.fixture(scope="module")
def fitted_detector():
    rng = np.random.default_rng(3)
    data = np.sin(np.arange(220) * 0.1) + rng.standard_normal(220) * 0.05
    detector = LSTMAutoencoderDetector(
        hidden_dim=8, sequence_length=12, epochs=4, seed=3
    ).fit(data)
    return detector, data


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="h"),
            "value": values,
        }
    )


@pytest.mark.slow
class TestArtifactLifecycle:
    def test_round_trip_restores_everything(self, fitted_detector, tmp_path):
        detector, data = fitted_detector
        frame = _frame(data)
        path = save_lstm_artifact(
            detector,
            tmp_path,
            dataset_frame=frame,
            train_points=150,
            fit_seconds=0.5,
        )
        assert path.name == "model_001"
        assert sorted(child.name for child in path.iterdir()) == [
            "calibration.json",
            "metadata.json",
            "model.pt",
            "preprocessing.json",
        ]

        loaded = load_lstm_artifact(path)
        assert loaded.detector._fitted is True
        assert loaded.detector._threshold == detector._threshold
        assert loaded.detector._train_mean == detector._train_mean
        assert loaded.detector._train_std == detector._train_std
        assert loaded.detector._device.type == "cpu"
        assert loaded.metadata["dataset"]["fingerprint_sha256"] == fingerprint_frame(frame)
        assert loaded.metadata["training"]["train_points"] == 150

        before = detector.detect(data)
        after = loaded.detector.detect(data)
        np.testing.assert_array_equal(before.labels, after.labels)
        np.testing.assert_allclose(before.scores, after.scores, atol=1e-8)

    def test_loading_never_calls_fit(self, fitted_detector, tmp_path, monkeypatch):
        detector, data = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)

        def fail(*args, **kwargs):
            raise AssertionError("fit() must not be called during load")

        monkeypatch.setattr(LSTMAutoencoderDetector, "fit", fail)
        loaded = load_lstm_artifact(path)
        assert loaded.detector._fitted is True

    def test_auto_versioning(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        first = save_lstm_artifact(detector, tmp_path)
        second = save_lstm_artifact(detector, tmp_path)
        assert first.name == "model_001"
        assert second.name == "model_002"
        assert next_model_version(tmp_path) == "model_003"

    def test_missing_file_is_rejected(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)
        (path / "calibration.json").unlink()
        with pytest.raises(ArtifactError, match="incomplete"):
            load_lstm_artifact(path)

    def test_corrupt_model_is_rejected(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)
        with (path / "model.pt").open("ab") as handle:
            handle.write(b"corrupt")
        with pytest.raises(ArtifactError, match="checksum"):
            load_lstm_artifact(path)

    def test_corrupt_json_is_rejected(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)
        (path / "metadata.json").write_text("{not-json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="cannot read metadata.json"):
            load_lstm_artifact(path)

    def test_unsupported_schema_is_rejected(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)
        metadata = json.loads((path / "metadata.json").read_text())
        metadata["schema_version"] = 99
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ArtifactError, match="unsupported artefact schema"):
            load_lstm_artifact(path)

    def test_unfitted_detector_cannot_be_saved(self, tmp_path):
        with pytest.raises(ArtifactError, match="fitted"):
            save_lstm_artifact(LSTMAutoencoderDetector(), tmp_path)

    def test_unsafe_version_is_rejected(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        with pytest.raises(ArtifactError, match="model_version"):
            save_lstm_artifact(detector, tmp_path, model_version="../escape")

    def test_training_history_round_trips(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        assert len(detector._training_history) > 0
        path = save_lstm_artifact(detector, tmp_path)
        loaded = load_lstm_artifact(path)

        assert loaded.detector._training_history == detector._training_history
        assert loaded.metadata["training"]["history"] == detector._training_history
        first = loaded.detector._training_history[0]
        assert set(first) == {"epoch", "train_loss", "validation_loss"}
        assert first["epoch"] == 1

    def test_delete_is_scoped_to_real_artefacts(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)

        with pytest.raises(ArtifactError, match="invalid model_version"):
            delete_lstm_artifact(tmp_path, "../escape")
        with pytest.raises(ArtifactError, match="not found"):
            delete_lstm_artifact(tmp_path, "model_999")

        deleted = delete_lstm_artifact(tmp_path, path.name)
        assert deleted == path.resolve()
        assert not path.exists()

    def test_reloaded_predictions_match_within_tolerance(
        self, fitted_detector, tmp_path
    ):
        detector, _ = fitted_detector
        path = save_lstm_artifact(detector, tmp_path)
        loaded = load_lstm_artifact(path)
        fresh = np.sin(np.arange(300) * 0.1 + 1.1) + np.random.default_rng(
            11
        ).standard_normal(300) * 0.05

        before = detector.detect(fresh)
        after = loaded.detector.detect(fresh)
        np.testing.assert_allclose(before.scores, after.scores, atol=1e-8)
        np.testing.assert_array_equal(before.labels, after.labels)
        assert before.threshold == after.threshold
