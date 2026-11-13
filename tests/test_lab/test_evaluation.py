import pandas as pd
import pytest

from spectral.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from spectral.lab.artifacts import load_lstm_artifact, save_lstm_artifact
from spectral.lab.data import prepare_frame
from spectral.lab.evaluation import (
    evaluate_loaded_model,
    predict_with_detector,
    run_detector_comparison,
    train_lstm_on_frame,
)
from spectral.lab.synthetic import SyntheticConfig, generate_synthetic_timeseries

SMALL = {
    "hidden_dim": 8,
    "sequence_length": 12,
    "epochs": 4,
    "seed": 5,
}


@pytest.fixture(scope="module")
def prepared_frame():
    dataset = generate_synthetic_timeseries(
        SyntheticConfig(
            n_points=320,
            anomaly_ratio=0.07,
            anomaly_types=("spike", "drop", "level_shift"),
            seed=21,
        )
    )
    frame, has_labels = prepare_frame(dataset.frame)
    assert has_labels
    return frame


@pytest.fixture(scope="module")
def training_result(prepared_frame):
    return train_lstm_on_frame(prepared_frame, train_fraction=0.7, params=SMALL)


@pytest.mark.slow
class TestEvaluation:
    def test_split_is_chronological_and_disjoint(self, training_result):
        assert (
            training_result.train_frame["timestamp"].max()
            < training_result.test_frame["timestamp"].min()
        )
        assert len(training_result.train_frame) + len(training_result.test_frame) == 320
        assert len(training_result.train_values) == len(training_result.train_frame)
        assert len(training_result.test_errors) == len(training_result.test_frame)

    def test_metrics_structure(self, training_result):
        for metrics in (training_result.train_metrics, training_result.test_metrics):
            assert metrics is not None
            assert set(metrics) >= {
                "precision",
                "recall",
                "f1",
                "accuracy",
                "false_positive_rate",
                "tp",
                "fp",
                "fn",
                "tn",
            }
            assert 0 <= metrics["precision"] <= 1
            assert 0 <= metrics["recall"] <= 1
            assert metrics["tp"] + metrics["fn"] == metrics["n_true"]

    def test_comparison_includes_all_detectors(self, training_result):
        comparison = run_detector_comparison(
            training_result.train_frame,
            training_result.test_frame,
            lstm_detector=training_result.detector,
        )
        assert set(comparison.rows["detector"]) == {
            "Z-Score",
            "MAD",
            "Grubbs",
            "Isolation Forest",
            "Spectral Residual",
            "Ensemble",
            "LSTM Autoencoder",
        }
        assert len(comparison.rows) == 7
        assert comparison.rows["test_f1"].between(0, 1).all()
        assert len(comparison.errors["LSTM Autoencoder"]) == len(
            training_result.test_frame
        )

    def test_unlabelled_data_has_no_metrics(self, training_result, prepared_frame):
        unlabelled = prepared_frame.copy()
        unlabelled["is_anomaly"] = pd.array([pd.NA] * len(unlabelled), dtype="Int64")
        prediction = predict_with_detector(training_result.detector, unlabelled)
        assert prediction.metrics is None
        assert len(prediction.result.labels) == len(unlabelled)

    def test_loaded_model_uses_recorded_split(
        self, training_result, prepared_frame, tmp_path
    ):
        path = save_lstm_artifact(
            training_result.detector,
            tmp_path,
            dataset_frame=prepared_frame,
            train_points=len(training_result.train_frame),
        )
        loaded = load_lstm_artifact(path)
        parts = evaluate_loaded_model(loaded, prepared_frame)

        assert parts["split_matched"] is True
        assert set(parts) == {"train", "test", "split_matched"}
        assert parts["train"].metrics is not None
        assert parts["test"].metrics is not None

    def test_predict_requires_fitted_detector(self, prepared_frame):
        with pytest.raises(RuntimeError, match="fitted"):
            predict_with_detector(LSTMAutoencoderDetector(sequence_length=12), prepared_frame)
