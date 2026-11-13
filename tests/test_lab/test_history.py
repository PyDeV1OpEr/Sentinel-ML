import json

import numpy as np

from spectral.lab.history import clear_history, load_history, record_experiment


class TestExperimentHistory:
    def test_round_trip_and_cleanup(self, tmp_path):
        path = tmp_path / "experiments.jsonl"
        record_experiment(
            kind="train",
            dataset_source="synthetic seed=42",
            dataset_fingerprint="abc123",
            detector="LSTM Autoencoder",
            model_version="model_001",
            params={"epochs": 10, "seed": 42},
            metrics={"f1": 0.91, "precision": np.float64(0.9)},
            duration_seconds=1.25,
            path=path,
        )
        record_experiment(
            kind="evaluate",
            detector="Z-Score",
            metrics={"f1": float("nan")},
            path=path,
        )

        history = load_history(path)
        assert len(history) == 2
        assert set(history["kind"]) == {"train", "evaluate"}
        first = history.iloc[0]
        assert first["model_version"] == "model_001"
        assert first["params"]["epochs"] == 10
        assert first["metrics"]["precision"] == 0.9
        # NaN is written as null, so the JSON stays standards-compliant.
        assert history.iloc[1]["metrics"]["f1"] is None

        assert clear_history(path) == 2
        assert not path.exists()
        assert load_history(path).empty

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "experiments.jsonl"
        path.write_text(
            json.dumps({"kind": "train", "metrics": {"f1": 0.5}}) + "\n"
            + "{not-json\n"
            + json.dumps({"kind": "evaluate"}) + "\n",
            encoding="utf-8",
        )
        history = load_history(path)
        assert len(history) == 2
        assert set(history["kind"]) == {"train", "evaluate"}

    def test_missing_file_is_empty(self, tmp_path):
        history = load_history(tmp_path / "absent.jsonl")
        assert history.empty
        assert clear_history(tmp_path / "absent.jsonl") == 0
