import numpy as np
import pytest

from spectral.lab.synthetic import (
    ANOMALY_TYPES,
    SyntheticConfig,
    generate_synthetic_timeseries,
    parse_anomaly_types,
)


class TestSyntheticGenerator:
    @pytest.mark.parametrize("anomaly_type", list(ANOMALY_TYPES))
    def test_each_type_produces_labels(self, anomaly_type):
        config = SyntheticConfig(
            n_points=300,
            anomaly_ratio=0.08,
            anomaly_types=(anomaly_type,),
            seed=11,
        )
        dataset = generate_synthetic_timeseries(config)

        assert dataset.labels.sum() > 0
        assert (dataset.frame["anomaly_type"] == anomaly_type).sum() > 0
        assert all(event.anomaly_type == anomaly_type for event in dataset.events)
        if anomaly_type == "missing":
            assert dataset.frame["value"].isna().any()

    def test_event_spans_match_labels_and_do_not_overlap(self):
        dataset = generate_synthetic_timeseries(
            SyntheticConfig(n_points=500, anomaly_ratio=0.15, seed=5)
        )

        covered = np.zeros(len(dataset.frame), dtype=int)
        for event in dataset.events:
            covered[event.start : event.end] += 1
            assert (dataset.frame["is_anomaly"].iloc[event.start : event.end] == 1).all()
        assert covered.max() <= 1
        assert (covered == 1).sum() == dataset.labels.sum()

    def test_same_seed_is_reproducible_and_different_seed_differs(self):
        config = SyntheticConfig(n_points=400, anomaly_ratio=0.08, seed=7)
        first = generate_synthetic_timeseries(config)
        second = generate_synthetic_timeseries(config)
        other = generate_synthetic_timeseries(
            SyntheticConfig(n_points=400, anomaly_ratio=0.08, seed=8)
        )

        assert first.frame.equals(second.frame)
        assert not first.frame["value"].equals(other.frame["value"])

    def test_zero_ratio_has_no_anomalies(self):
        dataset = generate_synthetic_timeseries(
            SyntheticConfig(n_points=200, anomaly_ratio=0.0, seed=1)
        )
        assert dataset.labels.sum() == 0
        assert dataset.events == []

    def test_summary_counts_and_events_frame(self):
        dataset = generate_synthetic_timeseries(
            SyntheticConfig(n_points=300, anomaly_ratio=0.1, seed=3)
        )
        summary = dataset.summary()
        assert summary["anomaly_points"] == int(dataset.labels.sum())
        events = dataset.events_frame()
        assert len(events) == len(dataset.events)
        assert list(events.columns) == [
            "type",
            "start",
            "end",
            "length",
            "start_timestamp",
            "end_timestamp",
        ]

    def test_invalid_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="anomaly_ratio"):
            SyntheticConfig(anomaly_ratio=0.9).validate()
        with pytest.raises(ValueError, match="unknown anomaly"):
            SyntheticConfig(anomaly_types=("nope",)).validate()
        with pytest.raises(ValueError, match="at least 20"):
            SyntheticConfig(n_points=10).validate()

    def test_parse_anomaly_types(self):
        assert parse_anomaly_types("spike,drop") == ("spike", "drop")
        assert set(parse_anomaly_types(None)) == set(ANOMALY_TYPES)
        with pytest.raises(ValueError, match="unknown anomaly"):
            parse_anomaly_types("spike,nope")
