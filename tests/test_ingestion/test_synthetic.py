import numpy as np
import pytest

from spectral.ingestion.synthetic import SyntheticSource


class TestSyntheticSource:
    def test_generates_correct_count(self):
        source = SyntheticSource(seed=42)
        points, labels = source.generate(n_points=500)

        assert len(points) == 500
        assert len(labels) == 500

    def test_anomaly_ratio_respected(self):
        source = SyntheticSource(seed=42)
        _, labels = source.generate(n_points=10000, anomaly_ratio=0.05)

        actual_ratio = labels.sum() / len(labels)
        # allow some tolerance since collective anomalies span multiple points
        assert 0.02 < actual_ratio < 0.12

    def test_sine_pattern(self):
        source = SyntheticSource(seed=42)
        points, _ = source.generate(n_points=200, patterns=["sine"], noise_level=0.0, anomaly_ratio=0.0)

        values = np.array([p.value for p in points])
        # should have periodic structure
        assert values.std() > 0

    def test_ground_truth_labels(self):
        source = SyntheticSource(seed=42)
        points, labels = source.generate(n_points=1000, anomaly_ratio=0.05)

        # metadata should mark anomalies
        for i, point in enumerate(points):
            assert point.metadata["is_anomaly"] == bool(labels[i])

    def test_deterministic_with_seed(self):
        s1 = SyntheticSource(seed=123)
        s2 = SyntheticSource(seed=123)

        p1, l1 = s1.generate(n_points=100)
        p2, l2 = s2.generate(n_points=100)

        v1 = [p.value for p in p1]
        v2 = [p.value for p in p2]
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(l1, l2)

    def test_timestamps_are_sequential(self):
        source = SyntheticSource(seed=42)
        points, _ = source.generate(n_points=100)

        for i in range(1, len(points)):
            assert points[i].timestamp > points[i - 1].timestamp
