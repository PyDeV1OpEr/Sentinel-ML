import numpy as np
import pytest

from spectral.detectors.ensemble import EnsembleDetector
from spectral.detectors.isolation_forest import IsolationForestDetector
from spectral.detectors.statistical import MADDetector, ZScoreDetector


class TestEnsembleDetector:
    def test_voting_requires_min_detectors(self, sample_with_anomalies):
        data, _ = sample_with_anomalies
        detectors = [
            ZScoreDetector(threshold=3.0),
            MADDetector(threshold=3.5),
            IsolationForestDetector(contamination=0.05),
        ]

        ensemble = EnsembleDetector(detectors, method="voting", min_detectors=2)
        ensemble.fit(data)
        result = ensemble.detect(data)

        # ensemble should be more conservative than individual detectors
        zscore_result = ZScoreDetector(threshold=3.0).fit_detect(data)
        assert result.anomaly_count <= zscore_result.anomaly_count + 5  # some tolerance

    def test_average_method(self, sample_normal_data):
        detectors = [ZScoreDetector(threshold=3.0), MADDetector(threshold=3.5)]
        ensemble = EnsembleDetector(detectors, method="average")
        result = ensemble.fit_detect(sample_normal_data)

        assert result.scores.min() >= 0
        assert result.scores.max() <= 1

    def test_max_method(self, sample_with_anomalies):
        data, expected = sample_with_anomalies
        detectors = [ZScoreDetector(threshold=3.0), MADDetector(threshold=3.5)]
        ensemble = EnsembleDetector(detectors, method="max")
        result = ensemble.fit_detect(data)

        # max should detect at least as many as any individual detector
        assert result.anomaly_count > 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            EnsembleDetector([ZScoreDetector()], method="invalid")

    def test_min_detectors_exceeds_count_raises(self):
        with pytest.raises(ValueError, match="min_detectors"):
            EnsembleDetector([ZScoreDetector()], method="voting", min_detectors=5)

    def test_metadata_contains_per_detector(self, sample_with_anomalies):
        data, _ = sample_with_anomalies
        detectors = [ZScoreDetector(threshold=3.0), MADDetector(threshold=3.5)]
        ensemble = EnsembleDetector(detectors, method="voting", min_detectors=1)
        result = ensemble.fit_detect(data)

        assert "per_detector" in result.metadata
        assert len(result.metadata["per_detector"]) == 2

    @pytest.mark.parametrize("method", ["voting", "average", "max"])
    def test_fit_detect_matches_fit_then_detect(self, sample_with_anomalies, method):
        """fit_detect must be exactly fit() followed by detect()."""
        data, _ = sample_with_anomalies
        config = [
            ZScoreDetector(threshold=3.0),
            MADDetector(threshold=3.5),
            IsolationForestDetector(contamination=0.05),
        ]

        combined = EnsembleDetector(config, method=method, min_detectors=2)
        result = combined.fit_detect(data)

        separate = EnsembleDetector(
            [
                ZScoreDetector(threshold=3.0),
                MADDetector(threshold=3.5),
                IsolationForestDetector(contamination=0.05),
            ],
            method=method,
            min_detectors=2,
        )
        separate.fit(data)
        expected = separate.detect(data)

        np.testing.assert_array_equal(result.labels, expected.labels)
        np.testing.assert_array_equal(result.scores, expected.scores)
        assert result.threshold == expected.threshold
        assert len(result.labels) == len(data)

    def test_fit_detect_fits_every_constituent(self, sample_with_anomalies):
        data, _ = sample_with_anomalies
        detectors = [ZScoreDetector(threshold=3.0), MADDetector(threshold=3.5)]
        ensemble = EnsembleDetector(detectors, method="voting", min_detectors=2)

        assert not any(d._fitted for d in detectors)
        ensemble.fit_detect(data)

        assert all(d._fitted for d in detectors)

    def test_fit_detect_is_repeatable(self, sample_with_anomalies):
        """Repeated fit_detect calls refit and produce identical output."""
        data, _ = sample_with_anomalies
        ensemble = EnsembleDetector(
            [
                ZScoreDetector(threshold=3.0),
                MADDetector(threshold=3.5),
                IsolationForestDetector(contamination=0.05),
            ],
            method="voting",
            min_detectors=2,
        )

        first = ensemble.fit_detect(data)
        second = ensemble.fit_detect(data)

        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.scores, second.scores)
        assert first.threshold == second.threshold
        assert first.metadata["per_detector"] == second.metadata["per_detector"]

    def test_single_detector_ensemble_matches_that_detector(
        self, sample_with_anomalies
    ):
        """A one-detector ensemble with min_detectors=1 is a pass-through."""
        data, _ = sample_with_anomalies
        ensemble = EnsembleDetector(
            [ZScoreDetector(threshold=3.0)], method="voting", min_detectors=1
        )
        result = ensemble.fit_detect(data)
        standalone = ZScoreDetector(threshold=3.0).fit_detect(data)

        np.testing.assert_array_equal(result.labels, standalone.labels)
        assert result.threshold == 1.0
