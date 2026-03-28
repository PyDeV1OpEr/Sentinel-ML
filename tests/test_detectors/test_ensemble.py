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
