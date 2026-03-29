import numpy as np
import pytest

from spectral.config.schema import DetectorConfig, PipelineConfig
from spectral.pipeline.runner import Pipeline, _severity_from_score


class TestPipeline:
    def test_full_pipeline_end_to_end(self, pipeline_config, sample_with_anomalies):
        data, expected = sample_with_anomalies
        pipeline = Pipeline(pipeline_config)
        anomalies = pipeline.detect(data)

        # should detect at least some of the injected anomalies
        assert len(anomalies) > 0

        detected_indices = {a.index for a in anomalies}
        hits = sum(1 for idx in expected if idx in detected_indices)
        assert hits >= 2  # at least 2 out of 5

    def test_pipeline_with_disabled_detectors(self, sample_with_anomalies):
        data, _ = sample_with_anomalies
        config = PipelineConfig(
            detectors=[
                DetectorConfig(name="zscore", enabled=True, params={"threshold": 3.0}),
                DetectorConfig(name="isolation_forest", enabled=False),
            ]
        )
        pipeline = Pipeline(config)
        assert len(pipeline._detectors) == 1
        assert pipeline._ensemble is None

        anomalies = pipeline.detect(data)
        assert len(anomalies) > 0

    def test_streaming_interface(self, pipeline_config):
        pipeline = Pipeline(pipeline_config)

        # fit on some training data first
        rng = np.random.default_rng(42)
        train_data = np.sin(np.arange(500) * 0.1) + rng.standard_normal(500) * 0.1
        pipeline.fit(train_data)

        # stream normal points, should get no anomalies for most
        anomaly_count = 0
        for i in range(200):
            value = np.sin(i * 0.1) + rng.standard_normal() * 0.1
            result = pipeline.detect_streaming(value)
            if result:
                anomaly_count += 1

        assert anomaly_count < 50  # mostly normal data

    def test_stats_tracking(self, pipeline_config, sample_normal_data):
        pipeline = Pipeline(pipeline_config)
        pipeline.detect(sample_normal_data)

        stats = pipeline.stats
        assert stats["points_processed"] == len(sample_normal_data)
        assert stats["fitted"] is True
        assert len(stats["detectors"]) == 2


class TestSeverity:
    def test_critical(self):
        assert _severity_from_score(0.95) == "critical"

    def test_high(self):
        assert _severity_from_score(0.75) == "high"

    def test_warning(self):
        assert _severity_from_score(0.55) == "warning"

    def test_low(self):
        assert _severity_from_score(0.3) == "low"
