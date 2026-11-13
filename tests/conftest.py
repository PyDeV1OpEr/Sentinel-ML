from datetime import datetime, timedelta

import numpy as np
import pytest

from spectral.config.schema import (
    DetectorConfig,
    EnsembleConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from spectral.ingestion.base import DataPoint


@pytest.fixture
def sample_normal_data():
    """1000-point sinusoidal signal with light noise, no anomalies."""
    rng = np.random.default_rng(42)
    t = np.arange(1000)
    signal = np.sin(2 * np.pi * t / 100) + rng.standard_normal(1000) * 0.1
    return signal


@pytest.fixture
def sample_with_anomalies():
    """Normal data with 5 injected spikes at known positions."""
    rng = np.random.default_rng(42)
    t = np.arange(1000)
    signal = np.sin(2 * np.pi * t / 100) + rng.standard_normal(1000) * 0.1

    # inject obvious spikes
    anomaly_indices = [100, 300, 500, 700, 900]
    for idx in anomaly_indices:
        signal[idx] += 8.0  # large spike, should be easy to detect

    return signal, anomaly_indices


@pytest.fixture
def anomaly_positions():
    """Ground truth anomaly indices matching sample_with_anomalies."""
    return [100, 300, 500, 700, 900]


@pytest.fixture
def pipeline_config():
    """Basic pipeline config for testing."""
    return PipelineConfig(
        name="test",
        window_size=100,
        step_size=1,
        preprocessing=PreprocessingConfig(normalize=True, method="minmax"),
        detectors=[
            DetectorConfig(name="zscore", params={"threshold": 3.0}),
            DetectorConfig(name="isolation_forest", params={"contamination": 0.05}),
        ],
        ensemble=EnsembleConfig(method="voting", min_detectors=2),
    )


@pytest.fixture
def sample_datapoints():
    """List of DataPoint objects for ingestion tests."""
    start = datetime(2026, 1, 1)
    rng = np.random.default_rng(42)
    points = []
    for i in range(100):
        value = np.sin(2 * np.pi * i / 50) + rng.standard_normal() * 0.1
        points.append(
            DataPoint(
                timestamp=start + timedelta(minutes=i),
                value=float(value),
                series_id="test-series",
            )
        )
    return points
