import numpy as np
import pytest

from spectral.detectors import (
    EnsembleDetector,
    GrubbsDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    MADDetector,
    SpectralResidualDetector,
    ZScoreDetector,
)


def _fast_lstm() -> LSTMAutoencoderDetector:
    return LSTMAutoencoderDetector(
        hidden_dim=4, sequence_length=5, epochs=1, batch_size=8, threshold_sigma=2.0
    )


DETECTOR_FACTORIES = [
    pytest.param(ZScoreDetector, id="zscore"),
    pytest.param(MADDetector, id="mad"),
    pytest.param(GrubbsDetector, id="grubbs"),
    pytest.param(IsolationForestDetector, id="isolation_forest"),
    pytest.param(lambda: SpectralResidualDetector(window_size=8), id="spectral"),
    pytest.param(_fast_lstm, id="lstm"),
]


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_empty_input_raises_clear_value_error(factory):
    with pytest.raises(ValueError, match="at least"):
        factory().fit_detect(np.array([], dtype=float))


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_two_dimensional_input_is_rejected(factory):
    with pytest.raises(ValueError, match="1D"):
        factory().fit_detect(np.ones((12, 2)))


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_non_finite_input_is_rejected(factory):
    data = np.sin(np.arange(30) * 0.2)
    data[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        factory().fit_detect(data)


@pytest.mark.parametrize("factory", DETECTOR_FACTORIES)
def test_finite_input_returns_one_label_per_point(factory):
    data = np.sin(np.arange(40) * 0.2) + np.random.default_rng(0).standard_normal(40) * 0.01
    detector = factory()
    result = detector.fit_detect(data)
    assert result.labels.shape == data.shape
    assert result.scores.shape == data.shape
    assert np.isfinite(result.scores).all()
    assert result.anomaly_count == int(result.labels.sum())


def test_ensemble_propagates_input_validation():
    ensemble = EnsembleDetector([ZScoreDetector(), MADDetector()])
    with pytest.raises(ValueError, match="1D"):
        ensemble.fit_detect(np.ones((10, 3)))


def test_grubbs_requires_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        GrubbsDetector().fit_detect(np.array([1.0, 2.0]))
