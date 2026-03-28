import numpy as np
import pytest

from spectral.pipeline.preprocessing import (
    Preprocessor,
    decompose_seasonality,
    detrend,
    fill_missing,
    normalize,
)


class TestNormalize:
    def test_minmax_scales_to_unit(self):
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        result = normalize(data, method="minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_zscore_centers_data(self):
        data = np.random.default_rng(42).standard_normal(1000) * 5 + 10
        result = normalize(data, method="zscore")
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_robust_uses_median_iqr(self):
        data = np.array([1, 2, 3, 4, 5, 100], dtype=float)
        result = normalize(data, method="robust")
        # median is 3.5, outlier at 100 shouldn't distort too much
        assert result[0] < result[-1]

    def test_constant_data_returns_zeros(self):
        data = np.ones(100)
        result = normalize(data, method="minmax")
        assert np.all(result == 0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            normalize(np.array([1, 2, 3]), method="invalid")


class TestDetrend:
    def test_removes_linear_trend(self):
        t = np.arange(100, dtype=float)
        data = 3 * t + 10 + np.random.default_rng(42).standard_normal(100) * 0.1
        result = detrend(data)

        # slope should be approximately 0 after detrending
        coeffs = np.polyfit(t, result, 1)
        assert abs(coeffs[0]) < 0.01


class TestDecomposeSeasonality:
    def test_extracts_seasonal_component(self):
        t = np.arange(200, dtype=float)
        seasonal = np.sin(2 * np.pi * t / 20)
        data = seasonal + 0.01 * t  # seasonal + slight trend

        trend, extracted_seasonal, residual = decompose_seasonality(data, period=20)

        # the extracted seasonal should correlate with true seasonal
        correlation = np.corrcoef(seasonal[20:-20], extracted_seasonal[20:-20])[0, 1]
        assert abs(correlation) > 0.5


class TestFillMissing:
    def test_interpolate_fills_nans(self):
        data = np.array([1, 2, np.nan, 4, 5], dtype=float)
        result = fill_missing(data, method="interpolate")
        assert not np.any(np.isnan(result))
        assert result[2] == pytest.approx(3.0)

    def test_forward_fill(self):
        data = np.array([1, 2, np.nan, np.nan, 5], dtype=float)
        result = fill_missing(data, method="forward")
        assert result[2] == 2.0
        assert result[3] == 2.0

    def test_mean_fill(self):
        data = np.array([1, 2, np.nan, 4, 5], dtype=float)
        result = fill_missing(data, method="mean")
        assert result[2] == pytest.approx(3.0)

    def test_no_nans_returns_unchanged(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result = fill_missing(data)
        np.testing.assert_array_equal(result, data)


class TestPreprocessor:
    def test_chains_operations(self):
        data = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        pp = Preprocessor(normalize_method="minmax", fill_method="interpolate")
        result = pp.transform(data)

        assert not np.any(np.isnan(result))
        assert result.min() >= 0
        assert result.max() <= 1
