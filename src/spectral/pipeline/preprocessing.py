from __future__ import annotations

import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize data using the specified method.

    Methods:
        minmax: scales to [0, 1]
        zscore: centers to mean=0, std=1
        robust: uses median and IQR, resilient to outliers
    """
    data = np.asarray(data, dtype=np.float64)

    if method == "minmax":
        dmin, dmax = data.min(), data.max()
        if dmax - dmin == 0:
            return np.zeros_like(data)
        return (data - dmin) / (dmax - dmin)

    elif method == "zscore":
        mean, std = data.mean(), data.std()
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std

    elif method == "robust":
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(data)
        return (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detrend(data: np.ndarray) -> np.ndarray:
    """Remove linear trend from data using least-squares fit."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    x = np.arange(n)
    coeffs = np.polyfit(x, data, 1)
    trend = np.polyval(coeffs, x)
    return data - trend


def decompose_seasonality(
    data: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose time series into trend, seasonal, and residual components.

    Uses a moving average for trend extraction and period-wise averaging
    for the seasonal component. Simple but effective for regular seasonality.

    Returns:
        (trend, seasonal, residual) tuple of arrays
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    # trend via centered moving average
    if period % 2 == 0:
        # for even period, use two-pass average
        kernel = np.ones(period) / period
        first_pass = np.convolve(data, kernel, mode="same")
        trend = np.convolve(first_pass, np.array([0.5, 0.5]), mode="same")
    else:
        kernel = np.ones(period) / period
        trend = np.convolve(data, kernel, mode="same")

    # seasonal: average the detrended values by position in cycle
    detrended = data - trend
    seasonal = np.zeros(n)
    for i in range(period):
        cycle_values = detrended[i::period]
        seasonal[i::period] = cycle_values.mean()

    # center the seasonal component
    seasonal -= seasonal[:period].mean()

    residual = data - trend - seasonal
    return trend, seasonal, residual


def fill_missing(data: np.ndarray, method: str = "interpolate") -> np.ndarray:
    """Handle NaN values in the data.

    Methods:
        interpolate: linear interpolation between valid points
        forward: forward fill (last valid observation carried forward)
        mean: fill with series mean
    """
    data = np.asarray(data, dtype=np.float64).copy()
    nans = np.isnan(data)

    if not nans.any():
        return data

    if method == "interpolate":
        valid = ~nans
        if valid.sum() < 2:
            data[nans] = 0.0
            return data
        indices = np.arange(len(data))
        data[nans] = np.interp(indices[nans], indices[valid], data[valid])

    elif method == "forward":
        for i in range(1, len(data)):
            if np.isnan(data[i]):
                data[i] = data[i - 1]
        # backfill the start if it begins with NaN
        first_valid = np.where(~nans)[0]
        if len(first_valid) > 0:
            data[: first_valid[0]] = data[first_valid[0]]

    elif method == "mean":
        data[nans] = np.nanmean(data)

    else:
        raise ValueError(f"Unknown fill method: {method}")

    return data


class Preprocessor:
    """Chains preprocessing steps based on configuration."""

    def __init__(
        self,
        normalize_method: str | None = "minmax",
        should_detrend: bool = False,
        seasonal_period: int | None = None,
        fill_method: str = "interpolate",
    ):
        self.normalize_method = normalize_method
        self.should_detrend = should_detrend
        self.seasonal_period = seasonal_period
        self.fill_method = fill_method

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all configured preprocessing steps."""
        result = np.asarray(data, dtype=np.float64)

        # handle missing values first
        result = fill_missing(result, self.fill_method)

        # decompose and use residual if seasonal period is set
        if self.seasonal_period is not None:
            _, _, result = decompose_seasonality(result, self.seasonal_period)

        # remove linear trend
        if self.should_detrend:
            result = detrend(result)

        # normalize
        if self.normalize_method:
            result = normalize(result, self.normalize_method)

        return result
