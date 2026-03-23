from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from spectral.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class ZScoreDetector(BaseDetector):
    """Anomaly detection using z-score thresholding.

    Points are flagged when their absolute z-score exceeds a threshold.
    Simple, fast, works well for roughly Gaussian data without heavy tails.
    """

    def __init__(self, threshold: float = 3.0, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
        self._mean = None
        self._std = None

    def fit(self, data: np.ndarray) -> ZScoreDetector:
        data = np.asarray(data, dtype=np.float64)
        self._mean = np.nanmean(data)
        self._std = np.nanstd(data)
        if self._std == 0:
            self._std = 1e-10
        self._fitted = True
        logger.info(f"ZScore fit: mean={self._mean:.4f}, std={self._std:.4f}")
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = np.asarray(data, dtype=np.float64)
        if not self._fitted:
            self.fit(data)

        z_scores = np.abs((data - self._mean) / self._std)
        # normalize scores to [0, 1] range
        max_z = z_scores.max() if z_scores.max() > 0 else 1.0
        scores = np.clip(z_scores / max_z, 0, 1)
        labels = (z_scores > self.threshold).astype(int)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            metadata={
                "detector": "zscore",
                "mean": float(self._mean),
                "std": float(self._std),
                "max_zscore": float(z_scores.max()),
            },
        )


class MADDetector(BaseDetector):
    """Anomaly detection using Median Absolute Deviation.

    More robust than z-score for data with existing outliers, since
    the median is less sensitive to extreme values than the mean.
    Uses the modified z-score: 0.6745 * (x - median) / MAD.
    """

    def __init__(self, threshold: float = 3.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
        self._median = None
        self._mad = None

    def fit(self, data: np.ndarray) -> MADDetector:
        data = np.asarray(data, dtype=np.float64)
        self._median = np.nanmedian(data)
        self._mad = np.nanmedian(np.abs(data - self._median))
        if self._mad == 0:
            self._mad = 1e-10
        self._fitted = True
        logger.info(f"MAD fit: median={self._median:.4f}, mad={self._mad:.4f}")
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = np.asarray(data, dtype=np.float64)
        if not self._fitted:
            self.fit(data)

        # modified z-score using consistency constant for normal distribution
        modified_z = 0.6745 * np.abs(data - self._median) / self._mad
        max_z = modified_z.max() if modified_z.max() > 0 else 1.0
        scores = np.clip(modified_z / max_z, 0, 1)
        labels = (modified_z > self.threshold).astype(int)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            metadata={
                "detector": "mad",
                "median": float(self._median),
                "mad": float(self._mad),
                "max_modified_z": float(modified_z.max()),
            },
        )


class GrubbsDetector(BaseDetector):
    """Anomaly detection using Grubbs' test for outliers.

    Iteratively applies Grubbs' test to find and remove outliers
    one at a time. Best suited for small to medium datasets where
    you want statistically rigorous outlier detection.
    """

    def __init__(self, alpha: float = 0.05, max_outliers: int | None = None, **kwargs):
        super().__init__(alpha=alpha, max_outliers=max_outliers, **kwargs)
        self.alpha = alpha
        self.max_outliers = max_outliers

    def fit(self, data: np.ndarray) -> GrubbsDetector:
        self._fitted = True
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        labels = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=np.float64)

        working_data = data.copy()
        working_indices = np.arange(n)
        found = 0
        max_iter = self.max_outliers or n

        while len(working_data) > 2 and found < max_iter:
            mean = np.mean(working_data)
            std = np.std(working_data, ddof=1)
            if std == 0:
                break

            # find point furthest from mean
            abs_diff = np.abs(working_data - mean)
            max_idx = np.argmax(abs_diff)
            g_stat = abs_diff[max_idx] / std

            # critical value from t-distribution
            t_crit = stats.t.ppf(1 - self.alpha / (2 * len(working_data)), len(working_data) - 2)
            n_curr = len(working_data)
            g_crit = ((n_curr - 1) / np.sqrt(n_curr)) * np.sqrt(
                t_crit**2 / (n_curr - 2 + t_crit**2)
            )

            if g_stat <= g_crit:
                break

            original_idx = working_indices[max_idx]
            labels[original_idx] = 1
            scores[original_idx] = min(g_stat / g_crit, 1.0)
            found += 1

            working_data = np.delete(working_data, max_idx)
            working_indices = np.delete(working_indices, max_idx)

        # fill non-outlier scores based on distance from mean
        mean = np.mean(data)
        std = np.std(data, ddof=1) if np.std(data, ddof=1) > 0 else 1.0
        for i in range(n):
            if labels[i] == 0:
                scores[i] = min(abs(data[i] - mean) / (std * 4), 0.99)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.alpha,
            metadata={
                "detector": "grubbs",
                "outliers_found": found,
                "alpha": self.alpha,
            },
        )
