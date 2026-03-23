from __future__ import annotations

import logging

import numpy as np

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
