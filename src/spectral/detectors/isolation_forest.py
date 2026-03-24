from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from spectral.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseDetector):
    """Anomaly detection using Isolation Forest.

    Converts a 1D time series into a feature matrix using lagged windows,
    then fits sklearn's IsolationForest on the constructed features.
    Works well for detecting both point anomalies and contextual anomalies
    that deviate from local temporal patterns.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str | int = "auto",
        window_features: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            window_features=window_features,
            random_state=random_state,
            **kwargs,
        )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.window_features = window_features
        self.random_state = random_state
        self._model = None

    def _build_features(self, data: np.ndarray) -> np.ndarray:
        """Convert 1D series to feature matrix using lagged windows.

        For each point t, the features are [x(t), x(t-1), ..., x(t-k+1)]
        where k = window_features. The first k-1 points are padded with
        the series mean to avoid losing data.
        """
        n = len(data)
        k = self.window_features
        padded = np.concatenate([np.full(k - 1, np.nanmean(data)), data])
        features = np.column_stack([padded[i : i + n] for i in range(k - 1, -1, -1)])
        return features

    def fit(self, data: np.ndarray) -> IsolationForestDetector:
        data = np.asarray(data, dtype=np.float64)
        features = self._build_features(data)

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self._model.fit(features)
        self._fitted = True
        logger.info(
            f"IsolationForest fit: n_samples={len(data)}, "
            f"n_features={features.shape[1]}, contamination={self.contamination}"
        )
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = np.asarray(data, dtype=np.float64)
        if not self._fitted:
            self.fit(data)

        features = self._build_features(data)

        # sklearn returns -1 for outliers, 1 for inliers
        raw_labels = self._model.predict(features)
        labels = (raw_labels == -1).astype(int)

        # decision_function returns signed distance to boundary
        # more negative = more anomalous
        raw_scores = self._model.decision_function(features)
        # flip and normalize to [0, 1] where 1 = most anomalous
        shifted = -raw_scores
        min_s, max_s = shifted.min(), shifted.max()
        if max_s - min_s > 0:
            scores = (shifted - min_s) / (max_s - min_s)
        else:
            scores = np.zeros_like(shifted)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.contamination,
            metadata={
                "detector": "isolation_forest",
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "anomaly_count": int(labels.sum()),
            },
        )
