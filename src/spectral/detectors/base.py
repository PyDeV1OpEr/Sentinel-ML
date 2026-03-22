from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from running a detector on a time series.

    Attributes:
        scores: anomaly score per point, normalized to [0, 1]
        labels: binary labels, 1 = anomaly, 0 = normal
        threshold: the score threshold used for labeling
        metadata: extra info from the detector (algorithm-specific)
    """

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    metadata: dict = field(default_factory=dict)

    @property
    def anomaly_indices(self) -> np.ndarray:
        return np.where(self.labels == 1)[0]

    @property
    def anomaly_count(self) -> int:
        return int(self.labels.sum())

    @property
    def anomaly_ratio(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float(self.labels.sum() / len(self.labels))


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors.

    Every detector must implement fit() and detect(). The fit step
    is optional for some algorithms (statistical tests that don't need
    training), but the interface stays consistent.
    """

    def __init__(self, **params):
        self.params = params
        self._fitted = False
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def fit(self, data: np.ndarray) -> BaseDetector:
        """Fit the detector on training data (assumed mostly normal).

        Args:
            data: 1D array of time-series values.

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def detect(self, data: np.ndarray) -> DetectionResult:
        """Run anomaly detection on the given data.

        Args:
            data: 1D array of time-series values.

        Returns:
            DetectionResult with scores, labels, and threshold.
        """
        ...

    def fit_detect(self, data: np.ndarray) -> DetectionResult:
        """Convenience method: fit on data and detect in one step."""
        self.fit(data)
        return self.detect(data)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"
