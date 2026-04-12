from __future__ import annotations

import logging

import numpy as np

from spectral.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Combines multiple detectors using voting, averaging, or max strategies.

    Not a BaseDetector subclass since it orchestrates other detectors
    rather than implementing its own detection logic.
    """

    METHODS = ("voting", "average", "max")

    def __init__(
        self,
        detectors: list[BaseDetector],
        method: str = "voting",
        min_detectors: int = 2,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {self.METHODS}")
        if min_detectors > len(detectors):
            raise ValueError(
                f"min_detectors ({min_detectors}) > number of detectors ({len(detectors)})"
            )

        self.detectors = detectors
        self.method = method
        self.min_detectors = min_detectors

    @property
    def name(self) -> str:
        return f"Ensemble({self.method})"

    def fit(self, data: np.ndarray) -> EnsembleDetector:
        for detector in self.detectors:
            detector.fit(data)
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        results = []
        for detector in self.detectors:
            result = detector.detect(data)
            results.append(result)
            logger.debug(
                f"{detector.name}: {result.anomaly_count} anomalies "
                f"(ratio={result.anomaly_ratio:.4f})"
            )

        all_scores = np.stack([r.scores for r in results])
        all_labels = np.stack([r.labels for r in results])

        if self.method == "voting":
            vote_counts = all_labels.sum(axis=0)
            labels = (vote_counts >= self.min_detectors).astype(int)
            # score = fraction of detectors that flagged
            scores = vote_counts / len(self.detectors)

        elif self.method == "average":
            scores = all_scores.mean(axis=0)
            # threshold at 0.5 for averaged scores
            labels = (scores > 0.5).astype(int)

        elif self.method == "max":
            scores = all_scores.max(axis=0)
            # use max label too (any detector flags it)
            labels = all_labels.max(axis=0)

        per_detector = {
            r.metadata.get("detector", d.name): {
                "anomaly_count": r.anomaly_count,
                "anomaly_ratio": r.anomaly_ratio,
            }
            for d, r in zip(self.detectors, results)
        }

        logger.info(
            f"Ensemble({self.method}): {int(labels.sum())} anomalies "
            f"from {len(self.detectors)} detectors"
        )

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.min_detectors / len(self.detectors),
            metadata={
                "detector": "ensemble",
                "method": self.method,
                "n_detectors": len(self.detectors),
                "per_detector": per_detector,
                "anomaly_count": int(labels.sum()),
            },
        )
