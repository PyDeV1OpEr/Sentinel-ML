from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from spectral.config.schema import PipelineConfig
from spectral.detectors import create_detector
from spectral.detectors.ensemble import EnsembleDetector
from spectral.pipeline.preprocessing import Preprocessor
from spectral.pipeline.window import SlidingWindow, StreamBuffer

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """A detected anomaly with context."""

    timestamp: datetime | None
    index: int
    value: float
    score: float
    detector: str
    severity: str
    metadata: dict = field(default_factory=dict)


def _severity_from_score(score: float) -> str:
    """Map anomaly score to severity level."""
    if score >= 0.9:
        return "critical"
    elif score >= 0.7:
        return "high"
    elif score >= 0.5:
        return "warning"
    return "low"


class Pipeline:
    """Central orchestrator connecting preprocessing, detection, and output.

    Loads detectors from config, applies preprocessing, runs detection
    (either batch or streaming), and returns structured anomaly results.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._preprocessor = Preprocessor(
            normalize_method=config.preprocessing.method if config.preprocessing.normalize else None,
            should_detrend=config.preprocessing.detrend,
            seasonal_period=(
                config.preprocessing.seasonal_period
                if config.preprocessing.decompose_seasonality
                else None
            ),
        )

        # build detectors from config
        self._detectors = []
        for det_config in config.detectors:
            if not det_config.enabled:
                continue
            detector = create_detector(det_config.name, **det_config.params)
            self._detectors.append(detector)
            logger.info(f"Loaded detector: {detector}")

        # ensemble
        self._ensemble = None
        if len(self._detectors) > 1:
            self._ensemble = EnsembleDetector(
                detectors=self._detectors,
                method=config.ensemble.method,
                min_detectors=config.ensemble.min_detectors,
            )

        # streaming buffer
        self._stream_buffer = StreamBuffer(
            size=config.window_size,
            step=config.step_size,
        )

        self._fitted = False
        self._points_processed = 0
        self._anomalies_detected = 0

    @property
    def detector_names(self) -> list[str]:
        return [d.name for d in self._detectors]

    @property
    def stats(self) -> dict:
        return {
            "points_processed": self._points_processed,
            "anomalies_detected": self._anomalies_detected,
            "detectors": self.detector_names,
            "fitted": self._fitted,
        }

    def fit(self, data: np.ndarray) -> Pipeline:
        """Fit all detectors on training data (assumed mostly normal)."""
        data = np.asarray(data, dtype=np.float64)
        processed = self._preprocessor.transform(data)

        if self._ensemble:
            self._ensemble.fit(processed)
        else:
            for detector in self._detectors:
                detector.fit(processed)

        self._fitted = True
        logger.info(f"Pipeline fitted on {len(data)} points")
        return self

    def detect(
        self,
        data: np.ndarray,
        timestamps: list[datetime] | None = None,
    ) -> list[Anomaly]:
        """Run anomaly detection on a batch of data.

        Args:
            data: 1D array of values
            timestamps: optional datetime for each point

        Returns:
            list of Anomaly objects for detected anomalies
        """
        data = np.asarray(data, dtype=np.float64)

        if not self._fitted:
            self.fit(data)

        processed = self._preprocessor.transform(data)

        # run detection
        if self._ensemble:
            result = self._ensemble.detect(processed)
            detector_name = "ensemble"
        elif len(self._detectors) == 1:
            result = self._detectors[0].detect(processed)
            detector_name = self._detectors[0].name
        else:
            # shouldn't happen, but handle gracefully
            logger.warning("No detectors configured")
            return []

        # build anomaly objects from detection result
        anomalies = []
        for idx in result.anomaly_indices:
            ts = timestamps[idx] if timestamps else None
            anomaly = Anomaly(
                timestamp=ts,
                index=int(idx),
                value=float(data[idx]),
                score=float(result.scores[idx]),
                detector=detector_name,
                severity=_severity_from_score(float(result.scores[idx])),
                metadata={"original_value": float(data[idx])},
            )
            anomalies.append(anomaly)

        self._points_processed += len(data)
        self._anomalies_detected += len(anomalies)
        logger.info(f"Detected {len(anomalies)} anomalies in {len(data)} points")
        return anomalies

    def detect_streaming(
        self,
        value: float,
        timestamp: datetime | None = None,
    ) -> Anomaly | None:
        """Process a single data point in streaming mode.

        Returns an Anomaly if the point triggers detection, None otherwise.
        Only runs detection when a full window has accumulated.
        """
        window = self._stream_buffer.push(value)
        if window is None:
            return None

        processed = self._preprocessor.transform(window)

        if self._ensemble:
            result = self._ensemble.detect(processed)
            detector_name = "ensemble"
        elif self._detectors:
            result = self._detectors[0].detect(processed)
            detector_name = self._detectors[0].name
        else:
            return None

        # check the last point in the window (most recent)
        last_idx = len(processed) - 1
        self._points_processed += 1

        if result.labels[last_idx] == 1:
            anomaly = Anomaly(
                timestamp=timestamp,
                index=self._points_processed,
                value=value,
                score=float(result.scores[last_idx]),
                detector=detector_name,
                severity=_severity_from_score(float(result.scores[last_idx])),
            )
            self._anomalies_detected += 1
            return anomaly

        return None
