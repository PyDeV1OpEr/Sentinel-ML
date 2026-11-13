from __future__ import annotations

import logging

import numpy as np
from scipy.fft import fft, ifft

from spectral.detectors.base import BaseDetector, DetectionResult, validate_series

logger = logging.getLogger(__name__)


class SpectralResidualDetector(BaseDetector):
    """Anomaly detection using Spectral Residual transform.

    Based on the SR method from "Time-Series Anomaly Detection Service at
    Microsoft" (Ren et al., 2019). The core idea: normal patterns produce
    predictable frequency components, while anomalies create spectral
    residuals (unexpected frequency energy) that show up as spikes in a
    saliency map.

    Steps:
    1. Compute FFT of the input series
    2. Calculate log amplitude spectrum
    3. Compute spectral residual = log amplitude - averaged log amplitude
    4. Reconstruct via inverse FFT to get saliency map
    5. Threshold the saliency map to flag anomalies
    """

    def __init__(
        self,
        window_size: int = 24,
        score_threshold: float = 3.0,
        **kwargs,
    ):
        super().__init__(window_size=window_size, score_threshold=score_threshold, **kwargs)
        self.window_size = window_size
        self.score_threshold = score_threshold

    def fit(self, data: np.ndarray) -> SpectralResidualDetector:
        # SR is a non-parametric method, no fitting needed
        self._fitted = True
        return self

    def _average_filter(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average filter to smooth the log spectrum."""
        if window <= 0:
            return values
        kernel = np.ones(window) / window
        # pad edges to keep output length consistent
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(values)]

    def _spectral_residual(self, data: np.ndarray) -> np.ndarray:
        """Compute the spectral residual of the input series."""
        freq = fft(data)
        amplitude = np.abs(freq)

        # avoid log(0) by adding small epsilon
        log_amplitude = np.log(amplitude + 1e-10)
        phase = np.angle(freq)

        # spectral residual: difference between log amplitude and its
        # smoothed version. captures unexpected frequency energy.
        smoothed = self._average_filter(log_amplitude, self.window_size)
        residual = log_amplitude - smoothed

        return residual, phase

    def _saliency_map(self, residual: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Reconstruct saliency map from spectral residual and phase."""
        freq = np.exp(residual) * np.exp(1j * phase)
        saliency = np.abs(ifft(freq))
        return saliency.real

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = validate_series(data)

        residual, phase = self._spectral_residual(data)
        saliency = self._saliency_map(residual, phase)

        # normalize saliency scores
        mean_sal = np.mean(saliency)
        std_sal = np.std(saliency)
        if std_sal == 0:
            std_sal = 1e-10

        # anomaly score based on how many stds above mean
        z_saliency = (saliency - mean_sal) / std_sal
        labels = (z_saliency > self.score_threshold).astype(int)

        # normalize to [0, 1]
        max_z = z_saliency.max() if z_saliency.max() > 0 else 1.0
        scores = np.clip(z_saliency / max_z, 0.0, 1.0)

        logger.info(
            f"SpectralResidual: {labels.sum()} anomalies detected, "
            f"mean_saliency={mean_sal:.6f}, std_saliency={std_sal:.6f}"
        )

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.score_threshold,
            metadata={
                "detector": "spectral_residual",
                "window_size": self.window_size,
                "mean_saliency": float(mean_sal),
                "std_saliency": float(std_sal),
                "anomaly_count": int(labels.sum()),
            },
        )
