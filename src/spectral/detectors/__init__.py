from spectral.detectors.base import BaseDetector, DetectionResult
from spectral.detectors.statistical import ZScoreDetector, MADDetector, GrubbsDetector
from spectral.detectors.isolation_forest import IsolationForestDetector
from spectral.detectors.spectral_residual import SpectralResidualDetector
from spectral.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from spectral.detectors.ensemble import EnsembleDetector

DETECTORS = {
    "zscore": ZScoreDetector,
    "mad": MADDetector,
    "grubbs": GrubbsDetector,
    "isolation_forest": IsolationForestDetector,
    "spectral_residual": SpectralResidualDetector,
    "lstm_autoencoder": LSTMAutoencoderDetector,
}


def create_detector(name: str, **params) -> BaseDetector:
    """Create a detector instance by name."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTORS.keys())}")
    return DETECTORS[name](**params)


__all__ = [
    "BaseDetector",
    "DetectionResult",
    "ZScoreDetector",
    "MADDetector",
    "GrubbsDetector",
    "IsolationForestDetector",
    "SpectralResidualDetector",
    "LSTMAutoencoderDetector",
    "EnsembleDetector",
    "DETECTORS",
    "create_detector",
]
