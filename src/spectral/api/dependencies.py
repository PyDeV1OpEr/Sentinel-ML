from __future__ import annotations

from fastapi import Depends

from spectral.api.service import DetectionService
from spectral.config import get_default_config
from spectral.pipeline.runner import Pipeline
from spectral.storage.repository import AnomalyRepository

_pipeline: Pipeline | None = None
_repository: AnomalyRepository | None = None
_service: DetectionService | None = None


def get_pipeline() -> Pipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = get_default_config()
        _pipeline = Pipeline(config)
    return _pipeline


def set_pipeline(pipeline: Pipeline) -> None:
    """Replace the pipeline instance (used during app startup)."""
    global _pipeline
    _pipeline = pipeline


def get_repository() -> AnomalyRepository:
    """Get or create the singleton repository instance."""
    global _repository
    if _repository is None:
        _repository = AnomalyRepository()
    return _repository


def set_service(service: DetectionService | None) -> None:
    """Replace the detection service (used during app startup and in tests)."""
    global _service
    _service = service


def get_service(pipeline: Pipeline = Depends(get_pipeline)) -> DetectionService:
    """Return the app service, creating a storage-free one if none is set.

    The fallback keeps direct ``get_pipeline`` overrides (and tests) working
    without a running PostgreSQL instance.
    """
    global _service
    if _service is None or _service.pipeline is not pipeline:
        _service = DetectionService(pipeline)
    return _service
