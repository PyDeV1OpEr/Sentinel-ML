from __future__ import annotations

from functools import lru_cache

from spectral.config import get_default_config, load_config
from spectral.pipeline.runner import Pipeline
from spectral.storage.repository import AnomalyRepository

_pipeline: Pipeline | None = None
_repository: AnomalyRepository | None = None


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
