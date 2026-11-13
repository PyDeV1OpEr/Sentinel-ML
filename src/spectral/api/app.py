from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spectral import __version__
from spectral.alerting.dispatcher import AlertDispatcher
from spectral.api.dependencies import set_pipeline, set_service
from spectral.api.routes import router
from spectral.api.service import DetectionService
from spectral.api.websocket import manager, websocket_endpoint
from spectral.config import get_default_config, load_config
from spectral.pipeline.runner import Pipeline
from spectral.storage.repository import AnomalyRepository

logger = logging.getLogger(__name__)


def _cors_origins() -> list[str]:
    """Explicit browser origins, never ``*`` with credentials enabled."""
    raw = os.environ.get("SENTINEL_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the pipeline and optional storage/alerting integrations."""
    app.state.started_at = time.monotonic()

    config_path = os.environ.get("SPECTRAL_CONFIG")
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        config = get_default_config()
        logger.info("Using default config")

    pipeline = Pipeline(config)
    set_pipeline(pipeline)

    repository = AnomalyRepository()
    alerts = AlertDispatcher(config.alerting)
    service = DetectionService(
        pipeline,
        repository=repository,
        alerts=alerts,
        broadcaster=manager,
        storage_enabled=config.storage.enabled,
    )
    set_service(service)

    if config.storage.enabled:
        try:
            from spectral.storage.database import init_db

            await init_db()
            logger.info("Database initialized")
        except Exception as exc:
            # Storage is optional for a local demo. The service degrades to
            # detection-only instead of failing startup.
            logger.warning("Database init failed; continuing without storage: %s", exc)

    logger.info("Spectral v%s started with pipeline '%s'", __version__, config.name)
    try:
        yield
    finally:
        await service.aclose()
        set_service(None)
        logger.info("Shutting down")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Spectral",
        description="Real-time anomaly detection engine for time-series data",
        version=__version__,
        lifespan=lifespan,
    )

    origins = _cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    app.include_router(router)
    app.websocket("/ws/stream")(websocket_endpoint)

    return app


app = create_app()
