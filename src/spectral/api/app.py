from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spectral import __version__
from spectral.api.dependencies import set_pipeline
from spectral.api.routes import router
from spectral.api.websocket import websocket_endpoint
from spectral.config import get_default_config, load_config
from spectral.pipeline.runner import Pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown events."""
    # load config
    config_path = os.environ.get("SPECTRAL_CONFIG")
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        logger.info("Using default config")

    # init pipeline
    pipeline = Pipeline(config)
    set_pipeline(pipeline)

    # init database if storage is enabled
    if config.storage.enabled:
        try:
            from spectral.storage.database import init_db
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database init failed (storage disabled): {e}")

    logger.info(f"Spectral v{__version__} started with pipeline '{config.name}'")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Spectral",
        description="Real-time anomaly detection engine for time-series data",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.websocket("/ws/stream")(websocket_endpoint)

    return app


app = create_app()
