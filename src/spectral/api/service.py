"""Application service connecting detection to its optional integrations.

The FastAPI routes own HTTP concerns; this module owns the side effects of a
detection request. Storage, alerting and WebSocket delivery are optional: a
local demo with no PostgreSQL and no webhook must still detect anomalies. Any
failure in an optional integration is logged and never replaces the detection
response, so a dashboard outage cannot take the detector down with it.

The service is deliberately not a second detection implementation: it calls
``Pipeline.detect`` and ``Pipeline.detect_streaming`` exactly as the CLI does.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

import numpy as np

from spectral.alerting.dispatcher import AlertDispatcher
from spectral.pipeline.runner import Anomaly, Pipeline

logger = logging.getLogger(__name__)


class RepositoryLike(Protocol):
    """The subset of ``AnomalyRepository`` this service needs."""

    async def save_anomaly(self, anomaly: Anomaly, series_id: str = "default") -> str:
        ...

    async def save_batch(self, anomalies: list[Anomaly], series_id: str = "default") -> int:
        ...

    async def save_data_point(
        self,
        timestamp: datetime,
        value: float,
        series_id: str = "default",
        metadata: dict | None = None,
    ) -> str:
        ...


class BroadcasterLike(Protocol):
    async def broadcast_anomaly(self, anomaly: Anomaly, series_id: str = "default") -> None:
        ...


class DetectionService:
    """Run detection and fan the results out to optional consumers.

    Args:
        pipeline: the fitted-or-fit-on-demand detection pipeline.
        repository: persistent anomaly store. Ignored when ``storage_enabled``
            is false.
        alerts: optional alert dispatcher. Calls into it are offloaded to a
            worker thread because the HTTP client is blocking.
        broadcaster: optional WebSocket fan-out.
        storage_enabled: gate for database writes. Local demos run with this
            false so PostgreSQL is not a prerequisite.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        repository: RepositoryLike | None = None,
        alerts: AlertDispatcher | None = None,
        broadcaster: BroadcasterLike | None = None,
        storage_enabled: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.repository = repository
        self.alerts = alerts
        self.broadcaster = broadcaster
        self.storage_enabled = bool(storage_enabled and repository is not None)
        self._persistence_warned = False

    async def _persist_anomaly(self, anomaly: Anomaly, series_id: str) -> None:
        if not self.storage_enabled:
            return
        try:
            await self.repository.save_anomaly(anomaly, series_id=series_id)
        except Exception as exc:  # optional integration must not break detection
            if not self._persistence_warned:
                logger.warning("Anomaly persistence disabled after error: %s", exc)
                self._persistence_warned = True

    async def _persist_points(self, points: Iterable) -> None:
        if not self.storage_enabled:
            return
        for point in points:
            try:
                await self.repository.save_data_point(
                    timestamp=point.timestamp,
                    value=point.value,
                    series_id=point.series_id,
                    metadata=point.metadata,
                )
            except Exception as exc:
                if not self._persistence_warned:
                    logger.warning("Data point persistence disabled after error: %s", exc)
                    self._persistence_warned = True
                return

    async def _notify(self, anomaly: Anomaly, series_id: str) -> None:
        """Deliver one anomaly to the alert channel and WebSocket clients."""
        if self.alerts is not None:
            try:
                await asyncio.to_thread(self.alerts.dispatch, anomaly, series_id)
            except Exception as exc:
                logger.warning("Alert dispatch failed for %s: %s", series_id, exc)
        if self.broadcaster is not None:
            try:
                await self.broadcaster.broadcast_anomaly(anomaly, series_id=series_id)
            except Exception as exc:
                logger.warning("WebSocket broadcast failed for %s: %s", series_id, exc)

    async def ingest(self, points: list) -> list[Anomaly]:
        """Stream points through the pipeline, returning anomalies found.

        Each point is persisted before detection so the raw series remains
        queryable. An anomaly is persisted, alerted and broadcast immediately,
        preserving the point's ``series_id``.
        """
        await self._persist_points(points)

        anomalies: list[Anomaly] = []
        for point in points:
            result = self.pipeline.detect_streaming(
                value=point.value,
                timestamp=point.timestamp,
            )
            if result is None:
                continue
            result.metadata["series_id"] = point.series_id
            anomalies.append(result)
            await self._persist_anomaly(result, point.series_id)
            await self._notify(result, point.series_id)
        return anomalies

    async def detect_batch(
        self,
        values: list[float] | np.ndarray,
        timestamps: list[datetime] | None = None,
        *,
        series_id: str = "default",
    ) -> list[Anomaly]:
        """Run batch detection and fan out the anomalies it found."""
        anomalies = self.pipeline.detect(np.asarray(values, dtype=float), timestamps)

        if self.storage_enabled and anomalies:
            try:
                await self.repository.save_batch(anomalies, series_id=series_id)
            except Exception as exc:
                if not self._persistence_warned:
                    logger.warning("Anomaly persistence disabled after error: %s", exc)
                    self._persistence_warned = True

        for anomaly in anomalies:
            await self._notify(anomaly, series_id)
        return anomalies

    async def aclose(self) -> None:
        """Release the alert dispatcher's HTTP client."""
        if self.alerts is not None:
            try:
                await asyncio.to_thread(self.alerts.close)
            except Exception as exc:  # pragma: no cover - defensive shutdown
                logger.debug("Alert dispatcher close failed: %s", exc)
