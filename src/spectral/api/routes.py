from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from spectral.api.dependencies import get_pipeline, get_repository, get_service
from spectral.api.schemas import (
    AnomalyOut,
    DataPointBatch,
    DataPointIn,
    DetectionRequest,
    DetectionResponse,
    PipelineStatus,
)
from spectral.api.service import DetectionService
from spectral.pipeline.runner import Pipeline
from spectral.storage.repository import AnomalyRepository

logger = logging.getLogger(__name__)

router = APIRouter()


def _anomaly_out(result, series_id: str = "default") -> AnomalyOut:
    return AnomalyOut(
        timestamp=result.timestamp,
        value=result.value,
        score=result.score,
        detector=result.detector,
        severity=result.severity,
        series_id=series_id,
    )


def _require_storage(pipeline: Pipeline) -> None:
    if not pipeline.config.storage.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Storage is disabled. Enable storage in the pipeline config and "
                "set DATABASE_URL to query persisted anomalies."
            ),
        )


@router.post("/ingest", response_model=dict)
async def ingest_data(
    data: DataPointIn | DataPointBatch,
    service: DetectionService = Depends(get_service),
):
    """Ingest data points, run streaming detection and notify consumers."""
    points = [data] if isinstance(data, DataPointIn) else data.points
    anomalies = await service.ingest(points)
    return {
        "ingested": len(points),
        "anomalies": [
            _anomaly_out(
                anomaly, str(anomaly.metadata.get("series_id", "default"))
            ).model_dump()
            for anomaly in anomalies
        ],
        "anomaly_count": len(anomalies),
    }


@router.post("/detect", response_model=DetectionResponse)
async def detect_batch(
    request: DetectionRequest,
    service: DetectionService = Depends(get_service),
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Run batch anomaly detection on provided values."""
    anomalies = await service.detect_batch(
        request.values,
        request.timestamps,
        series_id=request.series_id,
    )

    return DetectionResponse(
        anomalies=[
            _anomaly_out(anomaly, request.series_id) for anomaly in anomalies
        ],
        total_points=len(request.values),
        anomaly_count=len(anomalies),
        anomaly_ratio=(
            len(anomalies) / len(request.values) if len(request.values) > 0 else 0
        ),
        stats=pipeline.stats,
    )


@router.get("/anomalies", response_model=list[AnomalyOut])
async def query_anomalies(
    series_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    min_severity: str | None = None,
    detector: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    pipeline: Pipeline = Depends(get_pipeline),
    repo: AnomalyRepository = Depends(get_repository),
):
    """Query stored anomalies with filters."""
    _require_storage(pipeline)
    records = await repo.query(
        series_id=series_id,
        start=start,
        end=end,
        min_severity=min_severity,
        detector=detector,
        limit=limit,
        offset=offset,
    )

    return [
        AnomalyOut(
            timestamp=r.timestamp,
            value=r.value,
            score=r.score,
            detector=r.detector,
            severity=r.severity,
            series_id=r.series_id,
        )
        for r in records
    ]


@router.get("/status", response_model=PipelineStatus)
async def get_status(
    request: Request,
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Get current pipeline status."""
    stats = pipeline.stats
    started = getattr(request.app.state, "started_at", None)
    uptime = max(0.0, time.monotonic() - started) if started is not None else 0.0
    return PipelineStatus(
        name=pipeline.config.name,
        state="running",
        uptime_seconds=uptime,
        points_processed=stats["points_processed"],
        anomalies_detected=stats["anomalies_detected"],
        detectors=stats["detectors"],
    )


@router.get("/metrics", response_model=dict)
async def get_metrics(
    series_id: str | None = None,
    window_hours: int = Query(default=24, ge=1),
    pipeline: Pipeline = Depends(get_pipeline),
    repo: AnomalyRepository = Depends(get_repository),
):
    """Get dashboard-ready metrics and statistics.

    When storage is disabled the response still contains live pipeline stats
    and an explicit ``storage_enabled: false`` flag, rather than pretending
    that an empty database was queried.
    """
    stats = pipeline.stats
    if not pipeline.config.storage.enabled:
        return {
            "pipeline": stats,
            "anomalies": {
                "total_anomalies": 0,
                "avg_score": 0.0,
                "window_hours": window_hours,
                "per_detector": {},
            },
            "window_hours": window_hours,
            "storage_enabled": False,
        }

    db_stats = await repo.get_stats(series_id=series_id, window_hours=window_hours)
    return {
        "pipeline": stats,
        "anomalies": db_stats,
        "window_hours": window_hours,
        "storage_enabled": True,
    }
