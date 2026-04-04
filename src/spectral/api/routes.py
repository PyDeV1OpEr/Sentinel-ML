from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, Query

from spectral.api.dependencies import get_pipeline, get_repository
from spectral.api.schemas import (
    AnomalyOut,
    DataPointBatch,
    DataPointIn,
    DetectionRequest,
    DetectionResponse,
    PipelineStatus,
)
from spectral.pipeline.runner import Pipeline
from spectral.storage.repository import AnomalyRepository

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=dict)
async def ingest_data(
    data: DataPointIn | DataPointBatch,
    pipeline: Pipeline = Depends(get_pipeline),
    repo: AnomalyRepository = Depends(get_repository),
):
    """Ingest data points and run real-time detection."""
    if isinstance(data, DataPointIn):
        points = [data]
    else:
        points = data.points

    anomalies_found = []
    for point in points:
        result = pipeline.detect_streaming(
            value=point.value,
            timestamp=point.timestamp,
        )
        if result:
            anomalies_found.append(
                AnomalyOut(
                    timestamp=result.timestamp,
                    value=result.value,
                    score=result.score,
                    detector=result.detector,
                    severity=result.severity,
                    series_id=point.series_id,
                )
            )

    return {
        "ingested": len(points),
        "anomalies": [a.model_dump() for a in anomalies_found],
        "anomaly_count": len(anomalies_found),
    }


@router.post("/detect", response_model=DetectionResponse)
async def detect_batch(
    request: DetectionRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Run batch anomaly detection on provided values."""
    values = np.array(request.values)
    timestamps = request.timestamps

    anomalies = pipeline.detect(values, timestamps)

    anomaly_out = [
        AnomalyOut(
            timestamp=a.timestamp,
            value=a.value,
            score=a.score,
            detector=a.detector,
            severity=a.severity,
        )
        for a in anomalies
    ]

    return DetectionResponse(
        anomalies=anomaly_out,
        total_points=len(values),
        anomaly_count=len(anomalies),
        anomaly_ratio=len(anomalies) / len(values) if len(values) > 0 else 0,
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
    repo: AnomalyRepository = Depends(get_repository),
):
    """Query stored anomalies with filters."""
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
async def get_status(pipeline: Pipeline = Depends(get_pipeline)):
    """Get current pipeline status."""
    stats = pipeline.stats
    return PipelineStatus(
        name=pipeline.config.name,
        state="running",
        uptime_seconds=0,  # TODO: track actual uptime
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
    """Get dashboard-ready metrics and statistics."""
    stats = pipeline.stats
    db_stats = await repo.get_stats(series_id=series_id, window_hours=window_hours)

    return {
        "pipeline": stats,
        "anomalies": db_stats,
        "window_hours": window_hours,
    }
