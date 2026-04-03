from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DataPointIn(BaseModel):
    timestamp: datetime
    value: float
    series_id: str = "default"
    metadata: dict = Field(default_factory=dict)


class DataPointBatch(BaseModel):
    points: list[DataPointIn]


class AnomalyOut(BaseModel):
    timestamp: datetime | None
    value: float
    score: float
    detector: str
    severity: str
    series_id: str = "default"


class AnomalyQuery(BaseModel):
    series_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    min_severity: str | None = None
    detector: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class DetectionRequest(BaseModel):
    values: list[float]
    timestamps: list[datetime] | None = None
    config: dict | None = None


class DetectionResponse(BaseModel):
    anomalies: list[AnomalyOut]
    total_points: int
    anomaly_count: int
    anomaly_ratio: float
    stats: dict = Field(default_factory=dict)


class PipelineStatus(BaseModel):
    name: str
    state: str
    uptime_seconds: float
    points_processed: int
    anomalies_detected: int
    detectors: list[str]
