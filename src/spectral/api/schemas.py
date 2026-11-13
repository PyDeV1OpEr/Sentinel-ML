from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class DataPointIn(BaseModel):
    timestamp: datetime
    value: float
    series_id: str = Field(default="default", min_length=1, max_length=128)
    metadata: dict = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class DataPointBatch(BaseModel):
    points: list[DataPointIn] = Field(min_length=1, max_length=10_000)


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
    values: list[float] = Field(min_length=1, max_length=100_000)
    timestamps: list[datetime] | None = None
    series_id: str = Field(default="default", min_length=1, max_length=128)
    config: dict | None = None

    @field_validator("values")
    @classmethod
    def _finite_values(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("values must all be finite")
        return values

    @model_validator(mode="after")
    def _matching_timestamps(self):
        if self.timestamps is not None and len(self.timestamps) != len(self.values):
            raise ValueError(
                "timestamps must have the same length as values"
            )
        return self


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
