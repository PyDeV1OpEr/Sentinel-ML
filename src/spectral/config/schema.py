from __future__ import annotations

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    name: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    normalize: bool = True
    method: str = "minmax"
    detrend: bool = False
    decompose_seasonality: bool = False
    seasonal_period: int | None = None
    fill_method: str = "interpolate"


class EnsembleConfig(BaseModel):
    method: str = "voting"
    min_detectors: int = 2


class AlertingConfig(BaseModel):
    enabled: bool = False
    webhook_url: str = ""
    cooldown_seconds: int = 300
    min_severity: str = "warning"


class StorageConfig(BaseModel):
    # Off by default so a fresh clone runs without PostgreSQL. Enable it and
    # set DATABASE_URL for persisted anomalies and the query endpoints.
    enabled: bool = False
    batch_size: int = 500


class PipelineConfig(BaseModel):
    """Top-level config for a detection pipeline."""

    name: str = "default"
    window_size: int = 100
    step_size: int = 1
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detectors: list[DetectorConfig] = Field(default_factory=list)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
