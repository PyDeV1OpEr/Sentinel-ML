from __future__ import annotations

import logging
from pathlib import Path

import yaml

from spectral.config.schema import DetectorConfig, PipelineConfig

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config() -> PipelineConfig:
    """Return a sensible default pipeline configuration."""
    return PipelineConfig(
        name="default",
        window_size=100,
        step_size=1,
        detectors=[
            DetectorConfig(name="zscore", params={"threshold": 3.0}),
            DetectorConfig(name="isolation_forest", params={"contamination": 0.05}),
            DetectorConfig(name="spectral_residual", params={"window_size": 24}),
        ],
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Values are merged with defaults, so you only need to specify
    what you want to override.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        logger.warning(f"Empty config file: {path}, using defaults")
        return get_default_config()

    # flatten nested structure if present
    config_dict = {}
    if "pipeline" in raw:
        config_dict.update(raw["pipeline"])
    if "preprocessing" in raw:
        config_dict["preprocessing"] = raw["preprocessing"]
    if "detectors" in raw:
        config_dict["detectors"] = raw["detectors"]
    if "ensemble" in raw:
        config_dict["ensemble"] = raw["ensemble"]
    if "alerting" in raw:
        config_dict["alerting"] = raw["alerting"]
    if "storage" in raw:
        config_dict["storage"] = raw["storage"]

    # merge with defaults
    default_dict = get_default_config().model_dump()
    merged = _deep_merge(default_dict, config_dict)

    config = PipelineConfig(**merged)
    logger.info(f"Loaded config from {path}: {config.name}")
    return config
