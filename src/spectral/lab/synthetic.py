"""Labelled synthetic time-series generation for detector experiments.

The upstream ``SyntheticSource`` remains untouched. This generator is richer:
it injects eight named anomaly types, records the exact event spans, and
returns a dataframe that is ready for training and one-to-one evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

ANOMALY_TYPES = (
    "spike",
    "drop",
    "level_shift",
    "seasonal",
    "noise_burst",
    "drift",
    "missing",
    "flatline",
)

_MAGNITUDE_TYPES = ("spike", "drop", "level_shift", "drift")


@dataclass(frozen=True)
class AnomalyEvent:
    """An injected anomaly and the half-open point range it covers."""

    anomaly_type: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {
            "type": self.anomaly_type,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "start_timestamp": None,
            "end_timestamp": None,
        }


@dataclass
class SyntheticConfig:
    """Configuration for :func:`generate_synthetic_timeseries`.

    ``trend`` is the total change over the series, ``seasonal_period`` is in
    points, and ``anomaly_ratio`` is the fraction of points that should carry
    an anomaly label. At least one event is created when the ratio is positive.
    """

    n_points: int = 2000
    start: str = "2026-01-01T00:00:00"
    freq: str = "1h"
    trend: float = 0.0
    seasonal_period: int = 24
    seasonal_amplitude: float = 1.0
    noise: float = 0.1
    anomaly_ratio: float = 0.03
    anomaly_types: tuple[str, ...] = ANOMALY_TYPES
    seed: int = 42

    def validate(self) -> None:
        if self.n_points < 20:
            raise ValueError("n_points must be at least 20")
        if self.seasonal_period < 2:
            raise ValueError("seasonal_period must be at least 2")
        if self.noise < 0:
            raise ValueError("noise must be non-negative")
        if self.seasonal_amplitude < 0:
            raise ValueError("seasonal_amplitude must be non-negative")
        if not 0 <= self.anomaly_ratio <= 0.5:
            raise ValueError("anomaly_ratio must be between 0 and 0.5")
        unknown = set(self.anomaly_types) - set(ANOMALY_TYPES)
        if unknown:
            raise ValueError(f"unknown anomaly types: {sorted(unknown)}")
        if self.anomaly_ratio > 0 and not self.anomaly_types:
            raise ValueError("anomaly_types cannot be empty when anomaly_ratio > 0")

    def with_overrides(self, **overrides) -> SyntheticConfig:
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown config fields: {sorted(unknown)}")
        updated = replace(self, **overrides)
        updated.validate()
        return updated


@dataclass
class SyntheticDataset:
    """A generated series plus its exact ground truth."""

    frame: pd.DataFrame
    events: list[AnomalyEvent]
    config: SyntheticConfig

    @property
    def labels(self) -> np.ndarray:
        return self.frame["is_anomaly"].to_numpy(dtype=int)

    @property
    def values(self) -> np.ndarray:
        return self.frame["value"].to_numpy(dtype=float)

    def summary(self) -> dict[str, int]:
        counts = self.frame.loc[self.frame["is_anomaly"] == 1, "anomaly_type"]
        summary = {name: int((counts == name).sum()) for name in ANOMALY_TYPES}
        summary["events"] = len(self.events)
        summary["anomaly_points"] = int(self.labels.sum())
        return summary

    def events_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=["type", "start", "end", "length", "start_timestamp", "end_timestamp"]
            )
        stamps = self.frame["timestamp"]
        rows = []
        for event in self.events:
            row = event.as_dict()
            row["start_timestamp"] = stamps.iloc[event.start]
            row["end_timestamp"] = stamps.iloc[event.end - 1]
            rows.append(row)
        return pd.DataFrame(rows)


def _event_length(
    anomaly_type: str, n_points: int, seasonal_period: int, rng: np.random.Generator
) -> int:
    if anomaly_type in ("spike", "drop"):
        return 1
    if anomaly_type == "level_shift":
        low, high = max(5, n_points // 40), max(8, n_points // 15)
    elif anomaly_type == "seasonal":
        low = high = max(seasonal_period, 8)
    elif anomaly_type == "noise_burst":
        low, high = 10, max(12, n_points // 50)
    elif anomaly_type == "drift":
        low, high = max(20, n_points // 40), max(30, n_points // 15)
    elif anomaly_type == "missing":
        low, high = 5, max(6, n_points // 100)
    elif anomaly_type == "flatline":
        low, high = 10, max(12, n_points // 40)
    else:  # pragma: no cover - validate() rejects unknown types
        raise ValueError(f"unknown anomaly type: {anomaly_type}")

    length = int(rng.integers(low, high + 1))
    return min(length, max(1, n_points // 3))


def generate_synthetic_timeseries(
    config: SyntheticConfig | None = None, **overrides
) -> SyntheticDataset:
    """Generate a labelled series with the requested anomaly types.

    Event spans never overlap. The ``anomaly_type`` column identifies each
    injected type; every injected point has ``is_anomaly == 1``. Missing-value
    events are represented as ``NaN`` in ``value`` and still carry a label.
    """
    cfg = config or SyntheticConfig()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_points
    t = np.arange(n, dtype=np.float64)

    trend = cfg.trend * (t / max(n - 1, 1))
    seasonal = cfg.seasonal_amplitude * np.sin(
        2 * np.pi * t / cfg.seasonal_period
    )
    signal = trend + seasonal
    if cfg.noise > 0:
        signal = signal + rng.standard_normal(n) * cfg.noise

    values = signal.astype(np.float64)
    scale = float(np.std(signal))
    if scale <= 0:
        scale = 1.0

    mask = np.zeros(n, dtype=bool)
    types = np.array(cfg.anomaly_types, dtype=object)
    rng.shuffle(types)
    target = int(round(n * cfg.anomaly_ratio)) if cfg.anomaly_ratio > 0 else 0
    if cfg.anomaly_ratio > 0:
        target = max(1, target)

    events: list[AnomalyEvent] = []
    covered = 0
    attempts = 0
    type_cursor = 0
    max_attempts = 500

    while covered < target and attempts < max_attempts:
        anomaly_type = str(types[type_cursor % len(types)])
        type_cursor += 1
        length = min(_event_length(anomaly_type, n, cfg.seasonal_period, rng), n)
        start = int(rng.integers(0, n - length + 1))
        end = start + length
        attempts += 1
        if mask[start:end].any():
            continue

        if anomaly_type == "spike":
            values[start:end] += rng.uniform(3.5, 6.0) * scale
        elif anomaly_type == "drop":
            values[start:end] -= rng.uniform(3.5, 6.0) * scale
        elif anomaly_type == "level_shift":
            direction = 1 if rng.random() > 0.5 else -1
            values[start:end] += direction * rng.uniform(3.0, 5.0) * scale
        elif anomaly_type == "seasonal":
            amplitude = max(cfg.seasonal_amplitude, 1.0)
            phase = t[start:end] + cfg.seasonal_period / 2
            values[start:end] += 1.8 * amplitude * np.sin(
                2 * np.pi * phase / cfg.seasonal_period
            )
        elif anomaly_type == "noise_burst":
            burst = max(cfg.noise, 0.05) * 6.0
            values[start:end] += rng.standard_normal(length) * burst
        elif anomaly_type == "drift":
            values[start:end] += np.linspace(0.0, 4.0 * scale, length)
        elif anomaly_type == "missing":
            values[start:end] = np.nan
        elif anomaly_type == "flatline":
            values[start:end] = values[start]

        mask[start:end] = True
        covered += length
        events.append(AnomalyEvent(anomaly_type=anomaly_type, start=start, end=end))

    if cfg.anomaly_ratio > 0 and not events:
        # Extremely small series with a large requested event: force a spike.
        start = int(rng.integers(0, n))
        values[start] += 4.0 * scale
        mask[start] = True
        events.append(AnomalyEvent(anomaly_type="spike", start=start, end=start + 1))

    anomaly_type_column = np.full(n, "none", dtype=object)
    for event in events:
        anomaly_type_column[event.start : event.end] = event.anomaly_type

    timestamps = pd.date_range(
        start=pd.Timestamp(cfg.start), periods=n, freq=cfg.freq
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "is_anomaly": mask.astype(int),
            "anomaly_type": anomaly_type_column,
        }
    )
    return SyntheticDataset(frame=frame, events=events, config=cfg)


def parse_anomaly_types(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse ``spike,drop,...`` from a CLI argument or an iterable."""
    if raw is None:
        return ANOMALY_TYPES
    if isinstance(raw, str):
        names = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in raw if str(part).strip())
    if not names:
        raise ValueError("at least one anomaly type is required")
    unknown = set(names) - set(ANOMALY_TYPES)
    if unknown:
        raise ValueError(
            f"unknown anomaly types: {sorted(unknown)}; choose from {list(ANOMALY_TYPES)}"
        )
    return names
