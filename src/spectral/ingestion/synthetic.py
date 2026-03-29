from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from spectral.ingestion.base import DataPoint

logger = logging.getLogger(__name__)


class SyntheticSource:
    """Generate synthetic time-series data with configurable anomalies.

    Useful for testing, demos, and evaluating detector performance
    against known ground truth.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_points: int = 1000,
        anomaly_ratio: float = 0.03,
        noise_level: float = 0.1,
        patterns: list[str] | None = None,
        start_time: datetime | None = None,
        interval_seconds: int = 60,
    ) -> tuple[list[DataPoint], np.ndarray]:
        """Generate synthetic time series with injected anomalies.

        Args:
            n_points: number of data points to generate
            anomaly_ratio: fraction of points that should be anomalous
            noise_level: standard deviation of Gaussian noise
            patterns: signal components to combine (sine, trend, seasonal, random_walk)
            start_time: starting timestamp
            interval_seconds: seconds between data points

        Returns:
            (data_points, labels) where labels[i]=1 indicates an anomaly
        """
        if patterns is None:
            patterns = ["sine", "seasonal"]

        if start_time is None:
            start_time = datetime(2026, 1, 1)

        t = np.arange(n_points, dtype=np.float64)
        signal = np.zeros(n_points)

        for pattern in patterns:
            if pattern == "sine":
                signal += np.sin(2 * np.pi * t / 100)
            elif pattern == "trend":
                signal += 0.002 * t
            elif pattern == "seasonal":
                # daily seasonality (assuming minutely data, 1440 points per day)
                period = 1440 // interval_seconds if interval_seconds < 1440 else 24
                signal += 0.5 * np.sin(2 * np.pi * t / period)
            elif pattern == "random_walk":
                steps = self.rng.standard_normal(n_points) * 0.05
                signal += np.cumsum(steps)

        # add noise
        signal += self.rng.standard_normal(n_points) * noise_level

        # inject anomalies
        n_anomalies = max(1, int(n_points * anomaly_ratio))
        labels = np.zeros(n_points, dtype=int)

        # distribute anomaly types
        n_point = n_anomalies // 2
        n_contextual = n_anomalies // 4
        n_collective = n_anomalies - n_point - n_contextual

        # point anomalies: sudden spikes
        point_indices = self.rng.choice(n_points, size=n_point, replace=False)
        for idx in point_indices:
            spike_magnitude = self.rng.uniform(3, 6) * signal.std()
            direction = self.rng.choice([-1, 1])
            signal[idx] += direction * spike_magnitude
            labels[idx] = 1

        # contextual anomalies: values that are normal globally but wrong for context
        available = np.where(labels == 0)[0]
        if len(available) >= n_contextual:
            ctx_indices = self.rng.choice(available, size=n_contextual, replace=False)
            for idx in ctx_indices:
                # shift value to what it would be half a period away
                shift = int(50 + self.rng.integers(-10, 10))
                source = (idx + shift) % n_points
                signal[idx] = signal[source] + self.rng.standard_normal() * noise_level * 0.5
                labels[idx] = 1

        # collective anomalies: short subsequences with different behavior
        for _ in range(max(1, n_collective // 5)):
            length = self.rng.integers(3, 8)
            start = self.rng.integers(0, n_points - length)
            # replace with flat or reversed pattern
            if self.rng.random() > 0.5:
                signal[start : start + length] = signal[start]  # flatline
            else:
                signal[start : start + length] = signal[start : start + length][::-1]
            labels[start : start + length] = 1

        # build data points
        data_points = []
        for i in range(n_points):
            ts = start_time + timedelta(seconds=i * interval_seconds)
            data_points.append(
                DataPoint(
                    timestamp=ts,
                    value=float(signal[i]),
                    series_id="synthetic",
                    metadata={"is_anomaly": bool(labels[i])},
                )
            )

        anomaly_count = labels.sum()
        logger.info(
            f"Generated {n_points} points with {anomaly_count} anomalies "
            f"({anomaly_count/n_points:.1%}), patterns={patterns}"
        )

        return data_points, labels
