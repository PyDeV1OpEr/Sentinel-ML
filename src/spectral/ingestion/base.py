from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass
class DataPoint:
    """A single time-series data point."""

    timestamp: datetime
    value: float
    series_id: str = "default"
    metadata: dict = field(default_factory=dict)


class BaseSource(ABC):
    """Abstract base class for data ingestion sources."""

    @abstractmethod
    def read(self) -> Iterator[DataPoint]:
        """Yield data points from the source."""
        ...
