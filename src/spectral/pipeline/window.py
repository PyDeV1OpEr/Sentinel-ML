from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class SlidingWindow:
    """Generates sliding windows over a 1D array.

    Used in batch mode to split a time series into overlapping
    windows for detector input.
    """

    def __init__(self, size: int, step: int = 1):
        if size < 1:
            raise ValueError(f"Window size must be >= 1, got {size}")
        if step < 1:
            raise ValueError(f"Step size must be >= 1, got {step}")
        self.size = size
        self.step = step

    def transform(self, data: np.ndarray) -> list[np.ndarray]:
        """Split data into sliding windows."""
        data = np.asarray(data)
        windows = []
        for start, end in self.indices(len(data)):
            windows.append(data[start:end])
        return windows

    def indices(self, data_length: int) -> list[tuple[int, int]]:
        """Return (start, end) index pairs for each window."""
        pairs = []
        start = 0
        while start + self.size <= data_length:
            pairs.append((start, start + self.size))
            start += self.step
        return pairs

    def window_count(self, data_length: int) -> int:
        if data_length < self.size:
            return 0
        return (data_length - self.size) // self.step + 1


class StreamBuffer:
    """Accumulates streaming data points and yields windows.

    Designed for real-time usage: push one value at a time,
    and get back a window when enough data has accumulated.
    """

    def __init__(self, size: int, step: int = 1):
        self.size = size
        self.step = step
        self._buffer = deque(maxlen=size)
        self._count_since_last = 0

    def push(self, value: float) -> np.ndarray | None:
        """Add a value to the buffer.

        Returns a numpy array (window) when enough data has
        accumulated, otherwise None.
        """
        self._buffer.append(value)
        self._count_since_last += 1

        if len(self._buffer) == self.size and self._count_since_last >= self.step:
            self._count_since_last = 0
            return np.array(self._buffer)
        return None

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._count_since_last = 0

    @property
    def current_size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.size
