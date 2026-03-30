from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from spectral.ingestion.base import BaseSource, DataPoint

logger = logging.getLogger(__name__)


class CSVSource(BaseSource):
    """Load time-series data from CSV, JSON, or Parquet files."""

    def __init__(
        self,
        path: str | Path,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
        series_id_col: str | None = None,
    ):
        self.path = Path(path)
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.series_id_col = series_id_col

    def _load_df(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.path)
        elif suffix == ".json":
            df = pd.read_json(self.path)
        elif suffix == ".parquet":
            df = pd.read_parquet(self.path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_col}' not found. "
                f"Available: {list(df.columns)}"
            )
        if self.value_col not in df.columns:
            raise ValueError(
                f"Value column '{self.value_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(self.timestamp_col)
        return df

    def read(self) -> Iterator[DataPoint]:
        df = self._load_df()
        logger.info(f"Loaded {len(df)} rows from {self.path}")

        for _, row in df.iterrows():
            series_id = "default"
            if self.series_id_col and self.series_id_col in df.columns:
                series_id = str(row[self.series_id_col])

            yield DataPoint(
                timestamp=row[self.timestamp_col].to_pydatetime(),
                value=float(row[self.value_col]),
                series_id=series_id,
            )
