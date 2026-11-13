"""Local-first experiment history.

Each experiment is appended as one JSON object per line, so a partially
written file still yields every earlier record. The store is deliberately
tiny and dependency-free: a recruiter can inspect it with ``cat`` or Excel,
and the workbench can compare runs without a database.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HISTORY_PATH = Path(
    os.environ.get("SENTINEL_HISTORY_PATH", "reports/experiments.jsonl")
)

FIELDS = (
    "timestamp",
    "kind",
    "dataset_source",
    "dataset_fingerprint",
    "detector",
    "model_version",
    "params",
    "metrics",
    "duration_seconds",
    "notes",
)


def _clean(value):
    """Convert numpy scalars and remove NaN/inf so the JSON stays valid."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def record_experiment(
    *,
    kind: str,
    dataset_source: str | None = None,
    dataset_fingerprint: str | None = None,
    detector: str | None = None,
    model_version: str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
    duration_seconds: float | None = None,
    notes: str = "",
    path: str | Path | None = None,
) -> dict:
    """Append one experiment record and return it."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "dataset_source": dataset_source,
        "dataset_fingerprint": dataset_fingerprint,
        "detector": detector,
        "model_version": model_version,
        "params": _clean(params or {}),
        "metrics": _clean(metrics or {}),
        "duration_seconds": _clean(duration_seconds),
        "notes": notes,
    }
    target = Path(path) if path is not None else DEFAULT_HISTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return record


def load_history(path: str | Path | None = None) -> pd.DataFrame:
    """Return the history as a dataframe, skipping malformed lines."""
    target = Path(path) if path is not None else DEFAULT_HISTORY_PATH
    if not target.is_file():
        return pd.DataFrame(columns=FIELDS)
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                rows.append(record)
    if not rows:
        return pd.DataFrame(columns=FIELDS)
    return pd.DataFrame(rows)


def clear_history(path: str | Path | None = None) -> int:
    """Delete the history file and return the number of records removed."""
    target = Path(path) if path is not None else DEFAULT_HISTORY_PATH
    if not target.is_file():
        return 0
    count = len(load_history(target))
    target.unlink()
    return count
