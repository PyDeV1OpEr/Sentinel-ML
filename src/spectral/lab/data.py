"""CSV loading, validation, imputation and chronological splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "value")


def prepare_frame(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    label_col: str | None = "is_anomaly",
) -> tuple[pd.DataFrame, bool]:
    """Normalise a user dataframe into the lab's canonical schema.

    Returns ``(frame, has_labels)``. The returned frame has ``timestamp``,
    ``value`` (raw, possibly NaN), ``is_missing``, ``is_anomaly`` (nullable
    ``Int64``) and, when present, ``anomaly_type``. Values are not imputed
    here: callers split first and then call :func:`impute_values`, so
    interpolation cannot leak across a train/test boundary.
    """
    if timestamp_col not in frame.columns:
        raise ValueError(f"timestamp column {timestamp_col!r} not found")
    if value_col not in frame.columns:
        raise ValueError(f"value column {value_col!r} not found")
    if len(frame) < 10:
        raise ValueError("need at least 10 rows")

    timestamps = pd.to_datetime(frame[timestamp_col], errors="raise")
    if timestamps.isna().any():
        raise ValueError("timestamp column contains unparseable or missing values")

    try:
        values = pd.to_numeric(frame[value_col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value column {value_col!r} is not numeric") from exc
    if values.notna().sum() == 0:
        raise ValueError("value column contains no numeric values")

    prepared = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "value": values.to_numpy(),
            "is_missing": values.isna().to_numpy(),
        }
    )

    if label_col and label_col in frame.columns:
        raw_labels = pd.to_numeric(frame[label_col], errors="coerce")
        if raw_labels.isna().any():
            raise ValueError(f"label column {label_col!r} contains missing values")
        labels = raw_labels.astype(int)
        if not labels.isin([0, 1]).all():
            raise ValueError(f"label column {label_col!r} must contain 0/1")
        prepared["is_anomaly"] = pd.array(labels.to_numpy(), dtype="Int64")
        has_labels = True
    else:
        prepared["is_anomaly"] = pd.array([pd.NA] * len(prepared), dtype="Int64")
        has_labels = False

    if "anomaly_type" in frame.columns:
        prepared["anomaly_type"] = frame["anomaly_type"].to_numpy()

    prepared = (
        prepared.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    if len(prepared) < 10:
        raise ValueError("need at least 10 unique timestamps")
    return prepared, has_labels


def load_timeseries_csv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    label_col: str | None = "is_anomaly",
) -> tuple[pd.DataFrame, bool]:
    """Read a CSV and normalise it with :func:`prepare_frame`."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"input file not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    return prepare_frame(
        raw,
        timestamp_col=timestamp_col,
        value_col=value_col,
        label_col=label_col,
    )


def impute_values(values: np.ndarray | pd.Series) -> np.ndarray:
    """Linearly interpolate missing values, then forward/back fill edges.

    This is an offline helper. It is applied after the chronological split so
    that future points never inform the training half. A production streaming
    path would carry its own fill state instead.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    if series.notna().sum() == 0:
        raise ValueError("cannot impute a series with no observed values")
    if series.isna().any():
        series = series.interpolate(method="linear", limit_direction="both")
        series = series.ffill().bfill()
    return series.to_numpy(dtype=float)


def labels_from_frame(frame: pd.DataFrame) -> np.ndarray | None:
    """Return integer labels, or ``None`` when the frame is unlabelled."""
    if "is_anomaly" not in frame.columns:
        return None
    labels = frame["is_anomaly"].to_numpy(dtype="float")
    if np.isnan(labels).any():
        return None
    return labels.astype(int)


def chronological_split(
    frame: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, never randomly, and return reset-index halves."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    cut = int(round(len(frame) * train_fraction))
    if cut < 1 or cut >= len(frame):
        raise ValueError("train_fraction leaves an empty split")
    train = frame.iloc[:cut].reset_index(drop=True)
    test = frame.iloc[cut:].reset_index(drop=True)
    return train, test


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Stable SHA-256 of timestamps and raw values.

    NaN is replaced by a sentinel so different NaN payloads hash identically.
    Used to check that an artefact was trained on the dataset being evaluated.
    """
    timestamps = (
        pd.to_datetime(frame["timestamp"]).astype("int64").to_numpy(dtype=np.int64)
    )
    values = np.asarray(frame["value"], dtype=np.float64)
    values = np.where(np.isnan(values), np.finfo(np.float64).min, values)
    digest = hashlib.sha256()
    digest.update(np.int64(len(frame)).tobytes())
    digest.update(np.ascontiguousarray(timestamps).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def predictions_frame(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    scores: np.ndarray,
    errors: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the per-point export table used by the workbench and scripts."""
    output = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "value": frame["value"].to_numpy(),
            "is_missing": frame["is_missing"].to_numpy()
            if "is_missing" in frame.columns
            else False,
            "is_anomaly": frame["is_anomaly"].to_numpy()
            if "is_anomaly" in frame.columns
            else pd.NA,
            "predicted": np.asarray(predicted, dtype=int),
            "score": np.asarray(scores, dtype=float),
        }
    )
    if errors is not None:
        output["reconstruction_error"] = np.asarray(errors, dtype=float)
    if "anomaly_type" in frame.columns:
        output["anomaly_type"] = frame["anomaly_type"].to_numpy()
    return output
