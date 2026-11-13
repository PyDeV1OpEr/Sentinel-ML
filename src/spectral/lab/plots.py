"""Matplotlib figures for the experimental workbench.

All functions return a ``Figure`` and never call ``show``. Streamlit renders
them with ``st.pyplot``; scripts save them with ``fig.savefig``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _timestamps(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame["timestamp"]).to_numpy()


def plot_series(
    frame: pd.DataFrame,
    *,
    predicted: np.ndarray | None = None,
    title: str = "Time series",
) -> plt.Figure:
    """Raw series with ground-truth and predicted anomalies marked."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    stamps = _timestamps(frame)
    values = frame["value"].to_numpy(dtype=float)
    ax.plot(stamps, values, linewidth=0.8, color="#4c72b0", label="value")

    true_mask = (
        frame["is_anomaly"].fillna(0).to_numpy(dtype=int) == 1
        if "is_anomaly" in frame.columns
        else np.zeros(len(frame), dtype=int)
    )
    if true_mask.any():
        ax.scatter(
            stamps[true_mask],
            values[true_mask],
            s=26,
            c="#d62728",
            marker="o",
            label="actual anomaly",
            zorder=4,
        )
    if predicted is not None:
        predicted = np.asarray(predicted, dtype=int)
        if predicted.any():
            ax.scatter(
                stamps[predicted == 1],
                values[predicted == 1],
                s=34,
                c="#ff7f0e",
                marker="x",
                label="predicted anomaly",
                zorder=5,
            )
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_reconstruction_errors(
    frame: pd.DataFrame,
    errors: np.ndarray,
    threshold: float,
    *,
    predicted: np.ndarray | None = None,
    title: str = "Reconstruction error",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    stamps = _timestamps(frame)
    ax.plot(stamps, errors, linewidth=0.8, color="#2ca02c", label="point error")
    ax.axhline(threshold, color="#d62728", linestyle="--", label="threshold")

    true_mask = (
        frame["is_anomaly"].fillna(0).to_numpy(dtype=int) == 1
        if "is_anomaly" in frame.columns
        else np.zeros(len(frame), dtype=int)
    )
    if true_mask.any():
        ax.scatter(
            stamps[true_mask],
            errors[true_mask],
            s=24,
            c="#d62728",
            marker="o",
            label="actual anomaly",
            zorder=4,
        )
    if predicted is not None:
        predicted = np.asarray(predicted, dtype=int)
        if predicted.any():
            ax.scatter(
                stamps[predicted == 1],
                errors[predicted == 1],
                s=30,
                c="#ff7f0e",
                marker="x",
                label="predicted anomaly",
                zorder=5,
            )
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("MSE (normalised)")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_score_distribution(
    errors: np.ndarray,
    threshold: float | None = None,
    *,
    labels: np.ndarray | None = None,
    title: str = "Score distribution",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    errors = np.asarray(errors, dtype=float)
    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        bins = min(60, max(10, len(errors) // 20))
        ax.hist(errors[labels == 0], bins=bins, alpha=0.65, label="normal")
        if (labels == 1).any():
            ax.hist(errors[labels == 1], bins=bins, alpha=0.65, label="anomaly")
    else:
        ax.hist(errors, bins=min(60, max(10, len(errors) // 20)), alpha=0.75)
    if threshold is not None:
        ax.axvline(threshold, color="#d62728", linestyle="--", label="threshold")
    ax.set_title(title)
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("points")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_training_history(history: list[dict], title: str = "Training loss") -> plt.Figure:
    """Train and validation loss per epoch from ``detector._training_history``."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    if history:
        epochs = [entry["epoch"] for entry in history]
        ax.plot(epochs, [entry["train_loss"] for entry in history], label="train")
        ax.plot(
            epochs,
            [entry["validation_loss"] for entry in history],
            label="validation / calibration holdout",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (normalised)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no training history", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_type_recall(
    rows: pd.DataFrame, title: str = "Recall by anomaly type"
) -> plt.Figure:
    """Heatmap of detector recall for each anomaly type."""
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows.columns) + 2), 0.55 * len(rows) + 1.6))
    data = rows.fillna(0.0).to_numpy(dtype=float)
    image = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rows.columns)), labels=list(rows.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), labels=list(rows.index))
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(
                col,
                row,
                f"{data[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def plot_confusion(matrix: np.ndarray, title: str = "Confusion matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    matrix = np.asarray(matrix, dtype=int)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], labels=["predicted normal", "predicted anomaly"])
    ax.set_yticks([0, 1], labels=["actual normal", "actual anomaly"])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: float,
    title: str = "Precision-recall curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    if len(precision):
        ax.plot(recall, precision, linewidth=1.5, color="#1f77b4")
        ax.fill_between(recall, precision, alpha=0.15, color="#1f77b4")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{title} (AP={average_precision:.3f})")
    fig.tight_layout()
    return fig


def plot_threshold_sweep(
    rows: list[dict], selected_threshold: float | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if rows:
        thresholds = [row["threshold"] for row in rows]
        ax.plot(thresholds, [row["precision"] for row in rows], label="precision")
        ax.plot(thresholds, [row["recall"] for row in rows], label="recall")
        ax.plot(thresholds, [row["f1"] for row in rows], label="F1")
        if selected_threshold is not None:
            ax.axvline(
                selected_threshold,
                color="#d62728",
                linestyle="--",
                label="calibrated threshold",
            )
    ax.set_xlabel("score threshold (higher = more anomalous)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Threshold sensitivity (one-to-one events)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_detector_comparison(
    rows: pd.DataFrame, metric: str = "test_f1"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.45 * len(rows) + 1.2)))
    data = rows[["detector", metric]].dropna().sort_values(metric)
    ax.barh(data["detector"], data[metric], color="#4c72b0")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title("Detector comparison on held-out data")
    fig.tight_layout()
    return fig
