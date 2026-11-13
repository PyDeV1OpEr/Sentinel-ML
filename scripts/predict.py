#!/usr/bin/env python
"""Run a saved LSTM artefact on new data and export per-point predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectral.lab.artifacts import load_lstm_artifact
from spectral.lab.data import load_timeseries_csv, predictions_frame
from spectral.lab.evaluation import predict_with_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", required=True, help="Artefact directory")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--label-col", default="is_anomaly")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--output", "-o", default="data/predictions.csv")
    args = parser.parse_args()

    loaded = load_lstm_artifact(args.model)
    frame, has_labels = load_timeseries_csv(
        args.input,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        label_col=args.label_col,
    )
    prediction = predict_with_detector(
        loaded.detector, frame, tolerance=args.tolerance
    )
    result = prediction.result

    print(f"model: {loaded.model_version}; points: {len(frame)}; "
          f"threshold: {loaded.detector._threshold:.6f}")
    print(f"flagged: {result.anomaly_count} ({result.anomaly_ratio:.2%})")
    print(f"score range: [{result.scores.min():.3f}, {result.scores.max():.3f}]")
    print(f"reconstruction error: median={float(np.median(prediction.errors)):.6f}, "
          f"max={prediction.errors.max():.6f}")

    if has_labels:
        metrics = prediction.metrics
        print(
            f"metrics: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} TP={metrics['tp']} FP={metrics['fp']} "
            f"FN={metrics['fn']} TN={metrics['tn']}"
        )
    else:
        print("no ground-truth labels: precision/recall/F1 are intentionally not reported")

    table = predictions_frame(
        frame, result.labels, result.scores, prediction.errors
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
