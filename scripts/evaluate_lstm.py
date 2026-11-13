#!/usr/bin/env python
"""Evaluate a saved LSTM artefact on labelled data, using its recorded split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.lab.artifacts import load_lstm_artifact
from spectral.lab.data import load_timeseries_csv, predictions_frame
from spectral.lab.evaluation import evaluate_loaded_model


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _print_metrics(name: str, metrics: dict | None) -> None:
    if metrics is None:
        print(f"{name:10s} no ground-truth labels; metrics unavailable")
        return
    print(
        f"{name:10s} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f}  TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TN={metrics['tn']}  FPR={metrics['false_positive_rate']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", required=True, help="Artefact directory")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--label-col", default="is_anomaly")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--output", "-o", default="reports/evaluation.json")
    parser.add_argument("--export-csv", default=None,
                        help="Optional per-point predictions CSV")
    args = parser.parse_args()

    loaded = load_lstm_artifact(args.model)
    frame, has_labels = load_timeseries_csv(
        args.input,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        label_col=args.label_col,
    )
    print(f"model: {loaded.model_version} ({loaded.path})")
    print(f"dataset: {len(frame)} points; labels: {'yes' if has_labels else 'no'}")
    print(f"threshold: {loaded.detector._threshold:.6f}; "
          f"calibration: {loaded.detector._calibration_source}")

    parts = evaluate_loaded_model(loaded, frame, tolerance=args.tolerance)
    split_matched = bool(parts.get("split_matched"))
    if has_labels:
        print(f"split matched artefact fingerprint: {split_matched}")
        for name in ("train", "test", "all"):
            if name in parts:
                _print_metrics(name, parts[name].metrics)
    else:
        print("no ground-truth labels: showing prediction counts and score ranges only")
        for name in ("train", "test", "all"):
            if name in parts:
                result = parts[name].result
                print(f"{name:10s} flagged={result.anomaly_count} "
                      f"score range=[{result.scores.min():.3f}, {result.scores.max():.3f}]")

    payload = {
        "model_version": loaded.model_version,
        "model_path": str(loaded.path),
        "input": str(args.input),
        "split_matched": split_matched,
        "has_labels": has_labels,
        "tolerance": args.tolerance,
        "threshold": loaded.detector._threshold,
        "parts": {},
    }
    for name in ("train", "test", "all"):
        if name not in parts:
            continue
        prediction = parts[name]
        payload["parts"][name] = {
            "points": int(len(prediction.frame)),
            "flagged": int(prediction.result.anomaly_count),
            "metrics": prediction.metrics,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    print(f"wrote {output}")

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for name in ("train", "test", "all"):
            if name not in parts:
                continue
            prediction = parts[name]
            table = predictions_frame(
                prediction.frame,
                prediction.result.labels,
                prediction.result.scores,
                prediction.errors,
            )
            table.insert(0, "split", name)
            frames.append(table)
        pd.concat(frames, ignore_index=True).to_csv(export_path, index=False)
        print(f"wrote {export_path}")


if __name__ == "__main__":
    main()
