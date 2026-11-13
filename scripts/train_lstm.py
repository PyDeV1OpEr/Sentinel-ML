#!/usr/bin/env python
"""Train, evaluate and persist an LSTM autoencoder artefact."""

from __future__ import annotations

import argparse
import json

from spectral.lab.artifacts import save_lstm_artifact
from spectral.lab.data import load_timeseries_csv
from spectral.lab.evaluation import train_lstm_on_frame


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
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--label-col", default="is_anomaly")
    parser.add_argument("--output-root", default="artifacts/lstm_autoencoder")
    parser.add_argument("--model-version", default=None,
                        help="Defaults to the next model_NNN under --output-root")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    frame, has_labels = load_timeseries_csv(
        args.input,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        label_col=args.label_col,
    )
    print(f"loaded {len(frame)} points from {args.input}")
    print(f"labels: {'yes' if has_labels else 'no'}; missing values: "
          f"{int(frame['is_missing'].sum())}")

    params = {
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "threshold_sigma": args.threshold_sigma,
        "patience": args.patience,
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
    }
    result = train_lstm_on_frame(
        frame,
        train_fraction=args.train_fraction,
        params=params,
        tolerance=args.tolerance,
    )

    print(f"trained on {len(result.train_frame)} points, "
          f"held out {len(result.test_frame)} points")
    print(f"fit {result.fit_seconds:.2f}s; inference {result.inference_seconds:.3f}s; "
          f"process peak RSS {result.peak_memory_mb:.0f} MiB")
    print(f"threshold={result.detector._threshold:.6f} "
          f"({result.detector._calibration_source} calibration)")
    _print_metrics("train", result.train_metrics)
    _print_metrics("held-out", result.test_metrics)

    if args.no_save:
        print("--no-save set; artefact not written")
        return

    evaluation = {
        "train": result.train_metrics,
        "test": result.test_metrics,
    }
    path = save_lstm_artifact(
        result.detector,
        args.output_root,
        model_version=args.model_version,
        dataset_frame=frame,
        train_points=len(result.train_frame),
        evaluation=evaluation,
        fit_seconds=result.fit_seconds,
    )
    print(f"\nsaved artefact: {path}")
    print("files:", ", ".join(sorted(child.name for child in path.iterdir())))
    metadata = json.loads((path / "metadata.json").read_text())
    print(f"model version: {metadata['model_version']}  created: {metadata['created_at']}")
    print("\nnext:")
    print(f"  python scripts/predict.py --input {args.input} --model {path}")
    print(f"  python scripts/evaluate_lstm.py --input {args.input} --model {path}")


if __name__ == "__main__":
    main()
