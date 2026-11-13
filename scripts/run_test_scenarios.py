#!/usr/bin/env python
"""Run the reproducible A-I detector benchmark.

Protocol
--------
One clean seasonal baseline is generated. Every detector is fitted on the
first 70% of that baseline; the LSTM's calibration holdout is inside that
70%. Scenario A scores the clean held-out 30%. Scenarios B-H are separate,
unseen datasets with known ground truth. Scenario I scores an unlabelled CSV
and reports predictions and scores only - never precision, recall or F1.

All matching is one-to-one within a tolerance window: one predicted point can
satisfy at most one ground-truth anomaly and vice versa.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.lab.data import load_timeseries_csv, predictions_frame, prepare_frame
from spectral.lab.evaluation import (
    DEFAULT_LSTM_PARAMS,
    predict_with_detector,
    run_detector_comparison,
    train_lstm_on_frame,
)
from spectral.lab.metrics import recall_by_type
from spectral.lab.synthetic import SyntheticConfig, generate_synthetic_timeseries

DEFAULT_UNLABELLED_CSV = "examples/example_unlabelled.csv"

SCENARIOS: list[tuple[str, str, dict | None]] = [
    ("A_clean", "Clean seasonal holdout, no anomalies", None),
    (
        "B_isolated_spikes",
        "Isolated spikes",
        dict(anomaly_ratio=0.03, anomaly_types=("spike",)),
    ),
    (
        "C_dense_spikes",
        "Densely packed spikes and drops",
        dict(anomaly_ratio=0.08, anomaly_types=("spike", "drop")),
    ),
    (
        "D_level_shift",
        "Sudden level shifts",
        dict(anomaly_ratio=0.06, anomaly_types=("level_shift",)),
    ),
    (
        "E_drift",
        "Gradual drifts",
        dict(anomaly_ratio=0.06, anomaly_types=("drift",)),
    ),
    (
        "F_noise_burst",
        "Localised noise bursts",
        dict(anomaly_ratio=0.06, anomaly_types=("noise_burst",)),
    ),
    (
        "G_flatline",
        "Stuck sensor / flatline",
        dict(anomaly_ratio=0.03, anomaly_types=("flatline",)),
    ),
    (
        "H_mixed_unseen",
        "Mixed anomaly types, unseen seed",
        dict(anomaly_ratio=0.06),
    ),
]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _fmt(metrics: dict | None, key: str, digits: int = 3) -> str:
    if not metrics or key not in metrics:
        return "-"
    value = metrics[key]
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _type_recall(
    frame: pd.DataFrame, results: dict
) -> dict[str, dict[str, dict]]:
    """Per-detector, per-type recall with one-to-one matching."""
    if "anomaly_type" not in frame.columns:
        return {}
    labels = frame["is_anomaly"].to_numpy(dtype="float")
    if np.isnan(labels).any():
        return {}
    labels = labels.astype(int)
    types = frame["anomaly_type"].to_numpy()
    output: dict[str, dict[str, dict]] = {}
    for name, result in results.items():
        per_type = recall_by_type(labels, result.labels, types)
        if per_type:
            output[name] = per_type
    return output


def _scenario_markdown(report: dict) -> list[str]:
    lines = [
        f"## {report['key']} - {report['description']}",
        "",
        f"Evaluation data: {report['evaluation_source']}. "
        f"Ground-truth anomaly points: {report['summary']['anomaly_points']}.",
        "",
        "| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged "
        "| Fit (ms) | Inference (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        fit_ms = row.get("fit_seconds")
        inference_ms = row.get("inference_seconds")
        lines.append(
            "| {detector} | {precision} | {recall} | {f1} | {fpr} | {fp} | {fn} "
            "| {flagged} | {fit} | {inference} |".format(
                detector=row["detector"],
                precision=_fmt(row, "test_precision"),
                recall=_fmt(row, "test_recall"),
                f1=_fmt(row, "test_f1"),
                fpr=_fmt(row, "test_fpr"),
                fp=_fmt(row, "test_fp"),
                fn=_fmt(row, "test_fn"),
                flagged=_fmt(row, "test_anomalies"),
                fit=(
                    f"{float(fit_ms) * 1000:.2f}"
                    if isinstance(fit_ms, (int, float)) and np.isfinite(fit_ms)
                    else "-"
                ),
                inference=(
                    f"{float(inference_ms) * 1000:.2f}"
                    if isinstance(inference_ms, (int, float)) and np.isfinite(inference_ms)
                    else "-"
                ),
            )
        )
    per_type = report.get("per_type") or {}
    if per_type:
        types = sorted(
            {
                anomaly_type
                for detector_rows in per_type.values()
                for anomaly_type in detector_rows
            }
        )
        lines.extend(
            [
                "",
                "Recall by anomaly type (one-to-one matching):",
                "",
                "| Detector | " + " | ".join(types) + " |",
                "|---|" + "---|" * len(types),
            ]
        )
        for detector, detector_rows in per_type.items():
            cells = [
                _fmt(detector_rows.get(name), "recall")
                for name in types
            ]
            lines.append(f"| {detector} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--points", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument(
        "--unlabelled-csv",
        default=DEFAULT_UNLABELLED_CSV,
        help="CSV used for scenario I; labels are ignored deliberately",
    )
    parser.add_argument(
        "--skip-unlabelled",
        action="store_true",
        help="skip scenario I when no CSV is available",
    )
    args = parser.parse_args()

    # One clean training baseline for every scenario.
    baseline_dataset = generate_synthetic_timeseries(
        SyntheticConfig(n_points=args.points, anomaly_ratio=0.0, seed=args.seed)
    )
    baseline_frame, _ = prepare_frame(baseline_dataset.frame)
    baseline_result = train_lstm_on_frame(
        baseline_frame,
        train_fraction=0.7,
        params={
            **DEFAULT_LSTM_PARAMS,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
        },
        tolerance=args.tolerance,
    )
    print(
        f"clean baseline: {len(baseline_frame)} points; trained on "
        f"{len(baseline_result.train_frame)}, scored on "
        f"{len(baseline_result.test_frame)}; fit {baseline_result.fit_seconds:.2f}s; "
        f"threshold={baseline_result.detector._threshold:.6f} "
        f"({baseline_result.detector._calibration_source} calibration)"
    )

    rows: list[dict] = []
    scenario_reports: list[dict] = []

    for index, (key, description, overrides) in enumerate(SCENARIOS):
        if key == "A_clean":
            evaluation_frame = baseline_result.test_frame
            summary = baseline_dataset.summary()
            evaluation_source = "held-out 30% of the clean baseline"
        else:
            dataset = generate_synthetic_timeseries(
                SyntheticConfig(
                    n_points=args.points,
                    seed=args.seed + index + 100,
                    **(overrides or {}),
                )
            )
            evaluation_frame, _ = prepare_frame(dataset.frame)
            summary = dataset.summary()
            evaluation_source = (
                f"separate unseen synthetic dataset (seed={args.seed + index + 100})"
            )

        comparison = run_detector_comparison(
            baseline_result.train_frame,
            evaluation_frame,
            lstm_detector=baseline_result.detector,
            tolerance=args.tolerance,
        )
        scenario_rows = comparison.rows.copy()
        scenario_rows.insert(0, "scenario", key)
        rows.extend(scenario_rows.to_dict(orient="records"))
        scenario_reports.append(
            {
                "key": key,
                "description": description,
                "evaluation_source": evaluation_source,
                "summary": summary,
                "train_points": len(baseline_result.train_frame),
                "test_points": len(evaluation_frame),
                "rows": scenario_rows.to_dict(orient="records"),
                "per_type": _type_recall(evaluation_frame, comparison.results),
            }
        )
        print(f"\n=== {key}: {description} ===")
        print(f"    evaluation: {evaluation_source}; anomalies={summary['anomaly_points']}")
        print(
            comparison.rows[
                [
                    "detector",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_fp",
                    "test_fn",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )

    # Scenario I: unlabelled CSV inference. No metrics, by design.
    unlabelled_report: dict = {
        "status": "skipped",
        "reason": "pass --unlabelled-csv or remove --skip-unlabelled",
    }
    predictions_path = Path(args.output_dir) / "unlabelled_predictions.csv"
    if not args.skip_unlabelled:
        try:
            csv_path = Path(args.unlabelled_csv)
            # label_col=None forces unlabelled treatment even if the file
            # happens to contain an is_anomaly column.
            frame, _ = load_timeseries_csv(csv_path, label_col=None)
            prediction = predict_with_detector(
                baseline_result.detector,
                frame,
                tolerance=args.tolerance,
            )
            table = predictions_frame(
                prediction.frame,
                prediction.result.labels,
                prediction.result.scores,
                prediction.errors,
            )
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(predictions_path, index=False)
            unlabelled_report = {
                "status": "ok",
                "input": str(csv_path),
                "points": len(frame),
                "has_labels": False,
                "flagged": int(prediction.result.anomaly_count),
                "flagged_ratio": float(prediction.result.anomaly_ratio),
                "score_min": float(prediction.result.scores.min()),
                "score_max": float(prediction.result.scores.max()),
                "threshold": float(prediction.result.threshold),
                "predictions_csv": str(predictions_path),
                "note": (
                    "Unlabelled inference: predictions and scores only. "
                    "Precision, recall and F1 are deliberately omitted."
                ),
            }
            print(f"\n=== I unlabelled CSV: {csv_path} ===")
            print(
                f"points={len(frame)} flagged={prediction.result.anomaly_count} "
                f"({prediction.result.anomaly_ratio:.2%}); no ground truth, no metrics"
            )
        except Exception as exc:  # keep the benchmark report even if I fails
            unlabelled_report = {
                "status": "error",
                "input": args.unlabelled_csv,
                "error": str(exc),
            }
            print(f"\n=== I unlabelled CSV failed: {exc} ===")
    else:
        print("\n=== I unlabelled CSV: skipped ===")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "One clean seasonal baseline; every detector is fitted on its first "
            "70%. Scenario A scores the clean held-out 30%; B-H are separate unseen "
            "datasets. The LSTM calibration holdout is inside the clean training "
            "half. Scenario I is an unlabelled CSV: predictions and scores only."
        ),
        "parameters": {
            "points": args.points,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "lstm_fit_seconds": baseline_result.fit_seconds,
            "threshold": baseline_result.detector._threshold,
            "calibration_source": baseline_result.detector._calibration_source,
        },
        "scenarios": scenario_reports,
        "unlabelled_csv": unlabelled_report,
    }
    json_path = output_dir / "scenario_results.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    markdown = [
        "# Sentinel ML - test scenario results",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"Points per dataset: {args.points}; LSTM epochs: {args.epochs}; "
        f"hidden: {args.hidden_dim}; sequence: {args.sequence_length}; "
        f"tolerance: +/-{args.tolerance} points.",
        "",
        "Protocol: one **clean** seasonal baseline; every detector is fitted on "
        "its first 70%. Scenario A is the clean held-out 30%; B-H are separate "
        "unseen datasets. Training contains no labelled anomaly, and the LSTM "
        "calibration holdout is inside the clean training half.",
        "",
        "One-to-one matching: each predicted point can match at most one "
        "ground-truth point, and vice versa. Precision, recall and F1 are "
        "event-level, not raw point overlap.",
        "",
        "Scenario A has no anomalies, so its precision/recall/F1 are 0 by "
        "construction; the meaningful number there is the false-positive rate.",
        "",
    ]
    for report in scenario_reports:
        markdown.extend(_scenario_markdown(report))

    markdown.extend(
        [
            "## I_unlabelled_csv - unlabelled CSV inference",
            "",
            f"Input: `{unlabelled_report.get('input', args.unlabelled_csv)}`.",
            "",
            "```json",
            json.dumps(unlabelled_report, indent=2, default=_json_default),
            "```",
            "",
            "No ground truth exists for this file, so no precision, recall or F1 "
            "is reported. Predictions and scores are saved to "
            f"`{predictions_path}`.",
            "",
        ]
    )
    markdown_path = output_dir / "scenario_results.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(f"\nwrote {json_path}")
    print(f"wrote {markdown_path}")
    if unlabelled_report.get("status") == "ok":
        print(f"wrote {predictions_path}")


if __name__ == "__main__":
    main()
