#!/usr/bin/env python
"""Generate a labelled synthetic dataset for the Sentinel ML workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral.lab.synthetic import (
    ANOMALY_TYPES,
    SyntheticConfig,
    generate_synthetic_timeseries,
    parse_anomaly_types,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="data/lab_dataset.csv")
    parser.add_argument("--events-output", default=None,
                        help="Defaults to <output stem>.events.json")
    parser.add_argument("--points", "-n", type=int, default=2000)
    parser.add_argument("--start", default="2026-01-01T00:00:00")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--trend", type=float, default=0.0,
                        help="Total change over the series")
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--seasonal-amplitude", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--anomaly-ratio", "-r", type=float, default=0.05)
    parser.add_argument("--types", default="all",
                        help=f"Comma-separated subset of: {', '.join(ANOMALY_TYPES)}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-types", action="store_true")
    args = parser.parse_args()

    if args.list_types:
        print("\n".join(ANOMALY_TYPES))
        return

    anomaly_types = (
        ANOMALY_TYPES if args.types == "all" else parse_anomaly_types(args.types)
    )
    config = SyntheticConfig(
        n_points=args.points,
        start=args.start,
        freq=args.freq,
        trend=args.trend,
        seasonal_period=args.seasonal_period,
        seasonal_amplitude=args.seasonal_amplitude,
        noise=args.noise,
        anomaly_ratio=args.anomaly_ratio,
        anomaly_types=tuple(anomaly_types),
        seed=args.seed,
    )
    config.validate()
    dataset = generate_synthetic_timeseries(config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(output, index=False)

    events_path = (
        Path(args.events_output)
        if args.events_output
        else output.with_suffix(".events.json")
    )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event_records = dataset.events_frame().to_dict(orient="records")
    for record in event_records:
        for key in ("start_timestamp", "end_timestamp"):
            record[key] = str(record[key])
    events_path.write_text(
        json.dumps(event_records, indent=2), encoding="utf-8"
    )

    summary = dataset.summary()
    print(f"wrote {output} ({len(dataset.frame)} points)")
    print(f"wrote {events_path} ({len(dataset.events)} events)")
    ratio = summary["anomaly_points"] / len(dataset.frame)
    print("anomaly points:", summary["anomaly_points"], f"({ratio:.1%})")
    by_type = {
        key: value
        for key, value in summary.items()
        if value and key not in ("events", "anomaly_points")
    }
    print("by type:", by_type)
    print("\nnext:")
    print(f"  python scripts/train_lstm.py --input {output}")
    print(
        f"  python scripts/evaluate_lstm.py --input {output} "
        "--model artifacts/lstm_autoencoder/model_001"
    )


if __name__ == "__main__":
    main()
