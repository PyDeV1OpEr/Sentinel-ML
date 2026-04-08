from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from spectral import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """Spectral - Real-time anomaly detection for time-series data."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Pipeline config YAML file")
@click.option("--timestamp-col", default="timestamp", help="Timestamp column name")
@click.option("--value-col", default="value", help="Value column name")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "csv"]), default="table")
def detect(file: str, config: str | None, timestamp_col: str, value_col: str, fmt: str):
    """Run anomaly detection on a data file."""
    from spectral.config import get_default_config, load_config
    from spectral.ingestion.csv_loader import CSVSource
    from spectral.pipeline.runner import Pipeline

    if config:
        pipeline_config = load_config(config)
    else:
        pipeline_config = get_default_config()

    source = CSVSource(file, timestamp_col=timestamp_col, value_col=value_col)
    points = list(source.read())

    if not points:
        console.print("[red]No data points found in file[/red]")
        sys.exit(1)

    values = np.array([p.value for p in points])
    timestamps = [p.timestamp for p in points]

    console.print(f"Loaded {len(points)} data points from {file}")

    pipeline = Pipeline(pipeline_config)
    anomalies = pipeline.detect(values, timestamps)

    if fmt == "table":
        table = Table(title=f"Anomalies Detected: {len(anomalies)}")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Score", justify="right", style="yellow")
        table.add_column("Severity", style="bold")
        table.add_column("Detector")

        for a in anomalies:
            severity_color = {
                "critical": "red",
                "high": "dark_orange",
                "warning": "yellow",
                "low": "dim",
            }.get(a.severity, "white")

            table.add_row(
                str(a.timestamp) if a.timestamp else f"idx:{a.index}",
                f"{a.value:.4f}",
                f"{a.score:.4f}",
                f"[{severity_color}]{a.severity}[/{severity_color}]",
                a.detector,
            )
        console.print(table)

    elif fmt == "json":
        import json

        output = [
            {
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                "index": a.index,
                "value": a.value,
                "score": a.score,
                "severity": a.severity,
                "detector": a.detector,
            }
            for a in anomalies
        ]
        console.print(json.dumps(output, indent=2))

    elif fmt == "csv":
        click.echo("timestamp,index,value,score,severity,detector")
        for a in anomalies:
            ts = a.timestamp.isoformat() if a.timestamp else ""
            click.echo(f"{ts},{a.index},{a.value:.4f},{a.score:.4f},{a.severity},{a.detector}")

    stats = pipeline.stats
    console.print(
        f"\n[dim]Processed {stats['points_processed']} points, "
        f"found {stats['anomalies_detected']} anomalies "
        f"({stats['anomalies_detected']/stats['points_processed']:.1%})[/dim]"
    )


@cli.command()
@click.option("--points", "-n", default=1000, help="Number of data points")
@click.option("--anomaly-ratio", "-r", default=0.03, help="Fraction of anomalies")
@click.option("--noise", default=0.1, help="Noise level (std dev)")
@click.option("--patterns", "-p", multiple=True, default=["sine", "seasonal"],
              help="Signal patterns to include")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--seed", default=42, help="Random seed")
def generate(
    points: int,
    anomaly_ratio: float,
    noise: float,
    patterns: tuple[str],
    output: str | None,
    seed: int,
):
    """Generate synthetic time-series data with anomalies."""
    from spectral.ingestion.synthetic import SyntheticSource

    source = SyntheticSource(seed=seed)
    data_points, labels = source.generate(
        n_points=points,
        anomaly_ratio=anomaly_ratio,
        noise_level=noise,
        patterns=list(patterns),
    )

    anomaly_count = int(labels.sum())
    console.print(
        f"Generated {points} points with {anomaly_count} anomalies ({anomaly_count/points:.1%})"
    )

    if output:
        import csv

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value", "is_anomaly"])
            for dp, label in zip(data_points, labels):
                writer.writerow([dp.timestamp.isoformat(), dp.value, int(label)])

        console.print(f"Saved to {output_path}")
    else:
        table = Table(title="Sample (first 20 points)")
        table.add_column("Timestamp")
        table.add_column("Value", justify="right")
        table.add_column("Anomaly")

        for dp, label in zip(data_points[:20], labels[:20]):
            style = "red bold" if label else "dim"
            table.add_row(
                str(dp.timestamp),
                f"{dp.value:.4f}",
                "[red]YES[/red]" if label else "no",
                style=style,
            )
        console.print(table)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--config", "-c", type=click.Path(exists=True), help="Pipeline config YAML file")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, config: str | None, reload: bool):
    """Start the Spectral API server."""
    import os

    import uvicorn

    if config:
        os.environ["SPECTRAL_CONFIG"] = config

    console.print(f"Starting Spectral server on {host}:{port}")
    uvicorn.run(
        "spectral.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--url", default="http://localhost:8000", help="Server URL")
def status(url: str):
    """Check the status of a running Spectral server."""
    import httpx

    try:
        response = httpx.get(f"{url}/status", timeout=5.0)
        response.raise_for_status()
        data = response.json()

        table = Table(title="Pipeline Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Name", data["name"])
        table.add_row("State", f"[green]{data['state']}[/green]")
        table.add_row("Points Processed", str(data["points_processed"]))
        table.add_row("Anomalies Detected", str(data["anomalies_detected"]))
        table.add_row("Detectors", ", ".join(data["detectors"]))

        console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {url}[/red]")
        sys.exit(1)
    except httpx.HTTPError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
