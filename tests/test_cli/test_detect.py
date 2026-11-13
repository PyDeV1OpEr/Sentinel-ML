import csv
import json
from datetime import datetime, timedelta

from click.testing import CliRunner

from spectral.cli.main import cli


def _write_series(path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        start = datetime(2026, 1, 1)
        for i in range(120):
            value = 0.0 if i != 60 else 10.0
            writer.writerow([(start + timedelta(minutes=i)).isoformat(), value])


def test_detect_json_stdout_is_parseable(tmp_path):
    csv_path = tmp_path / "series.csv"
    _write_series(csv_path)

    result = CliRunner().invoke(cli, ["detect", str(csv_path), "--format", "json"])

    assert result.exit_code == 0
    assert isinstance(json.loads(result.stdout), list)
    assert "Processed 120 points" in result.stderr


def test_detect_csv_stdout_starts_with_header(tmp_path):
    csv_path = tmp_path / "series.csv"
    _write_series(csv_path)

    result = CliRunner().invoke(cli, ["detect", str(csv_path), "--format", "csv"])

    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == "timestamp,index,value,score,severity,detector"
