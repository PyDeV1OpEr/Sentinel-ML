import csv
from datetime import datetime

import pytest

from spectral.ingestion.csv_loader import CSVSource


class TestCSVSource:
    def test_reads_csv_file(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
            writer.writerow(["2026-01-01T00:00:00", "1.0"])
            writer.writerow(["2026-01-01T00:01:00", "2.0"])
            writer.writerow(["2026-01-01T00:02:00", "3.0"])

        source = CSVSource(csv_file)
        points = list(source.read())

        assert len(points) == 3
        assert points[0].value == 1.0
        assert points[2].value == 3.0

    def test_auto_detects_timestamp_format(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
            writer.writerow(["2026/01/01 00:00:00", "1.0"])
            writer.writerow(["2026/01/01 00:01:00", "2.0"])

        source = CSVSource(csv_file)
        points = list(source.read())

        assert len(points) == 2
        assert isinstance(points[0].timestamp, datetime)

    def test_missing_column_raises(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "val"])
            writer.writerow(["2026-01-01", "1.0"])

        source = CSVSource(csv_file)
        with pytest.raises(ValueError, match="Timestamp column"):
            list(source.read())

    def test_custom_column_names(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "reading", "sensor"])
            writer.writerow(["2026-01-01T00:00:00", "42.5", "temp-1"])

        source = CSVSource(
            csv_file, timestamp_col="ts", value_col="reading", series_id_col="sensor"
        )
        points = list(source.read())

        assert len(points) == 1
        assert points[0].value == 42.5
        assert points[0].series_id == "temp-1"

    def test_unsupported_format_raises(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a csv")

        source = CSVSource(txt_file)
        with pytest.raises(ValueError, match="Unsupported"):
            list(source.read())
