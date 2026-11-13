from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from spectral.api.service import DetectionService
from spectral.pipeline.runner import Anomaly


@dataclass
class Point:
    timestamp: datetime
    value: float
    series_id: str = "default"
    metadata: dict | None = None


def _anomaly(index: int = 3) -> Anomaly:
    return Anomaly(
        timestamp=datetime(2026, 1, 1),
        index=index,
        value=9.9,
        score=0.95,
        detector="ensemble",
        severity="critical",
    )


class FakePipeline:
    def __init__(self, anomalies=None):
        self.anomalies = anomalies or []
        self.stream_calls = 0
        self.batch_calls = 0

    def detect_streaming(self, value, timestamp=None):
        self.stream_calls += 1
        return self.anomalies.pop(0) if self.anomalies else None

    def detect(self, values, timestamps=None):
        self.batch_calls += 1
        return list(self.anomalies)

    @property
    def stats(self):
        return {
            "points_processed": 0,
            "anomalies_detected": 0,
            "detectors": ["fake"],
            "fitted": True,
        }


class FakeRepository:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.anomalies = []
        self.batch = []
        self.points = []

    async def save_anomaly(self, anomaly, series_id="default"):
        if self.fail:
            raise RuntimeError("database down")
        self.anomalies.append((anomaly, series_id))
        return "id"

    async def save_batch(self, anomalies, series_id="default"):
        if self.fail:
            raise RuntimeError("database down")
        self.batch.append((list(anomalies), series_id))
        return len(anomalies)

    async def save_data_point(self, timestamp, value, series_id="default", metadata=None):
        if self.fail:
            raise RuntimeError("database down")
        self.points.append((timestamp, value, series_id, metadata))
        return "id"


class FakeAlerts:
    def __init__(self):
        self.calls = []
        self.closed = False

    def dispatch(self, anomaly, series_id="default"):
        self.calls.append((anomaly, series_id))
        return True

    def close(self):
        self.closed = True


class FakeBroadcaster:
    def __init__(self):
        self.calls = []

    async def broadcast_anomaly(self, anomaly, series_id="default"):
        self.calls.append((anomaly, series_id))


class TestDetectionService:
    async def test_ingest_fans_out_to_every_consumer(self):
        pipeline = FakePipeline([_anomaly()])
        repo = FakeRepository()
        alerts = FakeAlerts()
        broadcaster = FakeBroadcaster()
        service = DetectionService(
            pipeline,
            repository=repo,
            alerts=alerts,
            broadcaster=broadcaster,
            storage_enabled=True,
        )
        points = [
            Point(datetime(2026, 1, 1, 0, 0), 1.0, "sensor-a", {"unit": "C"}),
            Point(datetime(2026, 1, 1, 0, 1), 2.0, "sensor-a", None),
        ]

        anomalies = await service.ingest(points)

        assert len(anomalies) == 1
        assert pipeline.stream_calls == 2
        assert repo.points == [
            (datetime(2026, 1, 1, 0, 0), 1.0, "sensor-a", {"unit": "C"}),
            (datetime(2026, 1, 1, 0, 1), 2.0, "sensor-a", None),
        ]
        assert repo.anomalies[0][1] == "sensor-a"
        assert anomalies[0].metadata["series_id"] == "sensor-a"
        assert alerts.calls[0][1] == "sensor-a"
        assert broadcaster.calls[0][1] == "sensor-a"
        await service.aclose()
        assert alerts.closed is True

    async def test_detect_batch_persists_once_and_notifies_each(self):
        pipeline = FakePipeline([_anomaly(1), _anomaly(2)])
        repo = FakeRepository()
        alerts = FakeAlerts()
        broadcaster = FakeBroadcaster()
        service = DetectionService(
            pipeline,
            repository=repo,
            alerts=alerts,
            broadcaster=broadcaster,
            storage_enabled=True,
        )

        anomalies = await service.detect_batch(
            [1.0, 2.0, 3.0], series_id="sensor-b"
        )

        assert len(anomalies) == 2
        assert pipeline.batch_calls == 1
        assert len(repo.batch) == 1
        assert repo.batch[0][1] == "sensor-b"
        assert len(alerts.calls) == 2
        assert len(broadcaster.calls) == 2

    async def test_storage_disabled_never_touches_the_repository(self):
        pipeline = FakePipeline([_anomaly()])
        repo = FakeRepository()
        service = DetectionService(
            pipeline, repository=repo, storage_enabled=False
        )

        await service.detect_batch([1.0, 2.0])
        await service.ingest([Point(datetime(2026, 1, 1), 1.0)])

        assert repo.anomalies == []
        assert repo.batch == []
        assert repo.points == []

    async def test_database_failure_degrades_without_losing_detection(self):
        pipeline = FakePipeline([_anomaly()])
        repo = FakeRepository(fail=True)
        alerts = FakeAlerts()
        service = DetectionService(
            pipeline,
            repository=repo,
            alerts=alerts,
            storage_enabled=True,
        )

        anomalies = await service.detect_batch([1.0, 2.0])

        assert len(anomalies) == 1
        assert len(alerts.calls) == 1
        # A database outage must not stop later detections or alerts; only the
        # first persistence failure is logged to avoid log floods.
        await service.detect_batch([3.0, 4.0])
        assert len(alerts.calls) == 2
        assert service._persistence_warned is True

    async def test_alert_failure_does_not_block_broadcast(self):
        class BrokenAlerts:
            def dispatch(self, anomaly, series_id="default"):
                raise RuntimeError("webhook down")

            def close(self):
                pass

        broadcaster = FakeBroadcaster()
        service = DetectionService(
            FakePipeline([_anomaly()]),
            alerts=BrokenAlerts(),
            broadcaster=broadcaster,
            storage_enabled=False,
        )

        anomalies = await service.detect_batch([1.0])

        assert len(anomalies) == 1
        assert len(broadcaster.calls) == 1

    async def test_empty_batch_is_safe(self):
        service = DetectionService(FakePipeline([]), storage_enabled=False)
        assert await service.detect_batch(np.array([], dtype=float)) == []
