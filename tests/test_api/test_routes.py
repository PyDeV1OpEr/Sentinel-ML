import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectral.api.app import create_app
from spectral.api.dependencies import set_pipeline, set_service
from spectral.api.service import DetectionService
from spectral.api.websocket import manager
from spectral.config import get_default_config
from spectral.pipeline.runner import Pipeline


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        # Startup installs its own service; replace it with one that shares the
        # fitted pipeline and has no database dependency.
        config = get_default_config()
        pipeline = Pipeline(config)
        rng = np.random.default_rng(42)
        train = np.sin(np.arange(500) * 0.1) + rng.standard_normal(500) * 0.1
        pipeline.fit(train)
        set_pipeline(pipeline)
        set_service(DetectionService(pipeline, broadcaster=manager))
        yield c
    set_service(None)


class TestRoutes:
    def test_status_endpoint(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "state" in data
        assert "detectors" in data
        assert data["uptime_seconds"] >= 0

    def test_detect_batch(self, client):
        # create data with an obvious anomaly
        values = list(np.sin(np.arange(100) * 0.1))
        values[50] = 10.0  # spike

        response = client.post("/detect", json={"values": values})
        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data
        assert data["total_points"] == 100

    def test_ingest_single_point(self, client):
        response = client.post(
            "/ingest",
            json={
                "timestamp": "2026-01-01T00:00:00",
                "value": 1.5,
                "series_id": "test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 1

    def test_ingest_batch(self, client):
        points = [
            {"timestamp": f"2026-01-01T00:{i:02d}:00", "value": float(i)}
            for i in range(10)
        ]
        response = client.post("/ingest", json={"points": points})
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 10

    def test_detect_with_timestamps(self, client):
        values = list(np.sin(np.arange(50) * 0.1))
        timestamps = [f"2026-01-01T00:{i:02d}:00" for i in range(50)]

        response = client.post(
            "/detect", json={"values": values, "timestamps": timestamps}
        )
        assert response.status_code == 200

    def test_detect_rejects_mismatched_timestamps(self, client):
        response = client.post(
            "/detect",
            json={"values": [1.0, 2.0], "timestamps": ["2026-01-01T00:00:00"]},
        )
        assert response.status_code == 422

    def test_detect_rejects_non_finite_values(self, client):
        # JSON does not allow NaN; send a string that Pydantic must reject.
        response = client.post("/detect", json={"values": [1.0, "not-a-number"]})
        assert response.status_code == 422

    def test_storage_disabled_is_explicit(self, client):
        assert client.get("/anomalies").status_code == 503
        metrics = client.get("/metrics").json()
        assert metrics["storage_enabled"] is False
        assert metrics["anomalies"]["total_anomalies"] == 0
