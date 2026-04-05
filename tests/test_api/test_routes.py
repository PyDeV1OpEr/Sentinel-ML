import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectral.api.app import create_app
from spectral.api.dependencies import set_pipeline
from spectral.config import get_default_config
from spectral.pipeline.runner import Pipeline


@pytest.fixture
def client():
    app = create_app()
    config = get_default_config()
    pipeline = Pipeline(config)
    # fit on some data so it's ready
    rng = np.random.default_rng(42)
    train = np.sin(np.arange(500) * 0.1) + rng.standard_normal(500) * 0.1
    pipeline.fit(train)
    set_pipeline(pipeline)

    with TestClient(app) as c:
        yield c


class TestRoutes:
    def test_status_endpoint(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "state" in data
        assert "detectors" in data

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
