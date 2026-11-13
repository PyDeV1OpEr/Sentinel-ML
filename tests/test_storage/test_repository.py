from datetime import datetime

import pytest

from spectral.pipeline.runner import Anomaly


class TestAnomalyRepository:
    """Repository tests are integration tests that require a database.

    Run with: pytest -m integration
    For unit testing, the pipeline and detector tests cover the core logic.
    """

    @pytest.mark.integration
    async def test_save_and_query(self):
        from spectral.storage.repository import AnomalyRepository

        repo = AnomalyRepository()
        anomaly = Anomaly(
            timestamp=datetime(2026, 1, 1),
            index=0,
            value=10.5,
            score=0.95,
            detector="zscore",
            severity="critical",
        )

        anomaly_id = await repo.save_anomaly(anomaly, series_id="test")
        assert anomaly_id is not None

        results = await repo.query(series_id="test", limit=10)
        assert len(results) >= 1

    @pytest.mark.integration
    async def test_batch_save(self):
        from spectral.storage.repository import AnomalyRepository

        repo = AnomalyRepository()
        anomalies = [
            Anomaly(
                timestamp=datetime(2026, 1, 1, 0, i),
                index=i,
                value=float(i),
                score=0.8,
                detector="mad",
                severity="high",
            )
            for i in range(5)
        ]

        count = await repo.save_batch(anomalies, series_id="batch-test")
        assert count == 5

    @pytest.mark.integration
    async def test_stats(self):
        from spectral.storage.repository import AnomalyRepository

        repo = AnomalyRepository()
        stats = await repo.get_stats(window_hours=24)
        assert "total_anomalies" in stats
        assert "per_detector" in stats
