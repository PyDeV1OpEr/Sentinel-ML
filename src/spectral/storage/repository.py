from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select

from spectral.pipeline.runner import Anomaly
from spectral.storage.database import get_session
from spectral.storage.models import AnomalyModel, DataPointModel

logger = logging.getLogger(__name__)


class AnomalyRepository:
    """Persistence layer for anomaly detection results."""

    async def save_anomaly(self, anomaly: Anomaly, series_id: str = "default") -> str:
        """Save a single anomaly to the database. Returns the anomaly ID."""
        async with get_session() as session:
            record = AnomalyModel(
                series_id=series_id,
                timestamp=anomaly.timestamp or datetime.utcnow(),
                value=anomaly.value,
                score=anomaly.score,
                detector=anomaly.detector,
                severity=anomaly.severity,
                metadata_=anomaly.metadata,
            )
            session.add(record)
            await session.flush()
            return record.id

    async def save_batch(self, anomalies: list[Anomaly], series_id: str = "default") -> int:
        """Save multiple anomalies in a batch. Returns count saved."""
        async with get_session() as session:
            records = [
                AnomalyModel(
                    series_id=series_id,
                    timestamp=a.timestamp or datetime.utcnow(),
                    value=a.value,
                    score=a.score,
                    detector=a.detector,
                    severity=a.severity,
                    metadata_=a.metadata,
                )
                for a in anomalies
            ]
            session.add_all(records)
            return len(records)

    async def query(
        self,
        series_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        min_severity: str | None = None,
        detector: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AnomalyModel]:
        """Query stored anomalies with filters."""
        severity_order = {"low": 0, "warning": 1, "high": 2, "critical": 3}

        async with get_session() as session:
            stmt = select(AnomalyModel).order_by(AnomalyModel.timestamp.desc())

            if series_id:
                stmt = stmt.where(AnomalyModel.series_id == series_id)
            if start:
                stmt = stmt.where(AnomalyModel.timestamp >= start)
            if end:
                stmt = stmt.where(AnomalyModel.timestamp <= end)
            if detector:
                stmt = stmt.where(AnomalyModel.detector == detector)
            if min_severity and min_severity in severity_order:
                min_level = severity_order[min_severity]
                valid = [s for s, level in severity_order.items() if level >= min_level]
                stmt = stmt.where(AnomalyModel.severity.in_(valid))

            stmt = stmt.offset(offset).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_stats(
        self,
        series_id: str | None = None,
        window_hours: int = 24,
    ) -> dict:
        """Get aggregate stats for anomalies in a time window."""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)

        async with get_session() as session:
            stmt = select(
                func.count(AnomalyModel.id).label("total"),
                func.avg(AnomalyModel.score).label("avg_score"),
            ).where(AnomalyModel.created_at >= cutoff)

            if series_id:
                stmt = stmt.where(AnomalyModel.series_id == series_id)

            result = await session.execute(stmt)
            row = result.one()

            # per-detector breakdown
            det_stmt = (
                select(
                    AnomalyModel.detector,
                    func.count(AnomalyModel.id).label("count"),
                )
                .where(AnomalyModel.created_at >= cutoff)
                .group_by(AnomalyModel.detector)
            )
            if series_id:
                det_stmt = det_stmt.where(AnomalyModel.series_id == series_id)

            det_result = await session.execute(det_stmt)
            per_detector = {r.detector: r.count for r in det_result.all()}

            return {
                "total_anomalies": row.total or 0,
                "avg_score": float(row.avg_score) if row.avg_score else 0.0,
                "window_hours": window_hours,
                "per_detector": per_detector,
            }

    async def save_data_point(
        self,
        timestamp: datetime,
        value: float,
        series_id: str = "default",
        metadata: dict | None = None,
    ) -> str:
        """Persist a raw data point."""
        async with get_session() as session:
            record = DataPointModel(
                series_id=series_id,
                timestamp=timestamp,
                value=value,
                metadata_=metadata or {},
            )
            session.add(record)
            await session.flush()
            return record.id
