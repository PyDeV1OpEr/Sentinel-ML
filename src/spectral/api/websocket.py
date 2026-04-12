from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from spectral.pipeline.runner import Anomaly

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for real-time anomaly streaming."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(f"WebSocket connected, total={len(self._connections)}")

    def disconnect(self, websocket: WebSocket):
        self._connections.remove(websocket)
        logger.info(f"WebSocket disconnected, total={len(self._connections)}")

    async def broadcast_anomaly(self, anomaly: Anomaly, series_id: str = "default"):
        """Send anomaly alert to all connected clients."""
        if not self._connections:
            return

        message = {
            "type": "anomaly",
            "data": {
                "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None,
                "value": anomaly.value,
                "score": anomaly.score,
                "detector": anomaly.detector,
                "severity": anomaly.severity,
                "series_id": series_id,
                "index": anomaly.index,
            },
        }
        payload = json.dumps(message)

        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self._connections.remove(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time anomaly alerts.

    Clients connect and receive JSON messages whenever an
    anomaly is detected. Supports heartbeat pings.
    """
    await manager.connect(websocket)
    try:
        while True:
            # keep connection alive, handle client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
