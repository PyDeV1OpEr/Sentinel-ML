import json

import pytest
from fastapi.testclient import TestClient

from spectral.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestWebSocket:
    def test_websocket_connects(self, client):
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text("ping")
            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["type"] == "pong"
