from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from spectral.alerting.dispatcher import AlertDispatcher
from spectral.alerting.webhook import WebhookDispatcher
from spectral.config.schema import AlertingConfig
from spectral.pipeline.runner import Anomaly


@pytest.fixture
def sample_anomaly():
    return Anomaly(
        timestamp=datetime(2026, 1, 1),
        index=0,
        value=10.5,
        score=0.95,
        detector="zscore",
        severity="critical",
    )


@pytest.fixture
def low_severity_anomaly():
    return Anomaly(
        timestamp=datetime(2026, 1, 1),
        index=0,
        value=1.5,
        score=0.3,
        detector="zscore",
        severity="low",
    )


class TestWebhookDispatcher:
    @patch("spectral.alerting.webhook.httpx.Client")
    def test_sends_post_request(self, mock_client_cls, sample_anomaly):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        dispatcher = WebhookDispatcher(url="https://hooks.example.com/test")
        dispatcher._client = mock_client
        result = dispatcher.send(sample_anomaly, series_id="test")

        assert result is True
        mock_client.post.assert_called_once()

    @patch("spectral.alerting.webhook.httpx.Client")
    def test_cooldown_prevents_duplicate_alerts(self, mock_client_cls, sample_anomaly):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        dispatcher = WebhookDispatcher(url="https://hooks.example.com/test", cooldown_seconds=300)
        dispatcher._client = mock_client

        # first send should succeed
        assert dispatcher.send(sample_anomaly, series_id="series-a") is True
        # second send to same series should be suppressed
        assert dispatcher.send(sample_anomaly, series_id="series-a") is False
        # different series should still work
        assert dispatcher.send(sample_anomaly, series_id="series-b") is True


class TestAlertDispatcher:
    def test_severity_filter(self, low_severity_anomaly):
        config = AlertingConfig(
            enabled=True,
            webhook_url="https://hooks.example.com/test",
            min_severity="warning",
        )
        dispatcher = AlertDispatcher(config)
        assert dispatcher._should_alert(low_severity_anomaly) is False

    def test_disabled_does_not_send(self, sample_anomaly):
        config = AlertingConfig(enabled=False)
        dispatcher = AlertDispatcher(config)
        result = dispatcher.dispatch(sample_anomaly)
        assert result is False

    def test_critical_passes_filter(self, sample_anomaly):
        config = AlertingConfig(
            enabled=True,
            webhook_url="https://hooks.example.com/test",
            min_severity="warning",
        )
        dispatcher = AlertDispatcher(config)
        assert dispatcher._should_alert(sample_anomaly) is True
