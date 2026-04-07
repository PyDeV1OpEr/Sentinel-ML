from __future__ import annotations

import logging
import time

import httpx

from spectral.pipeline.runner import Anomaly

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """Send anomaly alerts via webhook POST requests.

    Includes cooldown tracking per series to prevent alert storms
    and retry logic with exponential backoff.
    """

    def __init__(
        self,
        url: str,
        cooldown_seconds: int = 300,
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        self.url = url
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_alert: dict[str, float] = {}
        self._client = httpx.Client(timeout=timeout)

    def _is_on_cooldown(self, series_id: str) -> bool:
        """Check if we're still in cooldown for this series."""
        last = self._last_alert.get(series_id)
        if last is None:
            return False
        return (time.time() - last) < self.cooldown_seconds

    def _build_payload(self, anomaly: Anomaly, series_id: str) -> dict:
        """Build a Slack-compatible webhook payload."""
        severity_emoji = {
            "critical": ":red_circle:",
            "high": ":large_orange_circle:",
            "warning": ":warning:",
            "low": ":white_circle:",
        }
        emoji = severity_emoji.get(anomaly.severity, ":question:")

        ts_str = anomaly.timestamp.isoformat() if anomaly.timestamp else "N/A"

        return {
            "text": f"{emoji} Anomaly detected in series `{series_id}`",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Anomaly Alert - {anomaly.severity.upper()}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Series:* `{series_id}`"},
                        {"type": "mrkdwn", "text": f"*Detector:* {anomaly.detector}"},
                        {"type": "mrkdwn", "text": f"*Score:* {anomaly.score:.4f}"},
                        {"type": "mrkdwn", "text": f"*Value:* {anomaly.value:.4f}"},
                        {"type": "mrkdwn", "text": f"*Timestamp:* {ts_str}"},
                        {"type": "mrkdwn", "text": f"*Severity:* {anomaly.severity}"},
                    ],
                },
            ],
        }

    def send(self, anomaly: Anomaly, series_id: str = "default") -> bool:
        """Send an alert for the given anomaly. Returns True if sent."""
        if self._is_on_cooldown(series_id):
            logger.debug(f"Alert suppressed for {series_id} (cooldown)")
            return False

        payload = self._build_payload(anomaly, series_id)

        for attempt in range(self.max_retries):
            try:
                response = self._client.post(self.url, json=payload)
                response.raise_for_status()
                self._last_alert[series_id] = time.time()
                logger.info(
                    f"Alert sent for {series_id}: "
                    f"severity={anomaly.severity}, score={anomaly.score:.4f}"
                )
                return True
            except httpx.HTTPError as e:
                wait = 2**attempt
                logger.warning(
                    f"Webhook attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait}s"
                )
                time.sleep(wait)

        logger.error(f"Failed to send alert after {self.max_retries} attempts")
        return False

    def close(self):
        self._client.close()
