from __future__ import annotations

import logging

from spectral.alerting.webhook import WebhookDispatcher
from spectral.config.schema import AlertingConfig
from spectral.pipeline.runner import Anomaly

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"low": 0, "warning": 1, "high": 2, "critical": 3}


class AlertDispatcher:
    """Route anomaly alerts through configured channels.

    Handles severity filtering and delegates to the appropriate
    dispatcher (currently webhook, extensible to email/PagerDuty/etc).
    """

    def __init__(self, config: AlertingConfig):
        self.config = config
        self._min_level = SEVERITY_LEVELS.get(config.min_severity, 1)
        self._dispatchers = []

        if config.enabled and config.webhook_url:
            self._dispatchers.append(
                WebhookDispatcher(
                    url=config.webhook_url,
                    cooldown_seconds=config.cooldown_seconds,
                )
            )
            logger.info(f"Webhook alerting enabled, cooldown={config.cooldown_seconds}s")

    def _should_alert(self, anomaly: Anomaly) -> bool:
        """Check if this anomaly meets the severity threshold."""
        level = SEVERITY_LEVELS.get(anomaly.severity, 0)
        return level >= self._min_level

    def dispatch(self, anomaly: Anomaly, series_id: str = "default") -> bool:
        """Send alert through all configured channels.

        Returns True if at least one channel succeeded.
        """
        if not self.config.enabled:
            return False

        if not self._should_alert(anomaly):
            logger.debug(
                f"Anomaly below severity threshold: {anomaly.severity} < {self.config.min_severity}"
            )
            return False

        sent = False
        for dispatcher in self._dispatchers:
            if dispatcher.send(anomaly, series_id):
                sent = True
        return sent

    def close(self):
        for dispatcher in self._dispatchers:
            if hasattr(dispatcher, "close"):
                dispatcher.close()
