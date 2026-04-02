from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

import redis

from spectral.ingestion.base import BaseSource, DataPoint

logger = logging.getLogger(__name__)


class RedisStreamSource(BaseSource):
    """Consume time-series data from a Redis Stream.

    Uses consumer groups for reliable message processing with
    automatic acknowledgment.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream_key: str = "spectral:data",
        consumer_group: str = "spectral-group",
        consumer_name: str = "spectral-worker-1",
        batch_size: int = 100,
        block_ms: int = 1000,
    ):
        self._redis = redis.from_url(redis_url)
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.batch_size = batch_size
        self.block_ms = block_ms
        self._running = True

        # create consumer group if it doesn't exist
        try:
            self._redis.xgroup_create(
                self.stream_key, self.consumer_group, id="0", mkstream=True
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            logger.debug(f"Consumer group already exists: {self.consumer_group}")

    def read(self) -> Iterator[DataPoint]:
        while self._running:
            messages = self._redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=self.batch_size,
                block=self.block_ms,
            )

            if not messages:
                continue

            for stream_name, entries in messages:
                for msg_id, fields in entries:
                    try:
                        # redis returns bytes, decode them
                        decoded = {
                            k.decode() if isinstance(k, bytes) else k: v.decode()
                            if isinstance(v, bytes)
                            else v
                            for k, v in fields.items()
                        }

                        yield DataPoint(
                            timestamp=datetime.fromisoformat(decoded["timestamp"]),
                            value=float(decoded["value"]),
                            series_id=decoded.get("series_id", "default"),
                        )

                        self._redis.xack(self.stream_key, self.consumer_group, msg_id)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse message {msg_id}: {e}")
                        continue

    def stop(self):
        self._running = False
