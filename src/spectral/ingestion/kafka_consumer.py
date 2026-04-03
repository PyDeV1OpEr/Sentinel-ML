from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Iterator

from spectral.ingestion.base import BaseSource, DataPoint

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Consumer, KafkaError
except ImportError:
    Consumer = None
    KafkaError = None


class KafkaSource(BaseSource):
    """Consume time-series data from a Kafka topic.

    Expects JSON messages with at least 'timestamp' and 'value' fields.
    Install confluent-kafka to use: pip install spectral[kafka]
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "spectral-data",
        group_id: str = "spectral-consumer",
        value_field: str = "value",
        timestamp_field: str = "timestamp",
        auto_offset_reset: str = "latest",
    ):
        if Consumer is None:
            raise ImportError(
                "confluent-kafka is required for Kafka ingestion. "
                "Install it with: pip install spectral[kafka]"
            )

        self.topic = topic
        self.value_field = value_field
        self.timestamp_field = timestamp_field

        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": True,
        })
        self._consumer.subscribe([topic])
        self._running = True
        logger.info(f"Kafka consumer initialized: topic={topic}, group={group_id}")

    def read(self) -> Iterator[DataPoint]:
        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    yield DataPoint(
                        timestamp=datetime.fromisoformat(data[self.timestamp_field]),
                        value=float(data[self.value_field]),
                        series_id=data.get("series_id", "default"),
                        metadata={
                            k: v
                            for k, v in data.items()
                            if k not in (self.timestamp_field, self.value_field, "series_id")
                        },
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse message: {e}")
                    continue
        finally:
            self._consumer.close()
            logger.info("Kafka consumer closed")

    def stop(self):
        self._running = False
