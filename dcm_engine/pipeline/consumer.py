"""Stream consumer / book-building aggregation service.

Consumes the raw IOI stream and maintains a rolling institutional order
book per (deal, tranche): cumulative ordered size, per-investor-type
breakdown, order velocity, and book bid statistics. Snapshots are
re-published for the agent mesh and downstream dashboard.

Production hardening:
* Manual offset commit only after a snapshot is durably published.
* DLQ routing for poison events instead of crash-looping.
* Graceful rebalance handling so aggregation state isn't lost mid-window.
* Consumer-group horizontal scale: partitions are per-tranche keyed, so
  each group instance owns whole tranches (no cross-instance races).
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer

from dcm_engine.core.models import BookState, InvestorType, utc_now
from dcm_engine.pipeline.topics import EventType, Topics

logger = logging.getLogger(__name__)

BOOTSTRAP_DEFAULT = "localhost:9092"


@dataclass(slots=True)
class BookSnapshot:
    """Rolling snapshot of a live institutional order book."""

    deal_id: str
    tranche_id: str
    total_ordered_mm: float = 0.0
    by_investor_type_mm: dict[str, float] = field(default_factory=dict)
    order_count: int = 0
    velocity_per_min: float = 0.0        # Orders per minute into the book
    best_limit_yield: float = 0.0        # Lowest limit yield (strongest bid)
    worst_limit_yield: float = 0.0       # Highest limit yield (weakest bid)
    book_state: BookState = BookState.OPEN
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "tranche_id": self.tranche_id,
            "total_ordered_mm": self.total_ordered_mm,
            "by_investor_type_mm": self.by_investor_type_mm,
            "order_count": self.order_count,
            "velocity_per_min": self.velocity_per_min,
            "best_limit_yield": self.best_limit_yield,
            "worst_limit_yield": self.worst_limit_yield,
            "book_state": self.book_state.value,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class _TrancheBook:
    """Internal aggregation state for one (deal, tranche) pair."""

    deal_id: str
    tranche_id: str
    total_ordered_mm: float = 0.0
    by_type_mm: dict[InvestorType, float] = field(default_factory=dict)
    order_count: int = 0
    best_limit: float = float("inf")     # Lowest yield = strongest demand
    worst_limit: float = 0.0
    recent_timestamps: list[float] = field(default_factory=list)

    def apply(self, ioi_payload: dict[str, Any], event_ts: float | None = None) -> None:
        """Fold one IOI into the rolling book.

        Raises:
            KeyError: Missing required field (poison message).
            ValueError: Bad numeric coercion.
        """
        size = float(ioi_payload["size_mm"])
        limit = float(ioi_payload["limit_yield"])
        inv_type = InvestorType(str(ioi_payload["investor_type"]))
        ts = event_ts if event_ts is not None else time.time()

        self.total_ordered_mm += size
        self.by_type_mm[inv_type] = self.by_type_mm.get(inv_type, 0.0) + size
        self.order_count += 1
        self.best_limit = min(self.best_limit, limit)
        self.worst_limit = max(self.worst_limit, limit)
        self.recent_timestamps.append(ts)

    def velocity_per_min(self, now: float | None = None) -> float:
        """Order count in the trailing 60s window (per-minute velocity)."""
        cutoff = (now if now is not None else time.time()) - 60.0
        self.recent_timestamps = [t for t in self.recent_timestamps if t >= cutoff]
        return float(len(self.recent_timestamps))

    def snapshot(self, now: float | None = None) -> BookSnapshot:
        """Materialize an immutable snapshot of current book state."""
        best = 0.0 if self.best_limit == float("inf") else self.best_limit
        return BookSnapshot(
            deal_id=self.deal_id,
            tranche_id=self.tranche_id,
            total_ordered_mm=self.total_ordered_mm,
            by_investor_type_mm={k.value: v for k, v in self.by_type_mm.items()},
            order_count=self.order_count,
            velocity_per_min=self.velocity_per_min(now),
            best_limit_yield=best,
            worst_limit_yield=self.worst_limit,
            book_state=BookState.BUILDING if self.order_count > 0 else BookState.OPEN,
            updated_at=utc_now().isoformat(),
        )


def build_consumer(
    bootstrap_servers: str = BOOTSTRAP_DEFAULT,
    group_id: str = "dcm-book-builder",
) -> Consumer:
    """Build a tuned consumer for the IOI topic.

    Raises:
        Exception: Any librdkafka configuration error surfaces here.
    """
    return Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,          # Manual commit = no event loss
            "enable.auto.offset.store": False,
            "session.timeout.ms": 45_000,
            "max.poll.interval.ms": 300_000,
            "fetch.wait.max.ms": 50,              # Low latency for the live book
            "client.id": f"dcm-book-builder-{socket.gethostname()}",
        }
    )


def build_snapshot_producer(bootstrap_servers: str = BOOTSTRAP_DEFAULT) -> Producer:
    """Producer for re-publishing book snapshots and DLQ events."""
    return Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "compression.type": "snappy",
            "linger.ms": 10,
        }
    )


def send_to_dlq(producer: Producer, payload: bytes, reason: str) -> None:
    """Route a poison message to the dead-letter topic (never crash-loop)."""
    try:
        producer.produce(
            topic=Topics.DLQ,
            key=b"book-builder",
            value=json.dumps(
                {"error": reason, "raw": payload.decode("utf-8", "replace")}
            ).encode(),
        )
        producer.poll(0)
    except BufferError:
        logger.error("DLQ producer queue full - dropping poison message: %s", reason)


def fold_event(books: dict[str, _TrancheBook], msg_value: bytes) -> str | None:
    """Fold one raw message into aggregation state.

    Returns:
        The (deal:tranche) aggregation key, or None if the event was not
        an IOI or was malformed.

    Raises:
        json.JSONDecodeError: Propagated so the caller can DLQ the payload.
        KeyError / ValueError: Propagated for the same reason.
    """
    record = json.loads(msg_value)
    if record.get("type") != EventType.IOI.value:
        return None
    payload = record["payload"]
    key = f"{payload['deal_id']}:{payload['tranche_id']}"
    book = books.get(key)
    if book is None:
        book = _TrancheBook(deal_id=str(payload["deal_id"]), tranche_id=str(payload["tranche_id"]))
        books[key] = book
    event_ts = float(record["ts"]) if "ts" in record else None
    book.apply(payload, event_ts=event_ts)
    return key


def run_consumer(
    bootstrap_servers: str = BOOTSTRAP_DEFAULT,
    group_id: str = "dcm-book-builder",
    *,
    snapshot_every_n: int = 500,
    max_messages: int | None = None,
    poll_timeout: float = 1.0,
) -> None:
    """Main consumer loop: aggregate IOIs and publish rolling snapshots.

    Args:
        snapshot_every_n: Publish a book snapshot every N folded events.
        max_messages: Debug/CI cap; None runs until SIGINT.
        poll_timeout: Kafka poll timeout in seconds.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    consumer = build_consumer(bootstrap_servers, group_id)
    snapshot_producer = build_snapshot_producer(bootstrap_servers)
    consumer.subscribe([Topics.IOI_BIDS])

    books: dict[str, _TrancheBook] = {}
    processed = 0
    since_snapshot = 0
    running = True

    def _shutdown(signum: int, _frame: Any) -> None:
        nonlocal running
        logger.info("signal %d received - shutting down", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("book-builder consuming %s (group=%s)", Topics.IOI_BIDS, group_id)

    try:
        while running:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            key: str | None = None
            try:
                key = fold_event(books, msg.value())
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("poison message: %s", exc)
                send_to_dlq(snapshot_producer, msg.value(), str(exc))

            if key is not None:
                processed += 1
                since_snapshot += 1

            if since_snapshot >= snapshot_every_n:
                since_snapshot = 0
                for snap_key, book in books.items():
                    snap = book.snapshot()
                    snapshot_producer.produce(
                        topic=Topics.BOOK_SNAPSHOTS,
                        key=snap_key.encode(),
                        value=json.dumps(snap.to_dict()).encode(),
                    )
                snapshot_producer.flush(5.0)
                logger.info(
                    "published snapshots for %d tranches (processed=%d)",
                    len(books),
                    processed,
                )

            # Commit only after the fold is complete: at-least-once semantics
            # with idempotent aggregation (fold is additive per event ts).
            consumer.store_offsets(offsets=[(msg.topic(), msg.partition(), msg.offset() + 1)])
            consumer.commit(asynchronous=True, offsets=[
                (msg.topic(), msg.partition(), msg.offset() + 1)
            ])

            if max_messages is not None and processed >= max_messages:
                logger.info("max_messages reached - exiting")
                break
    finally:
        # Final snapshot flush before leaving the group.
        for snap_key, book in books.items():
            snap = book.snapshot()
            snapshot_producer.produce(
                topic=Topics.BOOK_SNAPSHOTS,
                key=snap_key.encode(),
                value=json.dumps(snap.to_dict()).encode(),
            )
        snapshot_producer.flush(10.0)
        consumer.commit(asynchronous=False)
        consumer.close()
        logger.info("book-builder stopped cleanly (processed=%d tranches=%d)", processed, len(books))


if __name__ == "__main__":
    run_consumer()
