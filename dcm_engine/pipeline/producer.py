"""High-velocity IOI producer for the Redpanda/Kafka backbone.

Simulates institutional Indications of Interest (limit orders with sizes
and yield limits) plus secondary spreads and macro prints, then streams
them with production-grade delivery semantics:

* idempotent producer (no dupes on broker retry)
* acks=all + full ISR durability
* snappy compression, 5ms linger for throughput
* keying by (deal, tranche) so per-tranche ordering is preserved
"""

from __future__ import annotations

import json
import logging
import random
import socket
import time
from typing import Any

from confluent_kafka import Producer

from dcm_engine.core.models import (
    IOI,
    InvestorType,
    MacroPrint,
    SecondarySpreadTick,
)
from dcm_engine.pipeline.topics import EventType, Topics

logger = logging.getLogger(__name__)

BOOTSTRAP_DEFAULT = "localhost:9092"


def _error_cb(err: Any) -> None:
    """Librdkafka client-level errors (auth, connectivity, throttling)."""
    logger.error("kafka client error: %s", err)


def build_producer(bootstrap_servers: str = BOOTSTRAP_DEFAULT) -> Producer:
    """Build a tuned, idempotent producer.

    Raises:
        Exception: Any librdkafka configuration error surfaces here.
    """
    return Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,           # Exactly-once per partition
            "acks": "all",                        # Wait for full ISR
            "compression.type": "snappy",
            "linger.ms": 5,                       # Micro-batch for throughput
            "batch.num.messages": 10_000,
            "queue.buffering.max.messages": 200_000,
            "queue.buffering.max.kbytes": 512_000,
            "retries": 10,
            "retry.backoff.ms": 100,
            "message.timeout.ms": 30_000,         # Fail fast on undeliverable
            "client.id": f"dcm-producer-{socket.gethostname()}-{int(time.time())}",
        },
        error_cb=_error_cb,
    )


def build_key(deal_id: str, tranche_id: str) -> bytes:
    """Partition key preserving per-tranche event ordering."""
    return f"{deal_id}:{tranche_id}".encode()


class IOISimulator:
    """Simulates a realistic institutional book.

    Anchor real-money accounts grind limits tighter over time; hedge funds
    chase momentum; sizes follow a fat-tailed distribution. The simulator
    tracks cumulative book state so the consumer-side analytics can be
    validated against known ground truth.
    """

    INVESTOR_POOL: tuple[dict[str, Any], ...] = (
        {"id": "INV-PIMCO", "type": InvestorType.REAL_MONEY, "skill": 0.85},
        {"id": "INV-BLACKROCK", "type": InvestorType.REAL_MONEY, "skill": 0.80},
        {"id": "INV-ALLIANZ", "type": InvestorType.INSURANCE, "skill": 0.75},
        {"id": "INV-ECB", "type": InvestorType.CENTRAL_BANK, "skill": 0.60},
        {"id": "INV-MILLENNIUM", "type": InvestorType.HEDGE_FUND, "skill": 0.30},
        {"id": "INV-CITADEL", "type": InvestorType.HEDGE_FUND, "skill": 0.25},
        {"id": "INV-JPM-TREASURY", "type": InvestorType.BANK_TREASURY, "skill": 0.50},
    )

    def __init__(
        self,
        deal_id: str,
        tranche_id: str,
        initial_yield: float,
        target_mm: float,
        seed: int | None = None,
    ) -> None:
        self.deal_id = deal_id
        self.tranche_id = tranche_id
        self.current_yield = initial_yield
        self.target_mm = target_mm
        self._rng = random.Random(seed)
        self.book_mm: dict[InvestorType, float] = {}

    def next_ioi(self) -> IOI:
        """Generate one IOI and update internal book tracking."""
        inv = self._rng.choice(self.INVESTOR_POOL)
        inv_type: InvestorType = inv["type"]
        skill: float = inv["skill"]

        # Skilled real money anchors bid slightly inside the current level
        # (grinding the book tighter); fast money chases with more noise.
        if inv_type == InvestorType.HEDGE_FUND:
            limit = self.current_yield + self._rng.gauss(0.0, 0.0004)
        else:
            limit = self.current_yield + self._rng.gauss(0.0, 0.0002) - skill * 0.0002

        # Fat-tailed size distribution: mostly small clips, occasional block.
        size = self._rng.choice((5, 10, 10, 15, 25, 50, 100))
        self.book_mm[inv_type] = self.book_mm.get(inv_type, 0.0) + size

        return IOI(
            investor_id=str(inv["id"]),
            investor_type=inv_type,
            deal_id=self.deal_id,
            tranche_id=self.tranche_id,
            size_mm=float(size),
            limit_yield=round(limit, 6),
        )

    @property
    def total_ordered_mm(self) -> float:
        return sum(self.book_mm.values())

    @property
    def osr(self) -> float:
        """Ground-truth oversubscription ratio for validation."""
        return self.total_ordered_mm / self.target_mm if self.target_mm > 0 else 0.0


def make_spread_tick(
    issuer: str,
    tenor_years: float,
    g_spread_bps: float,
    benchmark_yield: float,
) -> SecondarySpreadTick:
    """Build one secondary spread tick (wraps model defaults)."""
    return SecondarySpreadTick(
        issuer=issuer,
        tenor_years=tenor_years,
        g_spread_bps=g_spread_bps,
        benchmark_yield=benchmark_yield,
        source="simulator",
    )


def make_macro_print(
    indicator: str,
    release: float,
    consensus: float,
) -> MacroPrint:
    """Build one macro release event."""
    return MacroPrint(indicator=indicator, release=release, consensus=consensus)


def envelope(event_type: EventType, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a domain payload in the platform event envelope."""
    return {
        "schema_version": "1.0",
        "type": event_type.value,
        "ts": time.time(),
        "payload": payload,
    }


def delivery_callback(err: Any, msg: Any) -> None:
    """Per-message delivery report; DLQ-style logging on failure."""
    if err is not None:
        logger.error("delivery failed for key=%s: %s", msg.key(), err)
    # Successful deliveries are silent - logging 50k/s would melt the desk.


def publish_ioi(
    producer: Producer,
    simulator: IOISimulator,
    topic: str = Topics.IOI_BIDS,
) -> IOI:
    """Generate and publish one IOI event."""
    ioi = simulator.next_ioi()
    producer.produce(
        topic=topic,
        key=build_key(simulator.deal_id, simulator.tranche_id),
        value=json.dumps(envelope(EventType.IOI, ioi.to_dict())).encode(),
        on_delivery=delivery_callback,
    )
    return ioi


def publish_spread_tick(
    producer: Producer,
    tick: SecondarySpreadTick,
    topic: str = Topics.SECONDARY_SPREADS,
) -> None:
    """Publish one secondary-market spread tick."""
    producer.produce(
        topic=topic,
        key=tick.issuer.encode(),
        value=json.dumps(envelope(EventType.SPREAD_TICK, tick.to_dict())).encode(),
        on_delivery=delivery_callback,
    )


def publish_macro_print(
    producer: Producer,
    macro: MacroPrint,
    topic: str = Topics.MACRO_PRINTS,
) -> None:
    """Publish one macro data release."""
    producer.produce(
        topic=topic,
        key=macro.indicator.encode(),
        value=json.dumps(envelope(EventType.MACRO_PRINT, macro.to_dict())).encode(),
        on_delivery=delivery_callback,
    )


def run_producer(
    bootstrap_servers: str = BOOTSTRAP_DEFAULT,
    *,
    rate_per_sec: float = 500.0,
    duration_sec: float = 60.0,
    deal_id: str = "DEAL-2026-EU-001",
    tranche_id: str = "TRN-10Y-SENIOR",
    initial_yield: float = 0.0475,
    target_mm: float = 1000.0,
    seed: int | None = 42,
) -> None:
    """Run the simulator loop: publish IOIs at `rate_per_sec` for `duration_sec`.

    Designed for graceful SIGINT shutdown: flushes in-flight messages before
    exiting so no event is lost mid-stream.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    producer = build_producer(bootstrap_servers)
    sim = IOISimulator(
        deal_id=deal_id,
        tranche_id=tranche_id,
        initial_yield=initial_yield,
        target_mm=target_mm,
        seed=seed,
    )

    interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
    deadline = time.monotonic() + duration_sec
    sent = 0

    logger.info(
        "streaming IOIs -> %s @ %.0f/s for %.0fs (deal=%s tranche=%s)",
        topic if (topic := Topics.IOI_BIDS) else topic,
        rate_per_sec,
        duration_sec,
        deal_id,
        tranche_id,
    )

    try:
        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            publish_ioi(producer, sim)
            # Occasionally sprinkle a spread tick and macro print.
            if sent % 200 == 0:
                publish_spread_tick(
                    producer,
                    make_spread_tick("BUND-10Y-COMP", 10.0, 85.0, 0.0265),
                )
            if sent % 1000 == 0:
                publish_macro_print(producer, make_macro_print("EU_HICP_YOY", 2.4, 2.3))
            sent += 1
            producer.poll(0)  # Serve delivery-report callbacks
            elapsed = time.monotonic() - loop_start
            if interval > elapsed:
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        logger.info("shutdown signal received - flushing")
    finally:
        remaining = producer.flush(10.0)
        if remaining > 0:
            logger.error("%d messages failed to flush", remaining)
        else:
            logger.info(
                "flush complete: sent=%d ordered_mm=%.0f osr=%.2fx",
                sent,
                sim.total_ordered_mm,
                sim.osr,
            )


if __name__ == "__main__":
    run_producer()
