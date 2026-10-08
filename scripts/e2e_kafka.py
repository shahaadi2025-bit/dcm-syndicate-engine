"""True end-to-end verification: broker -> producer -> consumer -> agent mesh.

Orchestrates the full streaming path against a reachable Redpanda/Kafka
broker and asserts a real pricing recommendation comes out the far end:

  1. IOI producer streams simulated institutional orders  -> dcm.ioi.bids.v1
  2. book-builder consumer aggregates them into a book    -> dcm.book.snapshots.v1
  3. the freshest snapshot is read back off the broker
  4. the LangGraph agent mesh prices the tranche from it

Requires a reachable broker at DCM_KAFKA_BOOTSTRAP (default localhost:9092).
Used by the `e2e` CI job (Redpanda container) and by operators:

    python scripts/e2e_kafka.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from dcm_engine.agents.graph import (
    MAX_SPREAD_BPS,
    MIN_SPREAD_BPS,
    run_pricing_decision,
)
from dcm_engine.agents.llm import RuleBasedProvider
from dcm_engine.core.models import MacroPrint, SecondarySpreadTick, Tranche
from dcm_engine.pipeline.consumer import run_consumer
from dcm_engine.pipeline.producer import run_producer
from dcm_engine.pipeline.topics import Topics

BOOTSTRAP = os.environ.get("DCM_KAFKA_BOOTSTRAP", "localhost:9092")
N_IOIS = int(os.environ.get("E2E_IOIS", "120"))
DEAL_ID = "E2E-DEAL-2026"
TRANCHE_ID = "TRN-E2E-10Y"
SNAPSHOT_WAIT_SEC = 25.0


def _stage(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def read_latest_snapshot(bootstrap: str) -> dict[str, Any] | None:
    """Drain BOOK_SNAPSHOTS and return the freshest record for the e2e tranche."""
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"e2e-snapshot-{int(time.time() * 1000)}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([Topics.BOOK_SNAPSHOTS])
    latest: dict[str, Any] | None = None
    deadline = time.time() + SNAPSHOT_WAIT_SEC
    while time.time() < deadline:
        msg = consumer.poll(0.5)
        if msg is None:
            if latest is not None:
                break  # drained what we came for
            continue
        if msg.error() or msg.value() is None:
            continue
        try:
            record = json.loads(msg.value())
        except (TypeError, ValueError):
            continue
        if record.get("tranche_id") == TRANCHE_ID:
            latest = record
    consumer.close()
    return latest


def main() -> int:
    _stage(f"1/4 streaming IOIs -> {Topics.IOI_BIDS} @ {BOOTSTRAP}")
    producer_thread = threading.Thread(
        target=run_producer,
        kwargs={
            "bootstrap_servers": BOOTSTRAP,
            "rate_per_sec": 150.0,
            "duration_sec": 2.0,
            "deal_id": DEAL_ID,
            "tranche_id": TRANCHE_ID,
            "initial_yield": 0.0475,
            "target_mm": 1000.0,
            "seed": 11,
        },
        daemon=True,
    )
    producer_thread.start()

    _stage("2/4 consuming IOIs -> aggregating book -> publishing snapshots")
    run_consumer(
        BOOTSTRAP,
        group_id=f"e2e-book-builder-{int(time.time() * 1000)}",
        snapshot_every_n=25,
        max_messages=N_IOIS,
        poll_timeout=0.5,
    )
    producer_thread.join(timeout=10.0)

    _stage(f"3/4 reading freshest snapshot back from {Topics.BOOK_SNAPSHOTS}")
    snapshot = read_latest_snapshot(BOOTSTRAP)
    if snapshot is None:
        print("FAIL: no book snapshot found on the broker", file=sys.stderr)
        return 1
    print(json.dumps(snapshot, indent=2)[:600], flush=True)

    _stage("4/4 running the LangGraph agent mesh on the Kafka-aggregated book")
    tranche = Tranche(
        tranche_id=str(snapshot["tranche_id"]),
        currency="USD",
        target_size_mm=1000.0,
        tenor_years=10.0,
        spread_guidance_bps=115.0,
        coupon=0.0475,
        benchmark_yield=0.0265,
    )
    ticks = [SecondarySpreadTick("E2E-COMP-10Y", 10.0, 108.0, 0.0265)]
    macros = [MacroPrint(indicator="E2E_CPI_YOY", release=2.8, consensus=2.9)]
    rec = run_pricing_decision(
        tranche=tranche,
        spread_ticks=ticks,
        macro_prints=macros,
        ordered_mm=float(snapshot["total_ordered_mm"]),
        by_investor_type_mm=dict(snapshot["by_investor_type_mm"]),
        velocity_per_min=float(snapshot["velocity_per_min"]),
        provider=RuleBasedProvider(),
    )
    print(
        f"\nRECOMMENDATION {rec.tranche_id}: spread={rec.final_spread_bps:.2f}bps "
        f"yield={rec.final_yield:.4%} price={rec.final_price:.3f} "
        f"osr={rec.oversubscription_ratio:.2f}x confidence={rec.confidence:.2f}",
        flush=True,
    )
    if not (MIN_SPREAD_BPS <= rec.final_spread_bps <= MAX_SPREAD_BPS):
        print("FAIL: recommendation outside desk guardrails", file=sys.stderr)
        return 1
    print(
        "\nE2E PASS: broker -> IOIs -> book aggregation -> agent mesh -> recommendation",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
