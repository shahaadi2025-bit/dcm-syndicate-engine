"""Topic registry + event schemas for the DCM streaming backbone.

Redpanda/Kafka topics are the source of truth for inter-component
communication. Centralizing topic names and schema versions prevents
cross-team drift (the #1 cause of silent stream breakage).
"""

from __future__ import annotations

from enum import StrEnum


class Topics(StrEnum):
    """Canonical topic names for the syndicate platform."""

    SECONDARY_SPREADS = "dcm.secondary.spreads.v1"
    MACRO_PRINTS = "dcm.macro.prints.v1"
    IOI_BIDS = "dcm.ioi.bids.v1"
    BOOK_SNAPSHOTS = "dcm.book.snapshots.v1"
    PRICING_RECOMMENDATIONS = "dcm.pricing.recommendations.v1"
    PRICING_DECISIONS = "dcm.pricing.decisions.v1"
    DLQ = "dcm.dead-letter.v1"


class EventType(StrEnum):
    """Event type discriminators carried in the envelope `type` field."""

    SPREAD_TICK = "SPREAD_TICK"
    MACRO_PRINT = "MACRO_PRINT"
    IOI = "IOI"
    BOOK_SNAPSHOT = "BOOK_SNAPSHOT"
    RECOMMENDATION = "RECOMMENDATION"
    DECISION = "DECISION"


SCHEMA_VERSION = "1.0"

# Per-topic production knobs (partitions sized for ~50k msg/s burst).
TOPIC_CONFIGS: dict[str, dict[str, int | str]] = {
    Topics.IOI_BIDS: {"num_partitions": 12, "replication_factor": 3},
    Topics.SECONDARY_SPREADS: {"num_partitions": 6, "replication_factor": 3},
    Topics.MACRO_PRINTS: {"num_partitions": 3, "replication_factor": 3},
    Topics.BOOK_SNAPSHOTS: {"num_partitions": 6, "replication_factor": 3},
    Topics.PRICING_RECOMMENDATIONS: {"num_partitions": 3, "replication_factor": 3},
    Topics.PRICING_DECISIONS: {"num_partitions": 3, "replication_factor": 3},
    Topics.DLQ: {"num_partitions": 3, "replication_factor": 3},
}
