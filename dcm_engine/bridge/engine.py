"""Live engine state for the API bridge.

`LiveBookEngine` owns the mutable deal state the pricing service exposes:

* a rolling institutional order book, either **simulated** (deterministic
  seeded generator, default) or **Kafka-sourced** (book snapshots consumed
  from the streaming pipeline), and
* a cached agent-mesh recommendation, refreshed at most every
  `RECOMMENDATION_TTL_SECONDS` so the LangGraph decision loop is never run
  more than once per TTL per tranche (cost + determinism control).

This class is transport-agnostic: FastAPI merely renders its state.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dcm_engine.agents.graph import run_pricing_decision
from dcm_engine.agents.llm import load_provider_from_env
from dcm_engine.core.models import (
    MacroPrint,
    SecondarySpreadTick,
    Tranche,
    TrancheRecommendation,
)
from dcm_engine.ml.forecaster import ModelArtifact


def _load_trained_demand_model() -> ModelArtifact | None:
    """Auto-load a CSV-trained artifact if present (DCM_MODEL_DIR or default)."""
    import os as _os
    from pathlib import Path as _Path

    env_dir = _os.environ.get("DCM_MODEL_DIR")
    candidates = [
        _Path(env_dir) / "demand_model.json" if env_dir else None,
        _Path(__file__).resolve().parents[1] / "ml" / "artifacts" / "demand_model.json",
    ]
    for cand in candidates:
        if cand and cand.exists():
            try:
                art = ModelArtifact.from_json(cand.read_text(encoding="utf-8"))
                logger.info("loaded trained demand model from %s", cand)
                return art
            except (ValueError, KeyError, OSError) as load_err:
                logger.warning("failed to load demand artifact %s: %s", cand, load_err)
    return None

logger = logging.getLogger(__name__)

RECOMMENDATION_TTL_SECONDS = 30.0
KAFKA_SNAPSHOT_TTL_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Optional Kafka snapshot ingestion (enabled with DCM_KAFKA_BOOTSTRAP)
# ---------------------------------------------------------------------------


def _try_kafka_snapshots() -> dict[str, Any] | None:
    """Consume the latest book snapshot per tranche, or None if not configured.

    Requires confluent-kafka and DCM_KAFKA_BOOTSTRAP. Uses a short-lived
    consumer that fast-forwards to the end of each partition so we always
    see the freshest snapshot without maintaining group state.
    """

    bootstrap = os.environ.get("DCM_KAFKA_BOOTSTRAP", "").strip()
    if not bootstrap:
        return None
    try:
        from confluent_kafka import Consumer, TopicPartition

        c = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": f"dcm-bridge-{int(time.time())}",  # unique: no state
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
            }
        )
        import json as _json

        from dcm_engine.pipeline.topics import Topics

        meta = c.list_topics(Topics.BOOK_SNAPSHOTS, timeout=2.0)
        parts = [
            TopicPartition(Topics.BOOK_SNAPSHOTS, p.id)
            for p in meta.topics[Topics.BOOK_SNAPSHOTS].partitions.values()
        ]
        latest: dict[str, dict[str, Any]] = {}
        snaps: list[tuple[str, dict[str, Any]]] = []

        def _collect(err: Any, msg: Any) -> None:
            if err is None and msg is not None and msg.value():
                try:
                    d = _json.loads(msg.value())
                    snaps.append((d["tranche_id"], d))
                except (KeyError, TypeError, ValueError, _json.JSONDecodeError) as parse_err:
                    logger.debug("skipping malformed snapshot: %s", parse_err)

        c.assign(parts)
        deadline = time.time() + 1.5
        while time.time() < deadline:
            msg = c.poll(0.2)
            if msg is not None:
                _collect(msg.error(), msg)
        c.close()
        for tid, d in snaps:
            latest[tid] = d
        if not latest:
            return None
        # Bridge serves a single deal; take any tranche's snapshot.
        return next(iter(latest.values()))
    except Exception as exc:  # Kafka down / topic missing -> fall back
        logger.warning("kafka snapshot fetch failed, using simulator: %s", exc)
        return None


# ruff: noqa: BLE001 - module boundary: any Kafka/serde failure must degrade to the simulator


# ---------------------------------------------------------------------------
# Deterministic simulated book
# ---------------------------------------------------------------------------


@dataclass
class DemoEngineState:
    """Mutable deal state used when no Kafka feed is attached."""

    tranche: Tranche
    ticks: list[SecondarySpreadTick]
    macros: list[MacroPrint]
    book: dict[str, dict[str, Any]] = field(default_factory=dict)
    velocity: float = 0.0
    rng_state: int = 424242

    def step(self) -> None:
        """Advance the simulated market by one tick (deterministic-ish)."""
        state = self.rng_state
        rnd = (state * 1103515245 + 12345) % 2147483648
        self.rng_state = rnd
        u = rnd / 2147483648.0

        n_new = 1 + int(u * 3)
        pool = (
            ("PIMCO", "REAL_MONEY"),
            ("BlackRock", "REAL_MONEY"),
            ("Norges", "CENTRAL_BANK"),
            ("Allianz", "INSURANCE"),
            ("Millennium", "HEDGE_FUND"),
            ("Citadel", "HEDGE_FUND"),
            ("JPM Treasury", "BANK_TREASURY"),
        )
        for i in range(n_new):
            name, inv_type = pool[(int(rnd >> (3 + i * 5)) % len(pool))]
            u2 = ((rnd >> (7 + i * 3)) % 1000) / 1000.0
            size = (5, 10, 15, 25, 50, 75, 100)[int(u2 * 7) % 7]
            entry = self.book.get(name)
            prev = entry["size_mm"] if entry else 0.0
            self.book[name] = {
                "investor_id": name,
                "investor_type": inv_type,
                "size_mm": prev + size,
                "limit_yield": round(self.tranche.benchmark_yield + 0.0100 - u2 * 0.004, 6),
            }
        self.velocity = self.velocity * 0.7 + n_new * 45.0


def build_default_state() -> DemoEngineState:
    """Seed the demo deal used when the bridge runs standalone."""
    tranche = Tranche(
        tranche_id="ACME-10Y-SENIOR",
        currency="USD",
        target_size_mm=1000.0,
        tenor_years=10.0,
        spread_guidance_bps=115.0,
        coupon=0.0475,
        benchmark_yield=0.0265,
    )
    ticks = [
        SecondarySpreadTick("ACME-9.5Y-2035", 9.5, 104.0, 0.0265),
        SecondarySpreadTick("ACME-10Y-2036", 10.0, 110.0, 0.0265),
        SecondarySpreadTick("PEER-10Y-2036", 10.0, 118.0, 0.0265),
    ]
    macros = [MacroPrint(indicator="US_CPI_YOY", release=2.8, consensus=2.9)]
    return DemoEngineState(tranche=tranche, ticks=ticks, macros=macros)


# ---------------------------------------------------------------------------
# The bridge engine
# ---------------------------------------------------------------------------


class LiveBookEngine:
    """Owns deal state; computes and caches agent recommendations."""

    def __init__(self) -> None:
        self.state = build_default_state()
        self._lock = threading.Lock()
        self._last_recommendation: TrancheRecommendation | None = None
        self._last_computed_at: float = 0.0
        self._demand_artifact = _load_trained_demand_model()
        self._kafka_snapshot: dict[str, Any] | None = None
        self._kafka_fetch_ts: float = 0.0

    def _maybe_refresh_kafka_snapshot(self) -> None:
        """TTL-gated pull of the freshest broker book snapshot.

        No-op unless DCM_KAFKA_BOOTSTRAP is set. Rate-limited so the
        short-lived consumer never runs more than once per TTL window.
        Callers hold the engine lock: the ~2s Kafka probe pauses WS ticks
        briefly, at most once per window.
        """
        if not os.environ.get("DCM_KAFKA_BOOTSTRAP", "").strip():
            return
        now = time.time()
        if now - self._kafka_fetch_ts < KAFKA_SNAPSHOT_TTL_SECONDS:
            return
        self._kafka_fetch_ts = now
        snap = _try_kafka_snapshots()
        if snap is not None:
            self._kafka_snapshot = snap

    def _apply_kafka_snapshot(self) -> bool:
        """Overlay the latest Kafka book snapshot onto served state.

        Returns True when served data comes from the streaming pipeline
        (book_source='kafka'); False keeps the deterministic simulator.
        """
        snap = self._kafka_snapshot
        if not snap:
            return False
        if str(snap.get("tranche_id")) != self.state.tranche.tranche_id:
            return False
        by_type: dict[str, Any] = snap.get("by_investor_type_mm") or {}
        bench = self.state.tranche.benchmark_yield
        best = float(snap.get("best_limit_yield") or 0.0)
        limit = best if best > 0 else bench + 0.0100
        self.state.book = {
            str(inv_type): {
                "investor_id": f"KAFKA-{inv_type}",
                "investor_type": str(inv_type),
                "size_mm": float(mm),
                "limit_yield": round(limit, 6),
            }
            for inv_type, mm in by_type.items()
            if float(mm) > 0
        }
        self.state.velocity = float(snap.get("velocity_per_min") or 0.0)
        return True

    @property
    def model_info(self) -> dict[str, Any]:
        """Provenance of the demand model backing predictions."""
        art = self._demand_artifact
        if art is None:
            return {"source": "heuristic", "model_type": None, "n_samples": 0}
        return {
            "source": "csv-trained ridge",
            "model_type": art.model_type,
            "n_samples": art.n_samples,
        }

    # -- aggregation -------------------------------------------------------

    def _aggregate(self) -> dict[str, Any]:
        book = self.state.book
        total = sum(v["size_mm"] for v in book.values())
        by_type: dict[str, float] = {}
        for v in book.values():
            t = str(v["investor_type"])
            by_type[t] = by_type.get(t, 0.0) + float(v["size_mm"])
        return {
            "tranche": self.state.tranche,
            "ordered_mm": total,
            "by_investor_type_mm": by_type,
            "velocity_per_min": self.state.velocity,
        }

    def _should_recompute(self, now: float) -> bool:
        if self._last_recommendation is None:
            return True
        return (now - self._last_computed_at) >= RECOMMENDATION_TTL_SECONDS

    # -- public API ----------------------------------------------------------

    def tick_and_recommend(self) -> dict[str, Any]:
        """Advance the sim one tick; refresh the agent recommendation on TTL expiry.

        Returns a JSON-safe payload consumed by both REST and WS handlers.
        """
        with self._lock:
            self._maybe_refresh_kafka_snapshot()
            kafka_applied = self._apply_kafka_snapshot()
            if not kafka_applied:
                self.state.step()
            agg = self._aggregate()
            now = time.time()
            if self._should_recompute(now):
                try:
                    self._last_recommendation = run_pricing_decision(
                        tranche=agg["tranche"],
                        spread_ticks=self.state.ticks,
                        macro_prints=self.state.macros,
                        ordered_mm=agg["ordered_mm"],
                        by_investor_type_mm=agg["by_investor_type_mm"],
                        velocity_per_min=agg["velocity_per_min"],
                        provider=load_provider_from_env(),
                    )
                    self._last_computed_at = now
                except Exception:
                    logger.exception("agent mesh failed; serving stale recommendation")

            rec = self._last_recommendation
            return {
                "tranche": agg["tranche"].to_dict(),
                "book": list(self.state.book.values()),
                "ordered_mm": round(agg["ordered_mm"], 1),
                "target_mm": agg["tranche"].target_size_mm,
                "velocity_per_min": round(agg["velocity_per_min"], 1),
                "osr": round(agg["ordered_mm"] / agg["tranche"].target_size_mm, 3)
                if agg["tranche"].target_size_mm > 0
                else 0.0,
                "recommendation": rec.to_dict() if rec else None,
                "server_time": now,
                "book_source": "kafka" if kafka_applied else "simulated",
            }

    def current(self) -> dict[str, Any]:
        """Snapshot without advancing the sim (for REST GETs)."""
        with self._lock:
            kafka_applied = self._apply_kafka_snapshot()
            agg = self._aggregate()
            rec = self._last_recommendation
            return {
                "tranche": agg["tranche"].to_dict(),
                "book": list(self.state.book.values()),
                "ordered_mm": round(agg["ordered_mm"], 1),
                "target_mm": agg["tranche"].target_size_mm,
                "velocity_per_min": round(agg["velocity_per_min"], 1),
                "osr": round(agg["ordered_mm"] / agg["tranche"].target_size_mm, 3)
                if agg["tranche"].target_size_mm > 0
                else 0.0,
                "recommendation": rec.to_dict() if rec else None,
                "server_time": time.time(),
                "book_source": "kafka" if kafka_applied else "simulated",
            }


# Packaged singleton (FastAPI app imports this).
_ENGINE: LiveBookEngine | None = None


def get_engine() -> LiveBookEngine:
    """Return the process-wide engine (created once)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LiveBookEngine()
    return _ENGINE
