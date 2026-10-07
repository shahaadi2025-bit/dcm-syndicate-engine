"""Tests for the streaming pipeline (simulation + aggregation logic)."""

from __future__ import annotations

import json

from dcm_engine.core.models import InvestorType
from dcm_engine.pipeline.consumer import _TrancheBook, fold_event
from dcm_engine.pipeline.producer import IOISimulator, envelope
from dcm_engine.pipeline.topics import EventType


class TestIOISimulator:
    def test_simulator_tracks_book(self) -> None:
        sim = IOISimulator(
            deal_id="D1",
            tranche_id="T1",
            initial_yield=0.0475,
            target_mm=1000.0,
            seed=7,
        )
        for _ in range(50):
            ioi = sim.next_ioi()
            assert ioi.size_mm > 0
            assert ioi.limit_yield > 0
        assert sim.total_ordered_mm > 0
        assert sim.osr > 0

    def test_simulator_deterministic_with_seed(self) -> None:
        sim_a = IOISimulator("D", "T", 0.0475, 1000.0, seed=99)
        sim_b = IOISimulator("D", "T", 0.0475, 1000.0, seed=99)
        iois_a = [sim_a.next_ioi() for _ in range(20)]
        iois_b = [sim_b.next_ioi() for _ in range(20)]
        assert [i.limit_yield for i in iois_a] == [i.limit_yield for i in iois_b]


class TestEnvelope:
    def test_envelope_wraps_payload(self) -> None:
        env = envelope(EventType.IOI, {"size_mm": 10.0})
        assert env["type"] == "IOI"
        assert env["payload"]["size_mm"] == 10.0
        assert "ts" in env


class TestFoldEvent:
    def _ioi_event(self) -> bytes:
        return json.dumps(
            envelope(
                EventType.IOI,
                {
                    "investor_id": "INV-X",
                    "investor_type": "REAL_MONEY",
                    "deal_id": "D1",
                    "tranche_id": "T1",
                    "size_mm": 25.0,
                    "limit_yield": 0.048,
                },
            )
        ).encode()

    def test_fold_accumulates(self) -> None:
        books: dict[str, _TrancheBook] = {}
        assert fold_event(books, self._ioi_event()) == "D1:T1"
        fold_event(books, self._ioi_event())
        book = books["D1:T1"]
        assert book.total_ordered_mm == 50.0
        assert book.order_count == 2
        assert book.by_type_mm[InvestorType.REAL_MONEY] == 50.0

    def test_fold_tracks_best_and_worst_limit(self) -> None:
        books: dict[str, _TrancheBook] = {}
        fold_event(books, self._ioi_event())
        book = books["D1:T1"]
        assert book.best_limit == 0.048
        assert book.worst_limit == 0.048

    def test_non_ioi_event_ignored(self) -> None:
        books: dict[str, _TrancheBook] = {}
        other = json.dumps(envelope(EventType.SPREAD_TICK, {"issuer": "X"})).encode()
        assert fold_event(books, other) is None
        assert len(books) == 0

    def test_poison_message_raises_for_dlq(self) -> None:
        books: dict[str, _TrancheBook] = {}
        import pytest

        with pytest.raises((KeyError, ValueError, json.JSONDecodeError)):
            fold_event(books, b"{ not valid json")
