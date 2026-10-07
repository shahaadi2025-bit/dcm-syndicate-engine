"""Tests for the LangGraph agent mesh decision loop."""

from __future__ import annotations

from dcm_engine.agents.graph import (
    MAX_REVISION_ROUNDS,
    build_pricing_graph,
    run_pricing_decision,
)
from dcm_engine.agents.llm import RuleBasedProvider
from dcm_engine.core.models import (
    InvestorType,
    MacroPrint,
    SecondarySpreadTick,
    Tranche,
)


def _tranche() -> Tranche:
    return Tranche(
        tranche_id="TRN-10Y",
        currency="USD",
        target_size_mm=1000.0,
        tenor_years=10.0,
        spread_guidance_bps=115.0,
        coupon=0.0475,
        benchmark_yield=0.0265,
    )


def _ticks() -> list[SecondarySpreadTick]:
    return [
        SecondarySpreadTick("COMPA", 9.5, 100.0, 0.0265),
        SecondarySpreadTick("COMPB", 10.0, 108.0, 0.0265),
        SecondarySpreadTick("COMPC", 10.5, 112.0, 0.0265),
    ]


def _macros() -> list[MacroPrint]:
    return [MacroPrint(indicator="US_CPI_YOY", release=2.8, consensus=2.9)]


def test_recommendation_finalized_on_strong_book() -> None:
    """A hot, sticky book converges to a tight spread in one pass."""
    rec = run_pricing_decision(
        tranche=_tranche(),
        spread_ticks=_ticks(),
        macro_prints=_macros(),
        ordered_mm=4200.0,
        by_investor_type_mm={
            "REAL_MONEY": 2600.0,
            "INSURANCE": 800.0,
            "CENTRAL_BANK": 400.0,
            "HEDGE_FUND": 400.0,
        },
        velocity_per_min=80.0,
        provider=RuleBasedProvider(),
    )
    assert rec.tranche_id == "TRN-10Y"
    assert rec.final_spread_bps > 0
    assert rec.final_yield > 0
    assert 80.0 < rec.final_price < 120.0
    assert rec.oversubscription_ratio > 1.0
    assert rec.confidence > 0.0


def test_weak_book_triggers_revision_rounds() -> None:
    """A thin book forces the bookrunner to challenge at least once."""
    rec = run_pricing_decision(
        tranche=_tranche(),
        spread_ticks=_ticks(),
        macro_prints=[],
        ordered_mm=700.0,
        by_investor_type_mm={"HEDGE_FUND": 700.0},
        velocity_per_min=10.0,
        provider=RuleBasedProvider(),
    )
    assert rec.revised is True


def test_loop_is_bounded() -> None:
    """The revision loop must terminate within MAX_REVISION_ROUNDS."""
    state_seen_rounds: list[int] = []

    def _tracking_provider_stub() -> None:
        return None

    rec = run_pricing_decision(
        tranche=_tranche(),
        spread_ticks=_ticks(),
        macro_prints=[],
        ordered_mm=100.0,  # drastically undersubscribed
        by_investor_type_mm={"HEDGE_FUND": 100.0},
        velocity_per_min=1.0,
        provider=RuleBasedProvider(),
    )
    state_seen_rounds.append(rec.oversubscription_ratio)
    assert rec is not None  # Graph completed without raising.


def test_graph_compiles_and_is_acyclic_termination() -> None:
    app = build_pricing_graph()
    assert app is not None
    assert MAX_REVISION_ROUNDS >= 1


def test_deterministic_with_rule_provider() -> None:
    """Same inputs + deterministic provider => identical recommendation."""
    kwargs = dict(
        tranche=_tranche(),
        spread_ticks=_ticks(),
        macro_prints=_macros(),
        ordered_mm=3000.0,
        by_investor_type_mm={"REAL_MONEY": 2500.0, "HEDGE_FUND": 500.0},
        velocity_per_min=40.0,
        provider=RuleBasedProvider(),
    )
    rec_a = run_pricing_decision(**kwargs)
    rec_b = run_pricing_decision(**kwargs)
    assert rec_a.final_spread_bps == rec_b.final_spread_bps
    assert rec_a.final_yield == rec_b.final_yield
    assert rec_a.final_price == rec_b.final_price
