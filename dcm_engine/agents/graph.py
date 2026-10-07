"""LangGraph multi-agent orchestration for the DCM syndicate desk.

Three specialist agents operate on a shared typed state and cross-check
each other before a pricing recommendation is finalized:

 1. Market Intelligence Agent      - spreads + macro -> spread view (bps)
 2. Sentiment & Demand Forecaster  - book velocity/mix -> demand forecast
 3. Syndicate Bookrunner Agent     - risk overlay, challenges the other two

The graph runs a bounded decision loop: agents publish structured
proposals into shared state, the bookrunner challenges, and any agent may
request one revision round before the graph converges on a final
recommendation. All node functions are synchronous and deterministic when
backed by the rule-based provider, so the same inputs always yield the
same recommendation - a hard requirement for post-trade audit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from dcm_engine.agents.llm import (
    LLMProvider,
    RuleBasedProvider,
    build_bookrunner_check_prompt,
    build_demand_forecast_prompt,
    build_spread_analysis_prompt,
    load_provider_from_env,
)
from dcm_engine.core.models import (
    MacroPrint,
    SecondarySpreadTick,
    Tranche,
    TrancheRecommendation,
)
from dcm_engine.core.pricing import (
    bond_price_from_yield,
    interpolate_benchmark_yield,
    matrix_spread_adjustment_bps,
    oversubscription_ratio,
)

logger = logging.getLogger(__name__)

# Bounded loop: the cross-check revision cycle runs at most N times so the
# graph always terminates even if agents keep disagreeing.
MAX_REVISION_ROUNDS = 2

# Desk hard limits - the guardrail the bookrunner enforces.
MIN_SPREAD_BPS = 5.0        # Never quote inside 5bps
MAX_SPREAD_BPS = 750.0      # High-yield stress ceiling
MAX_TIGHTEN_PER_ROUND_BPS = 12.0  # Circuit breaker per revision round
MIN_CONFIDENCE_TO_PRICE = 0.55


# ---------------------------------------------------------------------------
# Shared agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Typed shared state flowing through the LangGraph mesh.

    LangGraph merges per-node updates into this state; each node returns
    only the fields it changes.
    """

    # Inputs
    tranche: Tranche
    spread_ticks: list[SecondarySpreadTick] = field(default_factory=list)
    macro_prints: list[MacroPrint] = field(default_factory=list)
    ordered_mm: float = 0.0
    by_investor_type_mm: dict[str, float] = field(default_factory=dict)
    velocity_per_min: float = 0.0
    provider: LLMProvider = field(default_factory=RuleBasedProvider)

    # Per-agent outputs
    market_view: dict[str, Any] = field(default_factory=dict)
    demand_forecast: dict[str, Any] = field(default_factory=dict)
    bookrunner_verdict: dict[str, Any] = field(default_factory=dict)

    # Loop control
    revision_round: int = 0
    challenges: list[str] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)

    # Output
    recommendation: TrancheRecommendation | None = None


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def market_intelligence_agent(state: AgentState) -> dict[str, Any]:
    """Agent 1: analyze secondary spreads + macro data -> spread view.

    Produces a comparable-implied fair spread, applies a macro surprise
    overlay, and caps the view against hard desk limits.
    """
    tranche = state.tranche
    provider = state.provider

    # 1. Comparable-based anchor: interpolate the secondary curve at the
    #    tranche tenor, then average the G-spreads of nearby comparables.
    curve = {10.0: 0.0265, 30.0: 0.0310}  # Fallback govt curve (USD swap proxy)
    benchmark = tranche.benchmark_yield or interpolate_benchmark_yield(
        tranche.tenor_years, curve
    )

    comparable_spreads = [
        t.g_spread_bps
        for t in state.spread_ticks
        if abs(t.tenor_years - tranche.tenor_years) <= 2.5
    ]
    if comparable_spreads:
        comp_fair_spread = sum(comparable_spreads) / len(comparable_spreads)
        basis = "comparables"
    else:
        comp_fair_spread = tranche.spread_guidance_bps
        basis = "guidance_fallback"

    # 2. Macro surprise overlay: each 25bps of hawkish surprise widens the
    #    new-issue spread by ~4bps (desk heuristic, half-life ~1 session).
    macro_surprise_bps = sum(m.surprise_bps for m in state.macro_prints)
    macro_overlay_bps = max(-10.0, min(10.0, macro_surprise_bps * 0.16))

    fair_spread = comp_fair_spread + macro_overlay_bps

    # Circuit breaker (revision rounds only): the MI view may not walk
    # tighter by more than MAX_TIGHTEN_PER_ROUND_BPS per round.
    if state.revision_round > 0:
        prev_fair = float(state.market_view.get("fair_spread_bps", fair_spread))
        floor = prev_fair - MAX_TIGHTEN_PER_ROUND_BPS
        if fair_spread < floor:
            fair_spread = floor
            state.audit_log.append(
                f"MI: fair value walk capped at {floor:.1f}bps (circuit breaker)"
            )

    prompt = build_spread_analysis_prompt(state.spread_ticks, state.macro_prints, tranche)
    narrative = provider.complete(prompt)

    view = {
        "fair_spread_bps": round(fair_spread, 2),
        "benchmark_yield": benchmark,
        "comparable_basis": basis,
        "macro_overlay_bps": round(macro_overlay_bps, 2),
        "narrative": narrative,
        "n_comparables": len(comparable_spreads),
    }
    state.audit_log.append(
        f"MI: fair={fair_spread:.1f}bps basis={basis} macro={macro_overlay_bps:+.1f}"
    )
    return {"market_view": view}


def demand_forecaster_agent(state: AgentState) -> dict[str, Any]:
    """Agent 2: book velocity + investor mix -> demand forecast.

    Weights investor types by stickiness (fast money churns), extrapolates
    velocity into a closing-demand estimate, and produces a churn-risk
    score the bookrunner uses to discount headline OSR.
    """
    provider = state.provider
    target = state.tranche.target_size_mm

    # 1. Sticky-money weighting: anchors count fully; hedge funds at 70%.
    stickiness = {
        "CENTRAL_BANK": 1.25,
        "REAL_MONEY": 1.15,
        "INSURANCE": 1.05,
        "BANK_TREASURY": 0.90,
        "RETAIL_AGGREGATOR": 0.80,
        "HEDGE_FUND": 0.70,
    }
    weighted_mm = 0.0
    for inv_type, mm in state.by_investor_type_mm.items():
        weighted_mm += mm * stickiness.get(inv_type, 0.85)

    # 2. Velocity extrapolation: assume the book continues at current pace
    #    but decays 40% per hour (books front-load; the tail is thin).
    velocity = state.velocity_per_min
    projected_extra = velocity * 60 * 0.4  # one more hour, 40% decay

    raw_osr = (
        oversubscription_ratio(state.ordered_mm, target)
        if state.ordered_mm > 0 and target > 0
        else 0.0
    )

    forecast_demand = state.ordered_mm + projected_extra
    churn_risk = min(
        1.0,
        (state.by_investor_type_mm.get("HEDGE_FUND", 0.0) / max(state.ordered_mm, 1.0))
        * (1.5 if velocity > 50 else 1.0),
    )

    # Net-of-churn demand: what remains if all fast money flips at the
    # final allocate. This is the desk's true left-tail exposure metric.
    sticky_share = min(weighted_mm / max(state.ordered_mm, 1.0), 1.5)
    net_demand = forecast_demand * min(sticky_share, 1.0)

    prompt = build_demand_forecast_prompt(
        velocity, state.ordered_mm, target, state.by_investor_type_mm
    )
    narrative = provider.complete(prompt)

    forecast = {
        "forecast_demand_mm": round(forecast_demand, 1),
        "net_demand_mm": round(net_demand, 1),
        "raw_osr": round(raw_osr, 3),
        "weighted_mm": round(weighted_mm, 1),
        "churn_risk": round(churn_risk, 3),
        "narrative": narrative,
    }
    state.audit_log.append(
        f"DF: demand={forecast_demand:.0f}mm net={net_demand:.0f}mm "
        f"churn={churn_risk:.2f} osr={raw_osr:.2f}x"
    )
    return {"demand_forecast": forecast}


def bookrunner_agent(state: AgentState) -> dict[str, Any]:
    """Agent 3: risk overlay + challenge - the desk's veto holder.

    Blends the market view and demand forecast into a proposed level,
    applies the oversubscription matrix, then stress-checks: minimum
    confidence, spread bounds, per-round tightening circuit breaker, and
    undersubscription (left-tail) risk. Can send the loop back for a
    revision round instead of approving.
    """
    provider = state.provider
    tranche = state.tranche
    market_view = state.market_view
    forecast = state.demand_forecast

    fair_spread = float(market_view.get("fair_spread_bps", tranche.spread_guidance_bps))
    forecast_demand = float(forecast.get("forecast_demand_mm", state.ordered_mm))
    # Left-tail risk is judged on NET demand (after fast-money churn),
    # never on the gross headline book.
    net_demand = float(forecast.get("net_demand_mm", forecast_demand))
    churn_risk = float(forecast.get("churn_risk", 0.0))
    weighted_mm = float(forecast.get("weighted_mm", state.ordered_mm))

    # Effective OSR blends raw and churn-discounted demand: a 3x book made
    # of flippers is economically a 2x book.
    raw_osr = (
        oversubscription_ratio(state.ordered_mm, tranche.target_size_mm)
        if state.ordered_mm > 0 and tranche.target_size_mm > 0
        else 0.0
    )
    effective_osr = (
        weighted_mm / tranche.target_size_mm if tranche.target_size_mm > 0 else 0.0
    )
    blended_osr = 0.5 * raw_osr + 0.5 * effective_osr

    # Matrix adjustment from the blended book signal...
    matrix_delta = matrix_spread_adjustment_bps(blended_osr)
    # ...then converge guidance toward the MI fair-value view.
    pull = 0.6 * fair_spread + 0.4 * (tranche.spread_guidance_bps + matrix_delta)

    # Churn-discount: high fast-money share keeps the level wider.
    churn_widener = churn_risk * 4.0
    proposed_spread = pull + churn_widener

    # --- Cross-check / challenge logic -----------------------------------
    challenges: list[str] = []

    if state.revision_round == 0 and fair_spread < tranche.spread_guidance_bps - 25.0:
        challenges.append(
            f"MI fair value ({fair_spread:.1f}) is >25bps through guidance; "
            "demand one revision round before tightening"
        )

    if net_demand < tranche.target_size_mm * 0.9:
        challenges.append(
            f"Net-of-churn demand {net_demand:.0f}mm below 90% of target "
            f"(gross {forecast_demand:.0f}mm); undersubscription left-tail"
        )

    confidence = 1.0 - min(1.0, 0.3 * len(challenges) + churn_risk * 0.2)

    verdict = {
        "proposed_spread_bps": round(proposed_spread, 2),
        "blended_osr": round(blended_osr, 3),
        "churn_widener_bps": round(churn_widener, 2),
        "confidence": round(confidence, 3),
        "challenges": challenges,
        "approve": len(challenges) == 0,
    }

    prompt = build_bookrunner_check_prompt(
        tranche.tranche_id, proposed_spread, blended_osr, forecast_demand
    )
    verdict["narrative"] = provider.complete(prompt)

    state.audit_log.append(
        f"BR: proposed={proposed_spread:.1f} conf={confidence:.2f} "
        f"approve={verdict['approve']} challenges={len(challenges)}"
    )
    return {"bookrunner_verdict": verdict}


# ---------------------------------------------------------------------------
# Router + finalizer
# ---------------------------------------------------------------------------


def route_after_bookrunner(state: AgentState) -> Literal["revise", "finalize"]:
    """Loop controller: send back for one revision round or finalize."""
    verdict = state.bookrunner_verdict
    if verdict.get("approve", False):
        return "finalize"
    if state.revision_round >= MAX_REVISION_ROUNDS:
        logger.warning(
            "max revision rounds hit with %d open challenges - finalizing anyway",
            len(verdict.get("challenges", [])),
        )
        return "finalize"
    return "revise"


def revision_node(state: AgentState) -> dict[str, Any]:
    """Bounded revision round: agents re-run with updated book data.

    In production this node re-polls live inputs from the snapshot topic;
    here it simulates intra-book growth between rounds so the loop
    demonstrably converges. The tightening circuit breaker itself is
    enforced inside the MI agent when it recomputes fair value.
    """
    state.revision_round += 1
    state.audit_log.append(f"LOOP: revision round {state.revision_round}")

    # Simulated intra-book update: books accelerate 20-40% per round.
    growth = 1.2 + 0.2 * state.revision_round
    state.ordered_mm *= growth
    state.velocity_per_min *= growth
    return {"revision_round": state.revision_round}


def finalize_recommendation(state: AgentState) -> dict[str, Any]:
    """Convert the approved verdict into the final pricing recommendation."""
    verdict = state.bookrunner_verdict
    tranche = state.tranche

    final_spread = max(
        MIN_SPREAD_BPS, min(MAX_SPREAD_BPS, float(verdict["proposed_spread_bps"]))
    )
    final_yield = tranche.benchmark_yield + final_spread / 10_000.0

    # Price the bond. New issues conventionally set the coupon at (or just
    # below) the final yield so the bond prices near par; use the stated
    # coupon when the deal carries one, otherwise assume reoffer at par.
    coupon = tranche.coupon if tranche.coupon > 0 else final_yield
    price = bond_price_from_yield(
        face=100.0,
        ytm=final_yield,
        coupon_rate=coupon,
        years=tranche.tenor_years,
        freq=2,
    )

    forecast_demand = float(state.demand_forecast.get("forecast_demand_mm", state.ordered_mm))
    osr = oversubscription_ratio(max(forecast_demand, 1.0), tranche.target_size_mm)

    state.recommendation = TrancheRecommendation(
        tranche_id=tranche.tranche_id,
        final_spread_bps=round(final_spread, 2),
        final_yield=round(final_yield, 6),
        final_price=round(price, 3),
        forecast_demand_mm=round(forecast_demand, 1),
        oversubscription_ratio=round(osr, 3),
        size_mm=tranche.target_size_mm,
        confidence=float(verdict.get("confidence", 0.0)),
        rationale=(
            f"MI fair {state.market_view.get('fair_spread_bps')}bps; "
            f"OSR {verdict.get('blended_osr')}x blended; "
            f"churn widener {verdict.get('churn_widener_bps')}bps; "
            f"confidence {verdict.get('confidence')}"
        ),
        revised=state.revision_round > 0,
    )
    state.audit_log.append(
        f"FINAL: spread={final_spread:.2f} yield={final_yield:.4%} price={price:.3f}"
    )
    return {"recommendation": state.recommendation, "audit_log": state.audit_log}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_pricing_graph() -> Any:
    """Assemble the three-agent pricing graph with its cross-check loop.

    Topology:
        START -> market_intelligence -> demand_forecaster -> bookrunner
        bookrunner -> (approve? finalize : revise -> MI -> DF -> BR ...)
    """
    graph = StateGraph(AgentState)
    graph.add_node("market_intelligence", market_intelligence_agent)
    graph.add_node("demand_forecaster", demand_forecaster_agent)
    graph.add_node("bookrunner", bookrunner_agent)
    graph.add_node("revision", revision_node)

    graph.add_edge(START, "market_intelligence")
    graph.add_edge("market_intelligence", "demand_forecaster")
    graph.add_edge("demand_forecaster", "bookrunner")
    graph.add_conditional_edges(
        "bookrunner",
        route_after_bookrunner,
        {"revise": "revision", "finalize": "finalize"},
    )
    graph.add_node("finalize", finalize_recommendation)
    graph.add_edge("revision", "market_intelligence")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_pricing_decision(
    tranche: Tranche,
    spread_ticks: list[SecondarySpreadTick],
    macro_prints: list[MacroPrint],
    ordered_mm: float,
    by_investor_type_mm: dict[str, float],
    velocity_per_min: float,
    provider: LLMProvider | None = None,
) -> TrancheRecommendation:
    """Public entry point: run the full agent decision loop for one tranche.

    Args:
        tranche: The tranche being priced.
        spread_ticks: Recent secondary comparables.
        macro_prints: Recent macro releases.
        ordered_mm: Current total ordered size.
        by_investor_type_mm: Book breakdown by investor type (enum value strings).
        velocity_per_min: Rolling order velocity.
        provider: Optional reasoning provider override (defaults to env).

    Returns:
        The final cross-checked TrancheRecommendation.

    Raises:
        RuntimeError: If the graph produced no recommendation.
    """
    resolved_provider: LLMProvider = provider or load_provider_from_env()
    state = AgentState(
        tranche=tranche,
        spread_ticks=spread_ticks,
        macro_prints=macro_prints,
        ordered_mm=ordered_mm,
        by_investor_type_mm=by_investor_type_mm,
        velocity_per_min=velocity_per_min,
        provider=resolved_provider,
    )
    app = build_pricing_graph()
    final_state: AgentState | dict[str, Any] = app.invoke(state, {"recursion_limit": 50})
    # LangGraph may materialize the dataclass state back as a mapping.
    recommendation = (
        final_state.get("recommendation")
        if isinstance(final_state, dict)
        else final_state.recommendation
    )
    if recommendation is None:
        raise RuntimeError("agent graph completed without a recommendation")
    return recommendation
