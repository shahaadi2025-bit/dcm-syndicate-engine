"""Runnable demo: one full agent-mesh pricing decision, end to end.

Executes without Kafka or an LLM API key (rule-based provider). Useful as a
smoke test of the whole decision loop and as desk documentation of how the
agents interact.

Usage:
    python -m dcm_engine.agents.demo
"""

from __future__ import annotations

from dcm_engine.agents.graph import run_pricing_decision
from dcm_engine.agents.llm import RuleBasedProvider
from dcm_engine.core.models import MacroPrint, SecondarySpreadTick, Tranche


def main() -> None:
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

    # Book snapshot as aggregated by the streaming pipeline.
    ordered_mm = 4150.0
    by_type = {
        "REAL_MONEY": 2400.0,
        "INSURANCE": 750.0,
        "CENTRAL_BANK": 350.0,
        "HEDGE_FUND": 650.0,
    }
    velocity = 85.0

    rec = run_pricing_decision(
        tranche=tranche,
        spread_ticks=ticks,
        macro_prints=macros,
        ordered_mm=ordered_mm,
        by_investor_type_mm=by_type,
        velocity_per_min=velocity,
        provider=RuleBasedProvider(),
    )

    print("=" * 72)
    print("DCM SYNDICATE PRICING RECOMMENDATION")
    print("=" * 72)
    print(f"Tranche        : {rec.tranche_id}  ({tranche.currency})")
    print(f"Target size    : {tranche.target_size_mm:,.0f}mm")
    print(f"Final spread   : {rec.final_spread_bps:.2f} bps")
    print(f"Final yield    : {rec.final_yield:.4%}")
    print(f"Final price    : {rec.final_price:.3f}")
    print(f"Forecast demand: {rec.forecast_demand_mm:,.1f}mm  "
          f"(OSR {rec.oversubscription_ratio:.2f}x)")
    print(f"Confidence     : {rec.confidence:.2f}")
    print(f"Revised        : {rec.revised}")
    print("-" * 72)
    print(f"Rationale: {rec.rationale}")


if __name__ == "__main__":
    main()
