"""LLM-backed reasoning providers for the agent mesh.

The LangGraph agents use an LLM for narrative analysis and rationale
generation. This module isolates all model access behind a tiny protocol
so the graph itself stays deterministic and unit-testable:

* `LLMProvider` - the protocol every provider implements.
* `RuleBasedProvider` - deterministic fallback for CI, offline runs, and
  model outages. Never makes a network call.
* `LangChainProvider` - wraps any LangChain chat model (ChatOpenAI,
  ChatAnthropic, ...) behind the same protocol.

Selection is via the `DCM_LLM_PROVIDER` env var: `rule` (default) or
`langchain`. Model credentials are read from standard env vars by the
LangChain client itself (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dcm_engine.core.models import MacroPrint, SecondarySpreadTick, Tranche


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for agent reasoning backends."""

    def complete(self, prompt: str) -> str:
        """Return a short natural-language analysis for the given prompt."""
        ...


@dataclass(slots=True)
class RuleBasedProvider:
    """Deterministic template-based provider.

    Encodes the desk's institutional heuristics in plain rules. Used as
    the default because an autonomous pricing engine must degrade to
    auditable deterministic behavior, never hallucinate a spread.
    """

    name: str = "rule-based"

    def complete(self, prompt: str) -> str:
        # The prompt already carries the structured facts; return the
        # deterministic desk heuristics keyed off markers in the prompt.
        if "SPREAD_ANALYSIS" in prompt:
            return (
                "Secondary comparables are the primary anchor. Macro prints "
                "modulate the level: hawkish surprises widen, dovish tighten. "
                "Recommendation derives from comparables + macro overlay only."
            )
        if "DEMAND_FORECAST" in prompt:
            return (
                "Book velocity and investor mix drive the demand curve. "
                "Fast money is discounted for churn risk; anchors are weighted."
            )
        if "BOOKRUNNER_CHECK" in prompt:
            return (
                "Risk exposure is evaluated against warehouse limits and "
                "undersubscription scenarios before any recommendation ships."
            )
        return "No domain-specific heuristic for this prompt; defaulting to conservative stance."


@dataclass(slots=True)
class LangChainProvider:
    """Adapter over a LangChain chat model. Constructed lazily so importing
    this module never requires model SDKs to be installed."""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1  # Low temperature: pricing narratives must be stable
    _client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI  # Optional dependency

                self._client = ChatOpenAI(model=self.model_name, temperature=self.temperature)
            except ImportError as exc:
                raise RuntimeError(
                    "langchain_openai not installed; pip install 'dcm-syndicate-engine[agents]'"
                ) from exc
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.invoke(prompt)
        return str(response.content)


def load_provider_from_env() -> LLMProvider:
    """Resolve the reasoning provider from configuration.

    Env vars:
        DCM_LLM_PROVIDER: 'rule' (default) | 'langchain'
        DCM_LLM_MODEL:    model name for the langchain provider
    """
    kind = os.environ.get("DCM_LLM_PROVIDER", "rule").strip().lower()
    if kind == "langchain":
        return LangChainProvider(
            model_name=os.environ.get("DCM_LLM_MODEL", "gpt-4o-mini")
        )
    return RuleBasedProvider()


# ---------------------------------------------------------------------------
# Structured prompt builders (shared by all agents)
# ---------------------------------------------------------------------------


def build_spread_analysis_prompt(
    ticks: list[SecondarySpreadTick],
    macros: list[MacroPrint],
    tranche: Tranche,
) -> str:
    """Prompt for the Market Intelligence Agent."""
    tick_lines = "\n".join(
        f"  - {t.issuer} {t.tenor_years:.1f}Y: G-spread {t.g_spread_bps:.1f}bps, "
        f"bench {t.benchmark_yield:.4%}"
        for t in ticks
    ) or "  - (no secondary ticks received)"
    macro_lines = "\n".join(
        f"  - {m.indicator}: {m.release} vs {m.consensus} consensus "
        f"(surprise {m.surprise_bps:+.1f}bps)"
        for m in macros
    ) or "  - (no macro prints)"

    return (
        "SPREAD_ANALYSIS requested.\n"
        f"Tranche: {tranche.tranche_id} tenor={tranche.tenor_years}Y "
        f"guidance={tranche.spread_guidance_bps:.1f}bps "
        f"bench={tranche.benchmark_yield:.4%}\n"
        f"Secondary comparables:\n{tick_lines}\n"
        f"Macro prints:\n{macro_lines}\n"
        "Assess whether initial guidance is inside, through, or wide of "
        "secondary and produce a spread view."
    )


def build_demand_forecast_prompt(
    velocity_per_min: float,
    total_ordered_mm: float,
    target_mm: float,
    by_type: dict[str, float],
) -> str:
    """Prompt for the Investor Sentiment & Demand Forecaster."""
    mix = ", ".join(f"{k}: {v:.0f}mm" for k, v in by_type.items()) or "none"
    osr = total_ordered_mm / target_mm if target_mm > 0 else 0.0
    return (
        "DEMAND_FORECAST requested.\n"
        f"Book: ordered={total_ordered_mm:.0f}mm target={target_mm:.0f}mm "
        f"(OSR={osr:.2f}x) velocity={velocity_per_min:.0f} orders/min\n"
        f"Investor mix: {mix}\n"
        "Forecast closing demand and flag churn risk from fast money."
    )


def build_bookrunner_check_prompt(
    tranche_id: str,
    proposed_spread_bps: float,
    osr: float,
    forecast_demand_mm: float,
) -> str:
    """Prompt for the Syndicate Bookrunner risk cross-check."""
    return (
        "BOOKRUNNER_CHECK requested.\n"
        f"Tranche {tranche_id}: proposed spread {proposed_spread_bps:.1f}bps, "
        f"OSR={osr:.2f}x, forecast demand {forecast_demand_mm:.0f}mm\n"
        "Validate against warehouse capacity, undersubscription risk, and "
        "left-tail scenarios. Approve or challenge the level."
    )
