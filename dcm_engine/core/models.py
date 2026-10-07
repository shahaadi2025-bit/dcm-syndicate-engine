"""Domain models shared across the DCM syndicate engine.

These dataclasses are the contract between the streaming pipeline
(Kafka/Redpanda), the LangGraph agent mesh, and the pricing core.
Keep them dependency-free so they import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def utc_now() -> datetime:
    """Timezone-aware UTC now (naive datetimes are a production bug source)."""
    return datetime.now(tz=timezone.utc)


class InvestorType(str, Enum):
    """Institutional investor classification driving allocation rules."""

    REAL_MONEY = "REAL_MONEY"          # Asset managers / pension funds
    HEDGE_FUND = "HEDGE_FUND"          # Fast money, higher churn
    BANK_TREASURY = "BANK_TREASURY"    # Bank treasury / balance-sheet accounts
    CENTRAL_BANK = "CENTRAL_BANK"      # Official sector, anchor demand
    INSURANCE = "INSURANCE"            # Buy & hold, liability matched
    RETAIL_AGGREGATOR = "RETAIL_AGGREGATOR"


class SignalDirection(str, Enum):
    """Direction of a market-intelligence signal."""

    TIGHTEN = "TIGHTEN"   # Spreads compressing -> price higher / yield lower
    WIDEN = "WIDEN"       # Spreads blowing out -> reprice wider
    NEUTRAL = "NEUTRAL"


class BookState(str, Enum):
    """Lifecycle state of a live order book for a deal."""

    OPEN = "OPEN"
    BUILDING = "BUILDING"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass(slots=True)
class MacroPrint:
    """A macroeconomic data release (CPI, NFP, PMI, GDP...)."""

    indicator: str
    release: float
    consensus: float
    unit: str = "idx"
    ts: datetime = field(default_factory=utc_now)

    @property
    def surprise_bps(self) -> float:
        """Signed surprise vs consensus, normalized to basis points."""
        if self.consensus == 0:
            return 0.0
        return (self.release - self.consensus) / abs(self.consensus) * 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "release": self.release,
            "consensus": self.consensus,
            "unit": self.unit,
            "ts": self.ts.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MacroPrint:
        return cls(
            indicator=str(payload["indicator"]),
            release=float(payload["release"]),
            consensus=float(payload["consensus"]),
            unit=str(payload.get("unit", "idx")),
        )


@dataclass(slots=True)
class SecondarySpreadTick:
    """One tick of secondary-market spread data for a comparable issuer/bucket."""

    issuer: str
    tenor_years: float
    g_spread_bps: float          # Spread over the interpolated govt curve
    benchmark_yield: float       # Interpolated benchmark govt yield (decimal)
    source: str = "internal"
    ts: datetime = field(default_factory=utc_now)

    @property
    def all_in_yield(self) -> float:
        """All-in secondary yield for the comparable bond (decimal)."""
        return self.benchmark_yield + self.g_spread_bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "tenor_years": self.tenor_years,
            "g_spread_bps": self.g_spread_bps,
            "benchmark_yield": self.benchmark_yield,
            "source": self.source,
            "ts": self.ts.isoformat(),
        }


@dataclass(slots=True)
class IOI:
    """Indication of Interest from an institutional investor.

    These arrive at high velocity on the bookbuilding topic. `limit_yield`
    is the investor's minimum acceptable yield (max price). Investors
    grinding their limit lower = strong demand signal.
    """

    investor_id: str
    investor_type: InvestorType
    deal_id: str
    tranche_id: str
    size_mm: float                       # Notional in millions
    limit_yield: float                   # Minimum acceptable YTM (decimal)
    price_limit: float | None = None     # Optional explicit price limit
    ts: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "investor_id": self.investor_id,
            "investor_type": self.investor_type.value,
            "deal_id": self.deal_id,
            "tranche_id": self.tranche_id,
            "size_mm": self.size_mm,
            "limit_yield": self.limit_yield,
            "price_limit": self.price_limit,
            "ts": self.ts.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IOI:
        return cls(
            investor_id=str(payload["investor_id"]),
            investor_type=InvestorType(str(payload["investor_type"])),
            deal_id=str(payload["deal_id"]),
            tranche_id=str(payload["tranche_id"]),
            size_mm=float(payload["size_mm"]),
            limit_yield=float(payload["limit_yield"]),
            price_limit=(
                float(payload["price_limit"]) if payload.get("price_limit") is not None else None
            ),
        )


@dataclass(slots=True)
class Tranche:
    """A tranche of a multi-tranche benchmark issuance."""

    tranche_id: str
    currency: str
    target_size_mm: float
    tenor_years: float
    spread_guidance_bps: float    # Initial price talk over benchmark (bps)
    coupon: float = 0.0
    benchmark_yield: float = 0.0
    book_state: BookState = BookState.OPEN

    @property
    def initial_yield(self) -> float:
        """Yield implied by initial guidance (decimal)."""
        return self.benchmark_yield + self.spread_guidance_bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tranche_id": self.tranche_id,
            "currency": self.currency,
            "target_size_mm": self.target_size_mm,
            "tenor_years": self.tenor_years,
            "spread_guidance_bps": self.spread_guidance_bps,
            "coupon": self.coupon,
            "benchmark_yield": self.benchmark_yield,
            "book_state": self.book_state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tranche:
        return cls(
            tranche_id=str(payload["tranche_id"]),
            currency=str(payload["currency"]),
            target_size_mm=float(payload["target_size_mm"]),
            tenor_years=float(payload["tenor_years"]),
            spread_guidance_bps=float(payload["spread_guidance_bps"]),
            coupon=float(payload.get("coupon", 0.0)),
            benchmark_yield=float(payload.get("benchmark_yield", 0.0)),
            book_state=BookState(str(payload.get("book_state", BookState.OPEN.value))),
        )


@dataclass(slots=True)
class TrancheRecommendation:
    """Final priced recommendation for one tranche, cross-checked by agents."""

    tranche_id: str
    final_spread_bps: float
    final_yield: float
    final_price: float
    forecast_demand_mm: float
    oversubscription_ratio: float
    size_mm: float
    confidence: float                 # 0..1 agent cross-check consensus score
    rationale: str
    revised: bool = False             # True if a reprice occurred intra-book

    def to_dict(self) -> dict[str, Any]:
        return {
            "tranche_id": self.tranche_id,
            "final_spread_bps": self.final_spread_bps,
            "final_yield": self.final_yield,
            "final_price": self.final_price,
            "forecast_demand_mm": self.forecast_demand_mm,
            "oversubscription_ratio": self.oversubscription_ratio,
            "size_mm": self.size_mm,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "revised": self.revised,
        }
