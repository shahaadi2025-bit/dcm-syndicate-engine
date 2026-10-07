"""Financial quant & pricing logic for the DCM syndicate engine.

Pure, dependency-free functions suitable for deterministic unit tests and
audit review. All yields/prices are unitless decimals unless suffixed `_bps`.

Conventions
-----------
* YTM is the periodic-compounded IRR of the bond cashflows.
* Spreads are in basis points over the interpolated benchmark curve.
* Price is clean price per 100 nominal.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Mapping

from dcm_engine.core.models import InvestorType

# ---------------------------------------------------------------------------
# Yield-to-Maturity
# ---------------------------------------------------------------------------


def bond_price_from_yield(
    *,
    face: float,
    ytm: float,
    coupon_rate: float,
    years: float,
    freq: int = 2,
) -> float:
    """Present value of a level-coupon bullet bond at the given YTM.

    Args:
        face: Par / face amount (e.g. 100 for price quoting).
        ytm: Yield to maturity per annum (decimal, e.g. 0.0475), compounded
            `freq` times per year.
        coupon_rate: Annual coupon rate (decimal, e.g. 0.045).
        years: Time to maturity in years (fractional allowed).
        freq: Compounding/coupon frequency per year (2 = semiannual).

    Returns:
        Dirty (full) price in the same units as `face`.

    Raises:
        ValueError: On non-positive face, non-positive frequency, or negative ytm.
    """
    if face <= 0:
        raise ValueError(f"face must be positive, got {face}")
    if freq < 1:
        raise ValueError(f"freq must be >= 1, got {freq}")
    if ytm <= 0:
        raise ValueError(f"ytm must be positive, got {ytm}")

    n = int(round(years * freq))
    if n <= 0:
        raise ValueError(f"years*freq must produce at least one coupon, got {years}")

    i = ytm / freq
    c = face * coupon_rate / freq
    # Discount factor for one period.
    df = (1.0 + i) ** (-n)
    # Annuity factor for the coupon stream.
    annuity = (1.0 - df) / i
    return c * annuity + face * df


def yield_to_maturity(
    *,
    price: float,
    face: float = 100.0,
    coupon_rate: float,
    years: float,
    freq: int = 2,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Solve YTM from clean/dirty price via bisection (robust, bracketed).

    Bisection is preferred over Newton-Raphson in a live pricing engine:
    Newton can diverge on near-zero coupon or stressed quotes, and a
    deterministic bracketed solve is trivially auditable.

    Args:
        price: Market price per 100 nominal (dirty).
        face: Redemption amount per unit quote (default 100).
        coupon_rate: Annual coupon rate (decimal).
        years: Time to maturity in years.
        freq: Coupon frequency per year.
        tol: Convergence tolerance on the PV error.
        max_iter: Hard iteration cap.

    Returns:
        YTM per annum (decimal).

    Raises:
        ValueError: If the bracket cannot be established or does not converge.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    def pv(y: float) -> float:
        return bond_price_from_yield(
            face=face, ytm=y, coupon_rate=coupon_rate, years=years, freq=freq
        )

    lo, hi = 1e-6, 2.0  # 0.0001% .. 200% bracket; sufficient for all markets
    f_lo, f_hi = pv(lo) - price, pv(hi) - price
    if f_lo * f_hi > 0:
        raise ValueError(
            "YTM bracket failed: price outside achievable range "
            f"(pv(lo)={pv(lo):.6f}, pv(hi)={pv(hi):.6f}, target={price:.6f})"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = pv(mid) - price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    raise ValueError(f"YTM bisection did not converge within {max_iter} iterations")


# ---------------------------------------------------------------------------
# Credit spread matrix: order-book oversubscription -> spread adjustment
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SpreadRule:
    """One cell of the oversubscription-driven spread matrix."""

    min_osr: float          # Inclusive lower bound of oversubscription ratio
    max_osr: float          # Exclusive upper bound (inf for the top cell)
    spread_delta_bps: float  # Signed adjustment to guidance spread


# Default institutional playbook: tighter as the book multiples up.
# OSR = ordered / target. 1.0x = covered; 3x+ = strong, tighten aggressively.
DEFAULT_SPREAD_MATRIX: tuple[SpreadRule, ...] = (
    SpreadRule(min_osr=0.0, max_osr=0.75, spread_delta_bps=+15.0),   # Undersubscribed
    SpreadRule(min_osr=0.75, max_osr=1.0, spread_delta_bps=+8.0),    # Barely building
    SpreadRule(min_osr=1.0, max_osr=2.0, spread_delta_bps=0.0),      # Covered
    SpreadRule(min_osr=2.0, max_osr=3.0, spread_delta_bps=-6.0),     # Healthy
    SpreadRule(min_osr=3.0, max_osr=5.0, spread_delta_bps=-12.0),    # Hot
    SpreadRule(min_osr=5.0, max_osr=float("inf"), spread_delta_bps=-18.0),  # Stampede
)

# Investors differ in signal quality: an anchor REAL_MONEY/central-bank order
# is a stronger tighten signal than hot HEDGE_FUND money that may flip.
# Values are multiplicative dampeners on the matrix delta per investor type.
_INVESTOR_SIGNAL_WEIGHT: Mapping[InvestorType, float] = {
    InvestorType.CENTRAL_BANK: 1.25,
    InvestorType.REAL_MONEY: 1.15,
    InvestorType.INSURANCE: 1.05,
    InvestorType.BANK_TREASURY: 0.90,
    InvestorType.RETAIL_AGGREGATOR: 0.80,
    InvestorType.HEDGE_FUND: 0.70,
}


def oversubscription_ratio(ordered_mm: float, target_mm: float) -> float:
    """OSR = total orders / target size. Guards against div-by-zero."""
    if target_mm <= 0:
        raise ValueError(f"target_mm must be positive, got {target_mm}")
    return ordered_mm / target_mm


def signal_weighted_osr(
    orders: Mapping[InvestorType, float],
    target_mm: float,
) -> float:
    """Oversubscription ratio weighted by investor signal quality.

    Real-money demand tightens pricing more than fast money because it is
    stickier at final allocation - the desk's core conviction metric.
    """
    if target_mm <= 0:
        raise ValueError(f"target_mm must be positive, got {target_mm}")
    weighted = sum(
        size_mm * _INVESTOR_SIGNAL_WEIGHT.get(inv_type, 0.85)
        for inv_type, size_mm in orders.items()
    )
    return weighted / target_mm


def matrix_spread_adjustment_bps(
    osr: float,
    matrix: tuple[SpreadRule, ...] = DEFAULT_SPREAD_MATRIX,
) -> float:
    """Look up the signed spread delta for an oversubscription ratio.

    Raises:
        ValueError: If osr is negative (a data-quality breach upstream).
    """
    if osr < 0:
        raise ValueError(f"osr must be non-negative, got {osr}")
    for rule in matrix:
        if rule.min_osr <= osr < rule.max_osr:
            return rule.spread_delta_bps
    # osr == top boundary exactly: clamp to the widest cell.
    return matrix[-1].spread_delta_bps


def adjusted_guidance_spread_bps(
    *,
    guidance_bps: float,
    ordered_mm: float,
    target_mm: float,
    orders: Mapping[InvestorType, float] | None = None,
    matrix: tuple[SpreadRule, ...] = DEFAULT_SPREAD_MATRIX,
    blend_fast_money: bool = True,
) -> float:
    """Final guidance spread after book-driven adjustment.

    Uses the signal-weighted OSR when a per-type breakdown is supplied,
    optionally blended with the raw OSR so a book of pure hedge-fund flips
    cannot tighten pricing beyond its true conviction.
    """
    raw_osr = oversubscription_ratio(ordered_mm, target_mm)
    if orders:
        weighted_osr = signal_weighted_osr(orders, target_mm)
        osr = (0.6 * weighted_osr + 0.4 * raw_osr) if blend_fast_money else weighted_osr
    else:
        osr = raw_osr
    delta = matrix_spread_adjustment_bps(osr, matrix)
    return guidance_bps + delta


# ---------------------------------------------------------------------------
# Tranche pricing waterfall
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WaterfallInput:
    """One tranche entering the pricing waterfall."""

    tranche_id: str
    target_size_mm: float
    spread_guidance_bps: float
    benchmark_yield: float
    coupon_rate: float = 0.0       # 0 => price as zero at final yield


@dataclass(slots=True, frozen=True)
class WaterfallResult:
    """Output of the tranche pricing waterfall for one tranche."""

    tranche_id: str
    guidance_spread_bps: float
    book_spread_bps: float         # After oversubscription adjustment
    final_spread_bps: float        # After allocation-pressure overlay
    final_yield: float
    final_price: float
    osr: float
    notes: tuple[str, ...] = ()


def tranche_pricing_waterfall(
    tranches: list[WaterfallInput],
    ordered_by_tranche: Mapping[str, float],
    orders_by_tranche: Mapping[str, Mapping[InvestorType, float]] | None = None,
    *,
    seniority_tightener_bps: Mapping[str, float] | None = None,
) -> list[WaterfallResult]:
    """Run the sequential pricing waterfall across tranches.

    Order of operations mirrors a live syndicate decision:
      1. Start from initial spread guidance.
      2. Apply the oversubscription matrix (book signal).
      3. Apply any seniority/hierarchy overlay (e.g. senior vs sub spread
         minimums must hold: sub must stay wider than senior).
      4. Convert spread -> all-in yield -> clean price.

    Args:
        tranches: Ordered tranche list (senior first).
        ordered_by_tranche: Total ordered MM per tranche_id.
        orders_by_tranche: Optional per-investor-type breakdown per tranche.
        seniority_tightener_bps: Optional per-tranche extra tightening floor
            deltas (e.g.{"senior": -2.0}) applied last.

    Returns:
        WaterfallResult per tranche, same order as input.

    Raises:
        KeyError: If a tranche lacks an entry in `ordered_by_tranche`.
    """
    orders_by_tranche = orders_by_tranche or {}
    seniority_tightener_bps = seniority_tightener_bps or {}
    results: list[WaterfallResult] = []

    for tr in tranches:
        if tr.tranche_id not in ordered_by_tranche:
            raise KeyError(f"missing ordered size for tranche {tr.tranche_id!r}")

        ordered = ordered_by_tranche[tr.tranche_id]
        osr = oversubscription_ratio(ordered, tr.target_size_mm)
        orders = orders_by_tranche.get(tr.tranche_id, {})

        notes: list[str] = []
        book_spread = adjusted_guidance_spread_bps(
            guidance_bps=tr.spread_guidance_bps,
            ordered_mm=ordered,
            target_mm=tr.target_size_mm,
            orders=orders or None,
        )
        if orders:
            notes.append("signal-weighted OSR applied")

        final_spread = book_spread + seniority_tightener_bps.get(tr.tranche_id, 0.0)
        if final_spread < book_spread:
            notes.append("seniority overlay tightened")

        final_yield = tr.benchmark_yield + final_spread / 10_000.0
        if tr.coupon_rate > 0:
            # Price the bond at its final yield given its coupon.
            price = bond_price_from_yield(
                face=100.0,
                ytm=final_yield,
                coupon_rate=tr.coupon_rate,
                years=10.0,  # Placeholder tenor; real books carry per-tranche tenor
                freq=2,
            )
        else:
            # Zero-coupon convention: price from yield with no coupon stream.
            price = bond_price_from_yield(
                face=100.0,
                ytm=final_yield,
                coupon_rate=0.0,
                years=10.0,
                freq=2,
            )

        results.append(
            WaterfallResult(
                tranche_id=tr.tranche_id,
                guidance_spread_bps=tr.spread_guidance_bps,
                book_spread_bps=book_spread,
                final_spread_bps=final_spread,
                final_yield=final_yield,
                final_price=price,
                osr=round(osr, 3),
                notes=tuple(notes),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Curve / benchmark interpolation helper
# ---------------------------------------------------------------------------


def interpolate_benchmark_yield(
    tenor_years: float,
    curve: Mapping[float, float],
) -> float:
    """Linearly interpolate a benchmark govt yield at an arbitrary tenor.

    Args:
        tenor_years: Target tenor in years.
        curve: Mapping of tenor_years -> yield (decimal). Must have >= 1 point.

    Returns:
        Interpolated yield (decimal). Clamped at the curve edges.
    """
    if not curve:
        raise ValueError("curve must contain at least one point")
    tenors = sorted(curve)
    if tenor_years <= tenors[0]:
        return curve[tenors[0]]
    if tenor_years >= tenors[-1]:
        return curve[tenors[-1]]
    idx = bisect_right(tenors, tenor_years)
    t0, t1 = tenors[idx - 1], tenors[idx]
    y0, y1 = curve[t0], curve[t1]
    if t1 == t0:
        return y0
    w = (tenor_years - t0) / (t1 - t0)
    return y0 + w * (y1 - y0)
