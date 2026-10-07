"""Unit tests for the quant pricing core (YTM, spread matrix, waterfall)."""

from __future__ import annotations

import pytest

from dcm_engine.core.models import InvestorType
from dcm_engine.core.pricing import (
    adjusted_guidance_spread_bps,
    bond_price_from_yield,
    interpolate_benchmark_yield,
    matrix_spread_adjustment_bps,
    oversubscription_ratio,
    signal_weighted_osr,
    tranche_pricing_waterfall,
    yield_to_maturity,
)


class TestYTM:
    def test_par_bond_ytm_equals_coupon(self) -> None:
        """A bond priced at par yields exactly its coupon rate."""
        ytm = yield_to_maturity(price=100.0, coupon_rate=0.05, years=10.0, freq=2)
        assert ytm == pytest.approx(0.05, abs=1e-6)

    def test_roundtrip_price_yield(self) -> None:
        """Solving YTM from a price and repricing must converge to the price."""
        price = 97.35
        ytm = yield_to_maturity(price=price, coupon_rate=0.045, years=7.5, freq=2)
        repriced = bond_price_from_yield(
            face=100.0, ytm=ytm, coupon_rate=0.045, years=7.5, freq=2
        )
        assert repriced == pytest.approx(price, abs=1e-6)

    def test_zero_coupon_discount_below_par(self) -> None:
        ytm = yield_to_maturity(price=61.39, coupon_rate=0.0, years=10.0, freq=2)
        assert ytm == pytest.approx(0.05, abs=1e-3)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError):
            yield_to_maturity(price=-5.0, coupon_rate=0.05, years=5.0)


class TestSpreadMatrix:
    def test_matrix_cells(self) -> None:
        assert matrix_spread_adjustment_bps(0.5) == 15.0
        assert matrix_spread_adjustment_bps(0.9) == 8.0
        assert matrix_spread_adjustment_bps(1.5) == 0.0
        assert matrix_spread_adjustment_bps(2.5) == -6.0
        assert matrix_spread_adjustment_bps(4.0) == -12.0
        assert matrix_spread_adjustment_bps(7.0) == -18.0

    def test_boundary_osr_exact_one_is_covered(self) -> None:
        assert matrix_spread_adjustment_bps(1.0) == 0.0

    def test_negative_osr_rejected(self) -> None:
        with pytest.raises(ValueError):
            matrix_spread_adjustment_bps(-0.1)

    def test_signal_weighting_discounts_hedge_funds(self) -> None:
        """A hedge-fund-only book must produce a lower weighted OSR than raw."""
        orders = {InvestorType.HEDGE_FUND: 300.0}
        weighted = signal_weighted_osr(orders, target_mm=100.0)
        assert weighted == pytest.approx(3.0 * 0.70)

    def test_anchors_boost_weighted_osr(self) -> None:
        orders = {InvestorType.CENTRAL_BANK: 100.0}
        weighted = signal_weighted_osr(orders, target_mm=100.0)
        assert weighted == pytest.approx(1.25)

    def test_adjusted_guidance_tightens_on_hot_book(self) -> None:
        adjusted = adjusted_guidance_spread_bps(
            guidance_bps=120.0, ordered_mm=4000.0, target_mm=1000.0
        )
        assert adjusted == pytest.approx(120.0 - 12.0)

    def test_adjusted_guidance_widens_on_weak_book(self) -> None:
        adjusted = adjusted_guidance_spread_bps(
            guidance_bps=120.0, ordered_mm=600.0, target_mm=1000.0
        )
        assert adjusted == pytest.approx(120.0 + 15.0)


class TestWaterfall:
    def test_two_tranche_waterfall_shapes(self) -> None:
        from dcm_engine.core.pricing import WaterfallInput

        tranches = [
            WaterfallInput(
                tranche_id="SENIOR",
                target_size_mm=1000.0,
                spread_guidance_bps=110.0,
                benchmark_yield=0.0265,
                coupon_rate=0.0365,
            ),
            WaterfallInput(
                tranche_id="SUB",
                target_size_mm=500.0,
                spread_guidance_bps=180.0,
                benchmark_yield=0.0265,
                coupon_rate=0.0445,
            ),
        ]
        results = tranche_pricing_waterfall(
            tranches,
            ordered_by_tranche={"SENIOR": 3200.0, "SUB": 900.0},
        )
        by_id = {r.tranche_id: r for r in results}
        # 3.2x book -> "Hot" cell (3.0-5.0x)
        assert by_id["SENIOR"].final_spread_bps == pytest.approx(110.0 - 12.0)
        # 1.8x book -> covered cell
        assert by_id["SUB"].final_spread_bps == pytest.approx(180.0)
        # Senior must remain tighter than sub (hierarchy preserved).
        assert by_id["SENIOR"].final_spread_bps < by_id["SUB"].final_spread_bps
        # Prices must be positive and sane.
        assert 80.0 < by_id["SENIOR"].final_price < 120.0

    def test_missing_order_raises(self) -> None:
        from dcm_engine.core.pricing import WaterfallInput

        tranches = [
            WaterfallInput(
                tranche_id="X", target_size_mm=100.0,
                spread_guidance_bps=100.0, benchmark_yield=0.03,
            )
        ]
        with pytest.raises(KeyError):
            tranche_pricing_waterfall(tranches, ordered_by_tranche={})


class TestCurve:
    def test_interpolation_midpoint(self) -> None:
        curve = {5.0: 0.03, 10.0: 0.04}
        assert interpolate_benchmark_yield(7.5, curve) == pytest.approx(0.035)

    def test_clamps_at_edges(self) -> None:
        curve = {5.0: 0.03, 10.0: 0.04}
        assert interpolate_benchmark_yield(2.0, curve) == pytest.approx(0.03)
        assert interpolate_benchmark_yield(30.0, curve) == pytest.approx(0.04)


class TestOSR:
    def test_osr_basic(self) -> None:
        assert oversubscription_ratio(2500.0, 1000.0) == pytest.approx(2.5)

    def test_zero_target_raises(self) -> None:
        with pytest.raises(ValueError):
            oversubscription_ratio(100.0, 0.0)
