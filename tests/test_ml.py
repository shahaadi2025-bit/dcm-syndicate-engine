"""Tests for the ML forecaster module."""

from __future__ import annotations

from dcm_engine.ml.forecaster import (
    NUM_FEATURES,
    DemandCurveForecaster,
    SpreadMovementModel,
    build_feature_vector,
    synthesize_training_history,
)


def _features() -> list[float]:
    return build_feature_vector(
        velocity_per_min=60.0,
        ordered_mm=2500.0,
        target_mm=1000.0,
        by_investor_type_mm={"REAL_MONEY": 2000.0, "HEDGE_FUND": 500.0},
        guidance_spread_bps=115.0,
        current_spread_bps=110.0,
        hours_open=3.0,
    )


class TestFeatures:
    def test_feature_vector_length(self) -> None:
        assert len(_features()) == NUM_FEATURES

    def test_deterministic(self) -> None:
        kwargs = dict(
            velocity_per_min=60.0,
            ordered_mm=2500.0,
            target_mm=1000.0,
            by_investor_type_mm={"REAL_MONEY": 2000.0},
            guidance_spread_bps=115.0,
            current_spread_bps=110.0,
            hours_open=3.0,
        )
        assert build_feature_vector(**kwargs) == build_feature_vector(**kwargs)

    def test_zero_target_no_crash(self) -> None:
        x = build_feature_vector(
            velocity_per_min=0.0,
            ordered_mm=0.0,
            target_mm=0.0,
            by_investor_type_mm={},
            guidance_spread_bps=100.0,
            current_spread_bps=100.0,
            hours_open=0.0,
        )
        assert len(x) == NUM_FEATURES


class TestDemandForecaster:
    def test_untrained_uses_heuristic_with_bands(self) -> None:
        f = DemandCurveForecaster()
        out = f.predict_demand(
            velocity_per_min=60.0,
            ordered_mm=2500.0,
            target_mm=1000.0,
            by_investor_type_mm={"REAL_MONEY": 2000.0, "HEDGE_FUND": 500.0},
            guidance_spread_bps=115.0,
            current_spread_bps=110.0,
            hours_open=3.0,
        )
        assert out["source"] == "heuristic"
        assert out["demand_low_mm"] <= out["demand_mm"] <= out["demand_high_mm"]
        assert out["demand_mm"] > 0

    def test_trains_on_synthetic_history(self) -> None:
        f = DemandCurveForecaster()
        for x, y in synthesize_training_history(n_books=120, seed=11):
            f.observe(x, y)
        artifact = f.fit()
        assert artifact is not None
        assert artifact.n_samples == 120
        assert len(artifact.weights) == NUM_FEATURES
        assert f.is_trained

    def test_trained_model_predicts_and_roundtrips_json(self) -> None:
        f = DemandCurveForecaster()
        for x, y in synthesize_training_history(n_books=150, seed=5):
            f.observe(x, y)
        artifact = f.fit()
        assert artifact is not None

        # JSON roundtrip: no pickle, artifact survives serialization.
        clone = DemandCurveForecaster(
            artifact=__import__("dcm_engine.ml.forecaster", fromlist=["ModelArtifact"])
            .ModelArtifact.from_json(artifact.to_json())
        )
        assert clone.is_trained

        out = clone.predict_demand(
            velocity_per_min=70.0,
            ordered_mm=2800.0,
            target_mm=1000.0,
            by_investor_type_mm={"REAL_MONEY": 2300.0, "HEDGE_FUND": 500.0},
            guidance_spread_bps=115.0,
            current_spread_bps=108.0,
            hours_open=3.5,
        )
        assert out["source"] == "model"
        assert out["demand_mm"] > 0


class TestSpreadMovement:
    def test_untrained_heuristic_probability(self) -> None:
        m = SpreadMovementModel()
        out = m.predict_tighten_probability(
            velocity_per_min=80.0,
            ordered_mm=3500.0,
            target_mm=1000.0,
            by_investor_type_mm={"REAL_MONEY": 3000.0, "HEDGE_FUND": 500.0},
            guidance_spread_bps=120.0,
            current_spread_bps=118.0,
            hours_open=2.0,
        )
        assert 0.0 <= out["tighten_probability"] <= 1.0
        assert out["source"] == "heuristic"

    def test_trains_and_predicts(self) -> None:
        import random

        rng = random.Random(3)
        m = SpreadMovementModel()
        history = synthesize_training_history(n_books=150, seed=8)
        for x, _y in history:
            # Label: sticky hot books tighten more often.
            label = (x[1] > 1.2 and x[3] < 0.4) or rng.random() < 0.2
            m.observe(x, label)
        artifact = m.fit()
        assert artifact is not None

        out = m.predict_tighten_probability(
            velocity_per_min=90.0,
            ordered_mm=4000.0,
            target_mm=1000.0,
            by_investor_type_mm={"REAL_MONEY": 3800.0},
            guidance_spread_bps=120.0,
            current_spread_bps=119.0,
            hours_open=1.5,
        )
        assert 0.0 <= out["tighten_probability"] <= 1.0
        assert out["source"] == "model"
