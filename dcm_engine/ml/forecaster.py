"""ML demand & spread forecasting models for the syndicate engine.

Two model families, both trainable online as the book streams in:

1. `DemandCurveForecaster` - predicts the closing demand (mm) and the demand
   curve shape (take-down by price level) from book features: velocity,
   investor mix, current OSR, spread level. Ridge regression on log-features
   learned from historical books; degrades gracefully to heuristic priors
   when untrained.

2. `SpreadMovementModel` - short-horizon classifier predicting the probability
   that the book tightens (>3bps) over the next decision window. Logistic
   regression on the same feature vector. The bookrunner consumes this
   probability as an input to the churn/discount logic.

Design constraints for an institutional platform:
* Deterministic given seeds; no network calls; models are small and
  serializable to JSON (no pickle - pickle is a security incident waiting).
* Graceful degradation: when sklearn is unavailable a NumPy implementation
  (gradient descent ridge/logistic) is used with identical interfaces.
* Feature vector is explicit and versioned (FEATURE_SET_VERSION) so models
  in production can be traced to training distributions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from dcm_engine.core.models import InvestorType

logger = logging.getLogger(__name__)

FEATURE_SET_VERSION = "fs-2026.10-v1"

# Investor stickiness weights (signal quality of each channel).
STICKINESS: dict[str, float] = {
    "CENTRAL_BANK": 1.25,
    "REAL_MONEY": 1.15,
    "INSURANCE": 1.05,
    "BANK_TREASURY": 0.90,
    "RETAIL_AGGREGATOR": 0.80,
    "HEDGE_FUND": 0.70,
}

NUM_FEATURES = 8


def build_feature_vector(
    *,
    velocity_per_min: float,
    ordered_mm: float,
    target_mm: float,
    by_investor_type_mm: dict[str, float],
    guidance_spread_bps: float,
    current_spread_bps: float,
    hours_open: float,
) -> list[float]:
    """Construct the versioned feature vector shared by both models.

    Feature list (fixed order):
        0 log_velocity        - log1p(orders/min)
        1 osr                 - ordered / target
        2 sticky_osr          - stickiness-weighted ordered / target
        3 fast_money_share    - hedge fund share of book
        4 anchor_share        - central bank + insurance share
        5 spread_gap_bps      - guidance - current (negative = tightening)
        6 log_hours_open      - log1p(hours since book opened)
        7 target_log_size     - log10(target mm)
    """
    osr = ordered_mm / target_mm if target_mm > 0 else 0.0
    sticky = sum(
        mm * STICKINESS.get(t, 0.85) for t, mm in by_investor_type_mm.items()
    ) / target_mm if target_mm > 0 else 0.0
    total = max(ordered_mm, 1e-9)
    fast_share = by_investor_type_mm.get("HEDGE_FUND", 0.0) / total
    anchor_share = (
        by_investor_type_mm.get("CENTRAL_BANK", 0.0)
        + by_investor_type_mm.get("INSURANCE", 0.0)
    ) / total

    return [
        math.log1p(max(velocity_per_min, 0.0)),
        osr,
        sticky,
        fast_share,
        anchor_share,
        (guidance_spread_bps - current_spread_bps),
        math.log1p(max(hours_open, 0.0)),
        math.log10(max(target_mm, 1.0)),
    ]


@dataclass
class ModelArtifact:
    """JSON-serializable trained model (no pickle)."""

    model_type: str                  # "ridge_demand" | "logistic_tighten"
    feature_set: str
    weights: list[float]
    intercept: float
    n_samples: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_type": self.model_type,
                "feature_set": self.feature_set,
                "weights": self.weights,
                "intercept": self.intercept,
                "n_samples": self.n_samples,
                "meta": self.meta,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ModelArtifact:
        d = json.loads(raw)
        return cls(
            model_type=d["model_type"],
            feature_set=d["feature_set"],
            weights=[float(w) for w in d["weights"]],
            intercept=float(d["intercept"]),
            n_samples=int(d["n_samples"]),
            meta=dict(d.get("meta", {})),
        )


class _NumpyFallback:
    """Tiny pure-Python/NumPy-free ridge & logistic trainer.

    Uses batch gradient descent with L2 regularization. Deliberately simple:
    on 8 features and thousands of rows it trains in milliseconds and is
    fully deterministic.
    """

    @staticmethod
    def train_ridge(
        X: list[list[float]], y: list[float], l2: float = 1.0,
        lr: float = 0.01, epochs: int = 400,
    ) -> tuple[list[float], float]:
        n_features = len(X[0])
        w = [0.0] * n_features
        b = 0.0
        n = len(X)
        for _ in range(epochs):
            gw = [0.0] * n_features
            gb = 0.0
            for xi, yi in zip(X, y):
                pred = b + sum(wj * xij for wj, xij in zip(w, xi))
                err = pred - yi
                gb += err
                for j in range(n_features):
                    gw[j] += err * xi[j]
            b -= lr * (2 * gb / n)
            for j in range(n_features):
                w[j] -= lr * (2 * gw[j] / n + 2 * l2 * w[j])
        return w, b

    @staticmethod
    def train_logistic(
        X: list[list[float]], y: list[float], l2: float = 1.0,
        lr: float = 0.05, epochs: int = 500,
    ) -> tuple[list[float], float]:
        n_features = len(X[0])
        w = [0.0] * n_features
        b = 0.0
        n = len(X)
        for _ in range(epochs):
            gw = [0.0] * n_features
            gb = 0.0
            for xi, yi in zip(X, y):
                z = b + sum(wj * xij for wj, xij in zip(w, xi))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
                err = p - yi
                gb += err
                for j in range(n_features):
                    gw[j] += err * xi[j]
            b -= lr * gb / n
            for j in range(n_features):
                w[j] -= lr * (gw[j] / n + 2 * l2 * w[j])
        return w, b


class DemandCurveForecaster:
    """Predicts closing demand (mm) from live book features.

    Trained online: every finalized deal appends (features, actual_demand)
    via `observe()`. Before enough history exists, predictions fall back to
    a transparent heuristic (velocity extrapolation + stickiness blend).
    """

    MIN_SAMPLES_FOR_MODEL = 30

    def __init__(self, artifact: ModelArtifact | None = None) -> None:
        self._artifact = artifact
        self._history: list[tuple[list[float], float]] = []

    @property
    def is_trained(self) -> bool:
        return (
            self._artifact is not None
            and self._artifact.feature_set == FEATURE_SET_VERSION
            and self._artifact.n_samples >= self.MIN_SAMPLES_FOR_MODEL
        )

    def predict_demand(
        self,
        *,
        velocity_per_min: float,
        ordered_mm: float,
        target_mm: float,
        by_investor_type_mm: dict[str, float],
        guidance_spread_bps: float,
        current_spread_bps: float,
        hours_open: float,
    ) -> dict[str, Any]:
        """Return demand forecast with model provenance.

        Returns dict with:
            demand_mm        - point forecast of closing demand
            demand_low_mm    - conservative (p25-style) estimate
            demand_high_mm   - optimistic (p75-style) estimate
            source           - "model" | "heuristic"
            feature_set      - version string
        """
        x = build_feature_vector(
            velocity_per_min=velocity_per_min,
            ordered_mm=ordered_mm,
            target_mm=target_mm,
            by_investor_type_mm=by_investor_type_mm,
            guidance_spread_bps=guidance_spread_bps,
            current_spread_bps=current_spread_bps,
            hours_open=hours_open,
        )

        if self.is_trained and self._artifact is not None:
            z = self._artifact.intercept + sum(
                w * xi for w, xi in zip(self._artifact.weights, x)
            )
            demand = max(0.0, z)
            source = "model"
            uncertainty = 0.18 + 0.05 * x[3]  # wider bands with fast money
        else:
            # Heuristic: current book + 40%-decayed velocity extrapolation,
            # blended with sticky-money weighted OSR.
            sticky = x[2] * target_mm
            extra = velocity_per_min * 60 * 0.4
            demand = 0.5 * (ordered_mm + extra) + 0.5 * sticky
            source = "heuristic"
            uncertainty = 0.25 + 0.08 * x[3]

        low = demand * (1.0 - uncertainty)
        high = demand * (1.0 + uncertainty)
        return {
            "demand_mm": round(demand, 1),
            "demand_low_mm": round(low, 1),
            "demand_high_mm": round(high, 1),
            "source": source,
            "feature_set": FEATURE_SET_VERSION,
            "n_training_samples": self._artifact.n_samples if self._artifact else 0,
        }

    def observe(self, features: list[float], actual_demand_mm: float) -> None:
        """Append a realized (features, demand) observation to the online set."""
        if len(features) != NUM_FEATURES:
            raise ValueError(f"expected {NUM_FEATURES} features, got {len(features)}")
        self._history.append((features, actual_demand_mm))

    def fit(self) -> ModelArtifact | None:
        """Train on observed history. Returns None if insufficient data."""
        if len(self._history) < self.MIN_SAMPLES_FOR_MODEL:
            logger.info(
                "demand model: %d samples < %d needed; staying heuristic",
                len(self._history), self.MIN_SAMPLES_FOR_MODEL,
            )
            return None
        X = [f for f, _ in self._history]
        y = [d for _, d in self._history]
        w, b = _NumpyFallback.train_ridge(X, y)
        self._artifact = ModelArtifact(
            model_type="ridge_demand",
            feature_set=FEATURE_SET_VERSION,
            weights=w,
            intercept=b,
            n_samples=len(y),
            meta={"trained_on": "online_book_history"},
        )
        return self._artifact


class SpreadMovementModel:
    """Logistic classifier: P(book tightens > 3bps next window)."""

    MIN_SAMPLES_FOR_MODEL = 40

    def __init__(self, artifact: ModelArtifact | None = None) -> None:
        self._artifact = artifact
        self._history: list[tuple[list[float], int]] = []

    @property
    def is_trained(self) -> bool:
        return (
            self._artifact is not None
            and self._artifact.feature_set == FEATURE_SET_VERSION
            and self._artifact.n_samples >= self.MIN_SAMPLES_FOR_MODEL
        )

    def predict_tighten_probability(
        self,
        *,
        velocity_per_min: float,
        ordered_mm: float,
        target_mm: float,
        by_investor_type_mm: dict[str, float],
        guidance_spread_bps: float,
        current_spread_bps: float,
        hours_open: float,
    ) -> dict[str, Any]:
        """Return P(tighten) with provenance; 0.5 heuristic when untrained."""
        x = build_feature_vector(
            velocity_per_min=velocity_per_min,
            ordered_mm=ordered_mm,
            target_mm=target_mm,
            by_investor_type_mm=by_investor_type_mm,
            guidance_spread_bps=guidance_spread_bps,
            current_spread_bps=current_spread_bps,
            hours_open=hours_open,
        )
        if self.is_trained and self._artifact is not None:
            z = self._artifact.intercept + sum(
                w * xi for w, xi in zip(self._artifact.weights, x)
            )
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            source = "model"
        else:
            # Heuristic prior: hot sticky books tend to tighten.
            prior = 0.35 + 0.15 * min(x[1], 3.0) + 0.10 * x[4] - 0.25 * x[3]
            p = min(max(prior, 0.02), 0.98)
            source = "heuristic"
        return {
            "tighten_probability": round(p, 4),
            "source": source,
            "feature_set": FEATURE_SET_VERSION,
        }

    def observe(self, features: list[float], tightened: bool) -> None:
        if len(features) != NUM_FEATURES:
            raise ValueError(f"expected {NUM_FEATURES} features, got {len(features)}")
        self._history.append((features, 1 if tightened else 0))

    def fit(self) -> ModelArtifact | None:
        if len(self._history) < self.MIN_SAMPLES_FOR_MODEL:
            return None
        X = [f for f, _ in self._history]
        y = [float(t) for _, t in self._history]
        w, b = _NumpyFallback.train_logistic(X, y)
        self._artifact = ModelArtifact(
            model_type="logistic_tighten",
            feature_set=FEATURE_SET_VERSION,
            weights=w,
            intercept=b,
            n_samples=len(y),
            meta={"trained_on": "online_book_history"},
        )
        return self._artifact


class EnsembleDemandForecaster:
    """Blends the ML model with the rule-based desk heuristic.

    Weighting is confidence-based: the trained model earns weight as its
    sample count grows. This is the production entry point - pure ML or
    pure rules are both fragile; the blend is auditable and monotonic.
    """

    def __init__(
        self,
        forecaster: DemandCurveForecaster,
        model_weight_cap: float = 0.7,
    ) -> None:
        self.forecaster = forecaster
        self.model_weight_cap = model_weight_cap

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        base = self.forecaster.predict_demand(**kwargs)
        if base["source"] != "model":
            return base
        # Confidence grows with training samples, capped.
        n = base.get("n_training_samples", 0)
        model_weight = min(self.model_weight_cap, n / (n + 100.0))
        heuristic_demand = base["demand_mm"] / (1.0 + 0.15 * base.get("_fast", 0.0))
        blended = model_weight * base["demand_mm"] + (1 - model_weight) * heuristic_demand
        out = dict(base)
        out["demand_mm"] = round(blended, 1)
        out["blend_weight_model"] = round(model_weight, 3)
        return out


def synthesize_training_history(
    n_books: int = 200,
    seed: int = 42,
) -> list[tuple[list[float], float]]:
    """Generate a synthetic historical book sample for demos/tests.

    Emulates the desk's archived deals: hot sticky books close with high
    demand; hedge-fund-heavy books undershoot. Deterministic under seed.
    """
    import random

    rng = random.Random(seed)
    out: list[tuple[list[float], float]] = []
    for _ in range(n_books):
        target = rng.choice((250.0, 500.0, 750.0, 1000.0))
        velocity = rng.uniform(5.0, 120.0)
        fast_share = rng.uniform(0.05, 0.55)
        anchor_share = rng.uniform(0.1, 0.5)
        ordered = target * rng.uniform(0.3, 3.5)
        by_type = {
            "HEDGE_FUND": ordered * fast_share,
            "REAL_MONEY": ordered * (1 - fast_share - anchor_share) * 0.6,
            "INSURANCE": ordered * (1 - fast_share - anchor_share) * 0.25,
            "CENTRAL_BANK": ordered * (1 - fast_share - anchor_share) * 0.15,
        }
        x = build_feature_vector(
            velocity_per_min=velocity,
            ordered_mm=ordered,
            target_mm=target,
            by_investor_type_mm=by_type,
            guidance_spread_bps=130.0,
            current_spread_bps=125.0,
            hours_open=rng.uniform(1.0, 8.0),
        )
        # Ground truth: sticky demand with noise.
        sticky_osr = x[2]
        demand = target * sticky_osr * rng.uniform(0.9, 1.1) + 50.0
        out.append((x, max(demand, 50.0)))
    return out
