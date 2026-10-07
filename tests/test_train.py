"""Tests for the universal CSV trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcm_engine.ml.train import feature_set_hash, save_artifacts, train_from_csv


@pytest.fixture()
def deals_csv(tmp_path: Path) -> Path:
    p = tmp_path / "deals.csv"
    rows = ["velocity,osr,fast_share,demand_mm,tightened"]
    import random

    rng = random.Random(1)
    for _ in range(120):
        v = rng.uniform(5, 120)
        osr = rng.uniform(0.3, 3.5)
        fs = rng.uniform(0.05, 0.5)
        d = 400 * osr * (1 - 0.3 * fs) + rng.uniform(-40, 40)
        t = 1 if (osr > 1.5 and fs < 0.3) else 0
        rows.append(f"{v:.1f},{osr:.2f},{fs:.2f},{d:.1f},{t}")
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_train_from_csv_full(deals_csv: Path) -> None:
    arts = train_from_csv(deals_csv, ["velocity", "osr", "fast_share"], "demand_mm", "tightened")
    assert arts["demand"] is not None
    assert arts["demand"].n_samples == 120
    assert len(arts["demand"].weights) == 3
    assert arts["tighten"] is not None
    # Regression sanity: the OSR coefficient should be positive (more demand).
    assert arts["demand"].weights[1] > 0


def test_train_without_label_skips_tighten(deals_csv: Path) -> None:
    arts = train_from_csv(deals_csv, ["velocity", "osr", "fast_share"], "demand_mm")
    assert arts["demand"] is not None
    assert arts["tighten"] is None


def test_missing_column_raises(deals_csv: Path) -> None:
    with pytest.raises(ValueError):
        train_from_csv(deals_csv, ["velocity", "nope"], "demand_mm")
    with pytest.raises(ValueError):
        train_from_csv(deals_csv, ["velocity"], "missing_target")


def test_artifact_json_roundtrip_and_hash_stability(deals_csv: Path, tmp_path: Path) -> None:
    arts = train_from_csv(deals_csv, ["osr", "velocity"], "demand_mm", "tightened")
    save_artifacts(arts, tmp_path)
    saved = json.loads((tmp_path / "demand_model.json").read_text(encoding="utf-8"))
    assert saved["model_type"] == "ridge_demand_csv"
    assert saved["feature_set"] == feature_set_hash(["osr", "velocity"])
    # Hash stability: same column set, different order -> same hash.
    assert feature_set_hash(["velocity", "osr"]) == feature_set_hash(["osr", "velocity"])
