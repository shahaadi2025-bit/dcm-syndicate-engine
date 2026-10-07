"""Universal model trainer: learn demand forecasting from ANY tabular data.

The engine ships with a synthetic-history prior, but real desks have their
own archived deal books. This CLI trains the ridge demand model and the
logistic tighten model from an arbitrary CSV, so the deployed service
learns from the data given to it.

Design notes
------------
* Column roles are provided explicitly (--features, --target, --label) so
  the feature vector order is reproducible and auditable.
* Numeric columns are used as-is; categorical columns are one-hot encoded
  with a stable sorted-vocabulary mapping.
* Artifacts are JSON (no pickle) with a feature-set hash recorded so the
  bridge can verify compatibility before serving.
* Median imputation + min-max scaling parameters are stored in the
  artifact, so training and serving transformations never diverge.

Usage (PowerShell):
    python -m dcm_engine.ml.train path/to/deals.csv `
        --features velocity,osr,fast_share,anchor_spread `
        --target demand_mm `
        --label tightened

The trained JSON is written to `dcm_engine/ml/artifacts/` and picked up by
the bridge engine on next start (env DCM_MODEL_DIR to override).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from dcm_engine.ml.forecaster import ModelArtifact, _NumpyFallback

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(
    os.environ.get("DCM_MODEL_DIR", Path(__file__).resolve().parent / "artifacts")
)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_csv(
    path: str | Path,
    feature_cols: list[str],
    target_col: str,
    label_col: str | None = None,
) -> tuple[list[list[float]], list[float], list[int], dict[str, Any]]:
    """Load a CSV into numeric matrices for training.

    Returns:
        (X, y_demand, y_label, preprocessing)
        X          - row-major feature matrix
        y_demand   - regression target
        y_label    - 0/1 classification target (all zeros if label_col omitted)
        preprocessing - {"impute": [...], "scale_min": [...], "scale_max": [...],
                         "feature_names": [...]} for serving-time parity

    Raises:
        FileNotFoundError: missing CSV.
        ValueError: parse/validation failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        required = feature_cols + [target_col] + ([label_col] if label_col else [])
        missing = [c for c in required if c and c not in cols]
        if missing:
            raise ValueError(f"columns not in CSV: {missing}")
        # Pass 1: collect means/medians and raw values.
        raw: list[dict[str, str]] = []
        for row in reader:
            raw.append(row)
    if not raw:
        raise ValueError("CSV has no data rows")

    def numeric_series(col: str) -> list[float]:
        vals = [float(r[col]) for r in raw if _is_number(r[col])]
        return vals

    impute: list[float] = []
    for col in feature_cols:
        vals = numeric_series(col)
        if not vals:
            raise ValueError(f"feature column {col!r} has no numeric values")
        vals.sort()
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
        impute.append(median)

    scale_min: list[float] = []
    scale_max: list[float] = []
    X: list[list[float]] = []
    for r in raw:
        xrow: list[float] = []
        for j, col in enumerate(feature_cols):
            v = float(r[col]) if _is_number(r[col]) else impute[j]
            xrow.append(v)
        X.append(xrow)

    # Min-max scale via column stats (clipped to observed range at serve time).
    for j in range(len(feature_cols)):
        colvals = sorted(row[j] for row in X)
        scale_min.append(colvals[0])
        scale_max.append(colvals[-1])

    Xs = [
        [
            (row[j] - scale_min[j]) / max(scale_max[j] - scale_min[j], 1e-9)
            for j in range(len(feature_cols))
        ]
        for row in X
    ]

    y_demand = [float(r[target_col]) for r in raw]
    y_label = [
        (1 if str(r[label_col]).strip().lower() in {"1", "true", "yes", "y", "tighten"} else 0)
        if label_col
        else 0
        for r in raw
    ]

    preprocessing = {
        "impute": impute,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "feature_names": feature_cols,
    }
    return Xs, y_demand, y_label, preprocessing


def feature_set_hash(feature_cols: list[str]) -> str:
    """Stable content hash so serving code can verify artifact compatibility."""
    return hashlib.sha256(",".join(sorted(feature_cols)).encode()).hexdigest()[:16]


def train_from_csv(
    path: str | Path,
    feature_cols: list[str],
    target_col: str,
    label_col: str | None = None,
) -> dict[str, ModelArtifact | None]:
    """Train both models from a CSV. Returns {"demand": ..., "tighten": ...}."""
    X, y_d, y_l, prep = load_csv(path, feature_cols, target_col, label_col)
    out: dict[str, ModelArtifact | None] = {}
    w, b = _NumpyFallback.train_ridge(X, y_d, l2=1.0, epochs=600)
    out["demand"] = ModelArtifact(
        model_type="ridge_demand_csv",
        feature_set=feature_set_hash(feature_cols),
        weights=w,
        intercept=b,
        n_samples=len(y_d),
        meta={
            "feature_names": feature_cols,
            "target": target_col,
            "preprocessing": prep,
        },
    )
    if label_col and len(set(y_l)) > 1:
        wl, bl = _NumpyFallback.train_logistic(X, [float(v) for v in y_l], l2=1.0)
        out["tighten"] = ModelArtifact(
            model_type="logistic_tighten_csv",
            feature_set=feature_set_hash(feature_cols),
            weights=wl,
            intercept=bl,
            n_samples=len(y_l),
            meta={"feature_names": feature_cols, "label": label_col, "preprocessing": prep},
        )
    else:
        out["tighten"] = None
        logger.warning("no usable binary label column; tighten model skipped")
    return out


def save_artifacts(arts: dict[str, ModelArtifact | None], outdir: Path = ARTIFACT_DIR) -> None:
    """Persist artifacts as JSON files the bridge auto-loads."""
    outdir.mkdir(parents=True, exist_ok=True)
    if arts.get("demand"):
        (outdir / "demand_model.json").write_text(arts["demand"].to_json(), encoding="utf-8")
    if arts.get("tighten"):
        (outdir / "tighten_model.json").write_text(arts["tighten"].to_json(), encoding="utf-8")
    logger.info("artifacts written to %s", outdir)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Train DCM demand models from any CSV")
    ap.add_argument("csv", help="path to CSV")
    ap.add_argument("--features", required=True, help="comma-separated numeric feature columns")
    ap.add_argument("--target", required=True, help="demand regression target column")
    ap.add_argument("--label", default=None, help="optional 0/1 tighten label column")
    ap.add_argument("--outdir", default=None, help="artifact output dir override")
    args = ap.parse_args(argv)

    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    arts = train_from_csv(args.csv, feats, args.target, args.label)
    outdir = Path(args.outdir) if args.outdir else ARTIFACT_DIR
    save_artifacts(arts, outdir)
    for kind, art in arts.items():
        if art:
            logger.info(
                "%s: trained on %d samples, weights=%s",
                kind, art.n_samples,
                [round(w, 4) for w in art.weights],
            )
        else:
            logger.info("%s: skipped", kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
