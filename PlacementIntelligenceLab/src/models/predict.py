"""Inference: single prediction, batch prediction, input validation, uncertainty."""
from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_cached(mtime: float):
    return joblib.load(config.BEST_MODEL_PATH)


def model_available() -> bool:
    return config.BEST_MODEL_PATH.exists()


def load_bundle() -> dict:
    if not model_available():
        raise ModelNotTrained("No trained model found. Run `python train_model.py` first.")
    return _load_cached(config.BEST_MODEL_PATH.stat().st_mtime)


def load_metrics() -> dict | None:
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else None


def load_metadata() -> dict | None:
    return json.loads(config.METADATA_PATH.read_text()) if config.METADATA_PATH.exists() else None


# ----------------------------------------------------------------------
def validate_row(row: dict, schema: dict) -> list[str]:
    """Return human-readable error strings (empty list = valid)."""
    errors = []
    for col, spec in schema["numeric"].items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"'{col}' is missing.")
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number (got {v!r}).")
            continue
        lo, hi = spec["min"], spec["max"]
        span = max(hi - lo, 1e-9)
        if fv < lo - 0.5 * span or fv > hi + 0.5 * span:
            errors.append(f"'{col}'={fv:g} is far outside the training range [{lo:g}, {hi:g}].")
    for col, spec in schema["categorical"].items():
        v = row.get(col)
        if v is None or str(v) not in spec["options"]:
            errors.append(f"'{col}' must be one of {spec['options']} (got {v!r}).")
    return errors


def _frame(rows: list[dict] | pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    df = df[schema["features"]]
    for c in schema["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in schema["categorical"]:
        df[c] = df[c].astype("object")
    return df


def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    b = load_bundle()
    return b["calibrated"].predict_proba(_frame(df, b["schema"]))[:, 1]


def fold_range(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Min/max probability across the calibrated CV sub-models: an *empirical
    spread* of the ensemble, not a formal confidence interval."""
    b = load_bundle()
    X = _frame(df, b["schema"])
    ps = np.column_stack([c.predict_proba(X)[:, 1] for c in b["calibrated"].calibrated_classifiers_])
    return ps.min(axis=1), ps.max(axis=1)


def predict_one(row: dict, threshold: float = 0.5) -> dict:
    b = load_bundle()
    errors = validate_row(row, b["schema"])
    if errors:
        return {"ok": False, "errors": errors}
    X = _frame([row], b["schema"])
    t0 = time.perf_counter()
    p = float(b["calibrated"].predict_proba(X)[0, 1])
    ms = (time.perf_counter() - t0) * 1000
    lo, hi = fold_range(X)
    return {"ok": True, "probability": p, "label": "Placed" if p >= threshold else "Not Placed",
            "threshold": threshold, "range_low": float(lo[0]), "range_high": float(hi[0]),
            "model": b["best_name"], "version": b["version"], "inference_ms": ms}


# ----------------------------------------------------------------------
def validate_batch(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split an uploaded frame into (valid, rejected_with_reasons, global_warnings)."""
    warns = []
    target = schema["target"]
    if target in df.columns:
        warns.append(f"Column '{target}' (the target) was found and IGNORED — it is never used as a feature.")
    extra = [c for c in df.columns if c not in schema["features"] and c != target]
    if extra:
        warns.append(f"Ignored non-feature columns: {extra}")
    missing_cols = [c for c in schema["features"] if c not in df.columns]
    if missing_cols:
        rej = df.copy()
        rej["rejection_reason"] = f"Missing required columns: {missing_cols}"
        return df.iloc[0:0], rej, warns + [f"Required columns missing: {missing_cols}"]
    reasons = []
    for i, r in df.iterrows():
        errs = validate_row(r.to_dict(), schema)
        reasons.append("; ".join(errs))
    reasons = pd.Series(reasons, index=df.index)
    valid = df[reasons == ""]
    rejected = df[reasons != ""].copy()
    rejected["rejection_reason"] = reasons[reasons != ""]
    return valid, rejected, warns


def predict_batch(valid: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    b = load_bundle()
    out = valid.copy()
    if len(valid) == 0:
        out["placement_probability"] = []
        out["predicted_outcome"] = []
        return out
    X = _frame(valid, b["schema"])
    p = b["calibrated"].predict_proba(X)[:, 1]
    lo, hi = fold_range(X)
    out["placement_probability"] = p.round(4)
    out["range_low"] = lo.round(4)
    out["range_high"] = hi.round(4)
    out["predicted_outcome"] = np.where(p >= threshold, "Placed", "Not Placed")
    return out
