"""Dataset loading, validation, target detection and schema inference.

Real data: drop a CSV into ``data/raw/``. The first CSV found is used.
No real dataset ships with this project. When ``data/raw`` is empty the
app falls back to a clearly labelled SYNTHETIC demo dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------
# Synthetic demo data (explicitly synthetic; for demonstration only)
# ----------------------------------------------------------------------
def generate_synthetic(n: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a SYNTHETIC placement dataset for demos and tests.

    The relationships below are *assumptions written by the developers*,
    including the socioeconomic effects. They are NOT findings from real
    students and results on this data must never be presented as real-world
    validation.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male"], n, p=[0.45, 0.55])
    income = rng.choice(
        ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"], n,
        p=[0.20, 0.28, 0.30, 0.15, 0.07])
    inc_rank = pd.Series(income).map(
        {"Low": 0, "Lower-Middle": 1, "Middle": 2, "Upper-Middle": 3, "High": 4}).to_numpy()
    parent_edu = np.array([
        rng.choice(["Below Graduate", "Graduate", "Postgraduate"],
                   p=[0.62 - 0.1 * r, 0.30 + 0.05 * r, 0.08 + 0.05 * r]) for r in inc_rank])
    internet = np.where(rng.random(n) < (0.55 + 0.1 * inc_rank), "Yes", "No")
    ability = rng.normal(0, 1, n)
    cgpa = np.clip(7.0 + 0.9 * ability + 0.08 * inc_rank + rng.normal(0, 0.45, n), 4.5, 10)
    ssc = np.clip(72 + 8 * ability + 1.2 * inc_rank + rng.normal(0, 6, n), 40, 99)
    hsc = np.clip(70 + 8 * ability + 1.2 * inc_rank + rng.normal(0, 6.5, n), 40, 99)
    aptitude = np.clip(60 + 11 * ability + rng.normal(0, 9, n), 15, 100)
    projects = np.clip(np.round(1.6 + 0.5 * ability + rng.normal(0, 1.1, n)), 0, 7).astype(int)
    internships = np.clip(np.round(0.7 + 0.25 * ability + 0.15 * inc_rank
                                   + rng.normal(0, 0.8, n)), 0, 4).astype(int)
    certs = np.clip(np.round(1.5 + 0.4 * ability + 0.2 * (internet == "Yes")
                             + rng.normal(0, 1.2, n)), 0, 8).astype(int)
    backlogs = np.clip(np.round(np.maximum(0, -0.5 * ability + rng.normal(0, 0.9, n))), 0, 5).astype(int)
    logit = (-7.6 + 0.62 * cgpa + 0.018 * aptitude + 0.22 * projects + 0.45 * internships
             + 0.12 * certs - 0.55 * backlogs + 0.012 * ssc
             + 0.10 * inc_rank + 0.25 * (internet == "Yes"))
    logit += rng.normal(0, 0.9, n)
    placed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "student_id": [f"SYN{i:05d}" for i in range(n)],
        "cgpa": cgpa.round(2), "ssc_percentage": ssc.round(1), "hsc_percentage": hsc.round(1),
        "aptitude_score": aptitude.round(1), "projects": projects,
        "internships": internships, "certifications": certs, "backlogs": backlogs,
        "household_income_band": income, "parental_education": parent_edu,
        "internet_access": internet, "gender": gender,
        "placed": np.where(placed == 1, "Placed", "Not Placed"),
    })
    # realistic data-quality issues so validation code has real work to do
    miss = rng.random(n) < 0.02
    df.loc[miss, "aptitude_score"] = np.nan
    miss2 = rng.random(n) < 0.015
    df.loc[miss2, "parental_education"] = np.nan
    return df


def ensure_synthetic_csv() -> Path:
    if not config.SYNTHETIC_CSV.exists():
        generate_synthetic().to_csv(config.SYNTHETIC_CSV, index=False)
    return config.SYNTHETIC_CSV


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def find_dataset_path() -> tuple[Path, bool]:
    """Return (path, is_synthetic). Real CSVs in data/raw take priority."""
    csvs = sorted(p for p in config.DATA_RAW.glob("*.csv"))
    if csvs:
        return csvs[0], False
    return ensure_synthetic_csv(), True


def detect_target(df: pd.DataFrame) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in config.TARGET_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        "Could not detect a placement target column. Rename it to one of "
        f"{config.TARGET_CANDIDATES} or add data/raw/column_map.json with "
        '{"target": "<your column>"}.')


def encode_target(series: pd.Series) -> pd.Series:
    """Map the real target values to 0/1 without silently guessing."""
    s = series.astype(str).str.strip().str.lower()
    uniq = set(s.dropna().unique())
    if not uniq.issubset({"nan"} | config.POSITIVE_LABELS | {
            "not placed", "no", "0", "false", "unplaced", "not selected", "n", "not hired"}):
        raise ValueError(f"Unrecognised target labels: {sorted(uniq)}. "
                         "Use Placed/Not Placed, Yes/No or 1/0.")
    return s.isin(config.POSITIVE_LABELS).astype(int).where(s != "nan")


def load_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load the dataset and return (df, info).

    ``info`` documents provenance: source path, synthetic flag, target column.
    """
    synthetic = False
    if path is None:
        path, synthetic = find_dataset_path()
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    map_file = config.DATA_RAW / "column_map.json"
    target = None
    if map_file.exists():
        cm = json.loads(map_file.read_text())
        target = cm.get("target")
        if cm.get("rename"):
            df = df.rename(columns=cm["rename"])
    target = target or detect_target(df)
    info = {"path": str(path), "file": path.name, "synthetic": synthetic,
            "target": target, "n_rows": int(len(df)), "n_cols": int(df.shape[1])}
    return df, info


# ----------------------------------------------------------------------
# Schema inference & validation
# ----------------------------------------------------------------------
def infer_schema(df: pd.DataFrame, target: str) -> dict:
    """Infer feature schema from the *actual* dataset (no invented fields)."""
    numeric, categorical, audit_only, dropped = {}, {}, [], []
    for col in df.columns:
        low = col.lower()
        if col == target:
            continue
        if low in config.ID_LIKE or df[col].nunique(dropna=True) >= 0.95 * len(df) and not pd.api.types.is_numeric_dtype(df[col]):
            dropped.append(col)
            continue
        if low in config.SENSITIVE_COLUMNS:
            audit_only.append(col)
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if clean.empty:
                dropped.append(col)
                continue
            numeric[col] = {
                "min": float(clean.min()), "max": float(clean.max()),
                "median": float(clean.median()), "mean": float(clean.mean()),
                "integer": bool(np.allclose(clean, clean.round())),
            }
        else:
            opts = sorted(s.dropna().astype(str).unique().tolist())
            if len(opts) > 30:
                dropped.append(col)
                continue
            categorical[col] = {"options": opts}
    socio = [c for c in list(numeric) + list(categorical)
             if any(h in c.lower() for h in config.SOCIOECONOMIC_HINTS)]
    return {"numeric": numeric, "categorical": categorical, "audit_only": audit_only,
            "dropped": dropped, "socioeconomic": socio,
            "academic": [c for c in list(numeric) + list(categorical) if c not in socio],
            "features": list(numeric) + list(categorical), "target": target}


def validate_dataset(df: pd.DataFrame, target: str) -> dict:
    """Dataset-level quality report (counts only, nothing is modified)."""
    y = encode_target(df[target])
    counts = y.value_counts(dropna=True)
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_target": int(y.isna().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "class_counts": {"placed": int(counts.get(1, 0)), "not_placed": int(counts.get(0, 0))},
        "placement_rate": float(y.mean()) if y.notna().any() else float("nan"),
        "imbalance_ratio": float(max(counts) / max(1, min(counts))) if len(counts) == 2 else float("inf"),
        "single_class": bool(len(counts) < 2),
    }


def prepare_xy(df: pd.DataFrame, schema: dict):
    """Return cleaned X (features only), y (0/1). Drops rows with missing target
    and exact duplicate rows; the counts are logged by the caller."""
    d = df.copy()
    d["__y__"] = encode_target(d[schema["target"]])
    n0 = len(d)
    d = d.dropna(subset=["__y__"]).drop_duplicates()
    X = d[schema["features"]].copy()
    for c in schema["categorical"]:
        X[c] = X[c].astype("object")
    y = d["__y__"].astype(int)
    audit = d[schema["audit_only"]].copy() if schema["audit_only"] else pd.DataFrame(index=d.index)
    return X, y, audit, {"rows_in": int(n0), "rows_used": int(len(d))}
