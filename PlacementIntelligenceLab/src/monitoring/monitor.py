"""Lightweight statistical monitoring (PSI, KS, chi-square, data quality).

Evidently is not used; this module is a self-contained statistical fallback.
It compares an uploaded *current* dataset against the training reference. It
does NOT monitor a live production system.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.models import predict

DEFAULT_THRESHOLDS = {"psi": 0.2, "ks_p": 0.01, "missing_delta": 0.05, "perf_drop_auc": 0.05}


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    else:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        c = cur.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, schema: dict, thr: dict) -> pd.DataFrame:
    rows = []
    for c in schema["features"]:
        if c not in cur.columns:
            rows.append({"feature": c, "type": "n/a", "psi": np.nan, "test": "missing column",
                         "p_value": np.nan, "drift": True, "note": "Column absent in current data"})
            continue
        v = psi(ref[c], cur[c])
        if c in schema["numeric"]:
            a, b = pd.to_numeric(ref[c], errors="coerce").dropna(), pd.to_numeric(cur[c], errors="coerce").dropna()
            p = stats.ks_2samp(a, b)[1] if len(a) and len(b) else np.nan
            test, typ = "KS two-sample", "numeric"
        else:
            cats = sorted(set(ref[c].dropna().astype(str)) | set(cur[c].dropna().astype(str)))
            tab = np.array([ref[c].astype(str).value_counts().reindex(cats).fillna(0),
                            cur[c].astype(str).value_counts().reindex(cats).fillna(0)])
            tab = tab[:, tab.sum(0) > 0]
            p = stats.chi2_contingency(tab)[1] if tab.shape[1] > 1 else np.nan
            test, typ = "Chi-square", "categorical"
        drift = bool((v > thr["psi"]) or (p == p and p < thr["ks_p"] and v > thr["psi"] / 2))
        rows.append({"feature": c, "type": typ, "psi": round(v, 4), "test": test,
                     "p_value": None if p != p else round(float(p), 6), "drift": drift, "note": ""})
    return pd.DataFrame(rows)


def data_quality(ref: pd.DataFrame, cur: pd.DataFrame, schema: dict, thr: dict) -> pd.DataFrame:
    rows = []
    for c in schema["features"]:
        if c not in cur.columns:
            continue
        mr, mc = ref[c].isna().mean(), cur[c].isna().mean()
        rows.append({"feature": c, "missing_ref_%": round(mr * 100, 2), "missing_cur_%": round(mc * 100, 2),
                     "delta_pp": round((mc - mr) * 100, 2), "flag": bool(mc - mr > thr["missing_delta"])})
    return pd.DataFrame(rows)


def run_monitor(cur: pd.DataFrame, thresholds: dict | None = None, label_col: str | None = None) -> dict:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    bundle = predict.load_bundle()
    schema = bundle["schema"]
    ref = pd.read_csv(config.TRAIN_REF_PATH).drop(columns=["__y__"])
    drift = feature_drift(ref, cur, schema, thr)
    quality = data_quality(ref, cur, schema, thr)
    out = {"n_current": int(len(cur)), "n_reference": int(len(ref)), "thresholds": thr,
           "drift_table": drift, "quality_table": quality}
    # prediction drift (only rows with all required columns)
    if all(c in cur.columns for c in schema["features"]):
        valid, rejected, _ = predict.validate_batch(cur, schema)
        out["n_rejected"] = int(len(rejected))
        if len(valid):
            ref_p = predict.predict_proba_df(ref)
            cur_p = predict.predict_proba_df(valid)
            ks = stats.ks_2samp(ref_p, cur_p)
            out["prediction_drift"] = {"ref_mean": float(ref_p.mean()), "cur_mean": float(cur_p.mean()),
                                       "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                                       "psi": psi(pd.Series(ref_p), pd.Series(cur_p)),
                                       "ref_proba": ref_p, "cur_proba": cur_p}
            if label_col and label_col in valid.columns:
                from src.data import loader
                from src.models.train import safe_metrics
                y = loader.encode_target(valid[label_col]).dropna()
                if y.nunique() == 2:
                    m = safe_metrics(y.to_numpy(), cur_p[valid.index.get_indexer(y.index)])
                    base = (predict.load_metrics() or {}).get("final_calibrated", {}).get("metrics", {})
                    out["performance"] = {"current": m, "reference_test": base,
                                          "auc_drop": (base.get("roc_auc", np.nan) - m["roc_auc"])}
                else:
                    out["performance_note"] = "Labels in the uploaded data contain a single class; AUC undefined."
    else:
        out["prediction_note"] = "Prediction drift skipped: required feature columns are missing."
    out["summary"] = {"drifted_features": int(drift["drift"].sum()), "n_features": int(len(drift)),
                      "quality_flags": int(quality["flag"].sum()) if len(quality) else 0}
    _append_history(out)
    return out


def _append_history(res: dict) -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_current": res["n_current"], **res["summary"],
           "pred_psi": (res.get("prediction_drift") or {}).get("psi"),
           "auc": ((res.get("performance") or {}).get("current") or {}).get("roc_auc")}
    config.MONITOR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MONITOR_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_history() -> pd.DataFrame:
    if not config.MONITOR_HISTORY.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in config.MONITOR_HISTORY.read_text().splitlines() if l.strip()])
