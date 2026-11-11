"""Explainability: SHAP when it works, permutation importance as honest fallback.

Every result carries a ``method`` field. A ranking is only labelled "SHAP"
if SHAP values were actually computed. SHAP/permutation values describe how the
*model* uses a feature (association), not a causal effect on placement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config
from src.models import predict

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAS_SHAP = False


def _background(n: int = 60) -> pd.DataFrame:
    ref = pd.read_csv(config.TRAIN_REF_PATH).drop(columns=["__y__"])
    return ref.sample(min(n, len(ref)), random_state=config.RANDOM_STATE)


def _typed(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    d = df[schema["features"]].copy()
    for c in schema["numeric"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in schema["categorical"]:
        d[c] = d[c].astype("object")
    return d


def _group_index(prep, schema) -> dict:
    """Map each original feature -> indices of its transformed columns."""
    names = list(prep.get_feature_names_out())
    groups = {}
    for f in schema["features"]:
        if f in schema["numeric"]:
            groups[f] = [i for i, n in enumerate(names) if n == f]
        else:
            groups[f] = [i for i, n in enumerate(names) if n.startswith(f + "_")]
    return groups


def _shap_values(bundle, X: pd.DataFrame):
    """SHAP on the transformed matrix, summed back to the original features
    (SHAP values are additive, so grouping one-hot columns is exact)."""
    schema, pipe = bundle["schema"], bundle["pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    bg_t = prep.transform(_typed(_background(), schema))
    Xt = prep.transform(_typed(X, schema))
    f = lambda A: model.predict_proba(A)[:, 1]
    expl = shap.Explainer(f, shap.maskers.Independent(bg_t, max_samples=len(bg_t)),
                          algorithm="permutation", seed=config.RANDOM_STATE)
    sv = expl(Xt, max_evals=2 * Xt.shape[1] + 1)
    groups = _group_index(prep, schema)
    vals = np.column_stack([np.asarray(sv.values)[:, idx].sum(axis=1) for idx in groups.values()])
    base = float(np.asarray(sv.base_values).reshape(-1)[0])
    return vals, base


def global_importance(n_eval: int = 150) -> dict:
    """Global importance on held-out rows. Returns DataFrame + method label."""
    b = predict.load_bundle()
    test = pd.read_csv(config.TEST_PRED_PATH)
    feats = b["schema"]["features"]
    Xte = test[feats].sample(min(n_eval, len(test)), random_state=config.RANDOM_STATE)
    if HAS_SHAP:
        try:
            vals, _ = _shap_values(b, Xte)
            df = pd.DataFrame({"feature": feats, "importance": np.abs(vals).mean(0)})
            return {"method": "SHAP (permutation explainer, mean |value|, on uncalibrated best pipeline)",
                    "table": df.sort_values("importance", ascending=False).reset_index(drop=True),
                    "values": vals, "X": Xte.reset_index(drop=True)}
        except Exception as exc:  # fall through to permutation importance
            note = f"SHAP failed ({type(exc).__name__}); showing permutation importance instead."
    else:
        note = "SHAP not installed; showing permutation importance instead."
    full = test.copy()
    Xf, yf = full[feats], full["y_true"]
    for c in b["schema"]["categorical"]:
        Xf[c] = Xf[c].astype("object")
    r = permutation_importance(b["pipeline"], Xf, yf, scoring="roc_auc", n_repeats=8,
                               random_state=config.RANDOM_STATE)
    df = pd.DataFrame({"feature": feats, "importance": r.importances_mean, "std": r.importances_std})
    return {"method": f"Permutation importance (drop in ROC-AUC on held-out data). {note}",
            "table": df.sort_values("importance", ascending=False).reset_index(drop=True),
            "values": None, "X": None}


def local_contributions(row: dict) -> dict:
    """Per-feature contributions to one prediction (probability units, vs baseline)."""
    b = predict.load_bundle()
    feats = b["schema"]["features"]
    x = pd.DataFrame([row])[feats]
    if HAS_SHAP:
        try:
            vals2d, base = _shap_values(b, x)
            vals = vals2d[0]
            df = pd.DataFrame({"feature": feats, "value": [row[c] for c in feats], "contribution": vals})
            return {"method": "SHAP (permutation explainer; probability units, uncalibrated best pipeline)",
                    "table": df.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True),
                    "base": base}
        except Exception:
            pass
    # fallback: occlusion — replace one feature at a time by its training median / mode
    ref = pd.read_csv(config.TRAIN_REF_PATH).drop(columns=["__y__"])
    f = lambda d: b["pipeline"].predict_proba(_typed(d, b["schema"]))[:, 1]
    base_p = float(f(x)[0])
    contribs = []
    for c in feats:
        alt = x.copy()
        alt[c] = ref[c].median() if c in b["schema"]["numeric"] else ref[c].mode().iloc[0]
        contribs.append(base_p - float(f(alt)[0]))
    df = pd.DataFrame({"feature": feats, "value": [row[c] for c in feats], "contribution": contribs})
    return {"method": "Occlusion (effect of replacing a feature with the training median/mode) — NOT SHAP",
            "table": df.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True),
            "base": None}


def strengths_and_gaps(row: dict, contrib_table: pd.DataFrame, top: int = 3) -> tuple[list[str], list[str]]:
    """Plain-language strengths/areas to improve from model contributions plus
    the student's position among *placed* students in the training reference."""
    ref = pd.read_csv(config.TRAIN_REF_PATH)
    placed = ref[ref["__y__"] == 1]
    s, g = [], []
    for _, r in contrib_table.iterrows():
        f, v, c = r["feature"], r["value"], r["contribution"]
        txt = f"{f.replace('_', ' ')} = {v}"
        if f in placed.columns and pd.api.types.is_numeric_dtype(placed[f]):
            med = placed[f].median()
            txt += f" (placed-cohort median {med:.4g})"
        (s if c > 0 else g).append((abs(c), txt + f" — model contribution {c:+.3f}"))
    s = [t for _, t in sorted(s, reverse=True)[:top]]
    g = [t for _, t in sorted(g, reverse=True)[:top]]
    return s, g
