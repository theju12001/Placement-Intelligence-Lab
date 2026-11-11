"""Deterministic local research assistant. Answers ONLY from computed results."""
from __future__ import annotations

import pandas as pd

from src.analytics import stats
from src.data import loader
from src.models import predict

TOPICS = [
    (("best model", "selected", "which model", "chosen"), "model"),
    (("accuracy", "auc", "f1", "precision", "recall", "brier", "metric", "performance"), "metrics"),
    (("important", "importance", "driver", "influence", "top feature", "shap"), "importance"),
    (("income", "parental", "socio", "economic", "internet"), "socio"),
    (("placement rate", "how many", "placed", "class", "balance", "dataset", "rows", "size"), "dataset"),
    (("missing", "quality", "duplicate"), "quality"),
    (("limit", "caveat", "synthetic", "trust", "reliab"), "limits"),
    (("fair", "bias", "gender", "protected"), "fairness"),
    (("cgpa", "backlog", "internship", "project", "certif", "aptitude", "ssc", "hsc", "academic"), "academic"),
]


def answer(question: str, importance: pd.DataFrame | None = None) -> str:
    q = question.lower()
    m, meta = predict.load_metrics(), predict.load_metadata()
    if not m or not meta:
        return "No trained model/metadata found yet. Run `python train_model.py`, then ask again."
    topic = next((t for kws, t in TOPICS if any(k in q for k in kws)), None)
    fin = m["final_calibrated"]["metrics"]
    ds, quality, schema = meta["dataset"], meta["quality"], meta["schema"]
    synth = " (note: this is SYNTHETIC demo data)" if ds.get("synthetic") else ""
    if topic == "model":
        return (f"The selected model is **{m['selected']}**, chosen by {m['selection_metric']}. "
                f"Ranking: {', '.join(m['ranking'])}. It is probability-calibrated with a sigmoid.{synth}")
    if topic == "metrics":
        return (f"Held-out test set (n={m['split']['test']}), calibrated {m['selected']}: ROC-AUC {fin['roc_auc']:.3f}, "
                f"PR-AUC {fin['pr_auc']:.3f}, accuracy {fin['accuracy']:.3f}, precision {fin['precision']:.3f}, "
                f"recall {fin['recall']:.3f}, F1 {fin['f1']:.3f}, Brier {fin['brier']:.3f}.{synth} "
                "These come from one split; they are estimates with sampling error.")
    if topic == "importance":
        if importance is None:
            return "Open the Model Lab page first to compute feature importance, then ask again."
        top = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in importance.head(5).itertuples())
        return f"Top model-contributing features: {top}. These reflect how the model uses features, not causal effects."
    if topic in ("socio", "academic"):
        df, info = loader.load_dataset()
        sc = loader.infer_schema(df, info["target"])
        res = stats.run_all_tests(df, sc, info["target"])
        want = sc["socioeconomic"] if topic == "socio" else sc["academic"]
        sub = res[res["feature"].isin(want)]
        for f in want:
            if f in q.replace(" ", "_") or f.split("_")[0] in q:
                sub = res[res["feature"] == f]
                break
        if sub.empty:
            return "The dataset contains no matching features for that question (nothing is invented)."
        lines = [f"- {r.feature}: {r.test}, p={r.p_value:.3g} (BH-adj {r.p_adj_BH:.3g}), "
                 f"{r.effect_name}={r.effect_size:.2f}" for r in sub.itertuples()]
        return "Statistical evidence (association only, not causation):\n" + "\n".join(lines)
    if topic == "dataset":
        c = quality["class_counts"]
        return (f"{quality['rows']} rows read; {c['placed']} placed vs {c['not_placed']} not placed "
                f"(rate {quality['placement_rate']:.1%}).{synth}")
    if topic == "quality":
        return (f"Duplicates: {quality['duplicates']}; missing cells: {quality['missing_cells']}; "
                f"columns with missing values: {quality['missing_by_column'] or 'none'}.")
    if topic == "fairness":
        return ("Protected attributes ("
                f"{', '.join(schema['audit_only']) or 'none in this dataset'}) are excluded from predictors and used only "
                "for subgroup audits. See the Responsible AI page; high accuracy does not imply fairness.")
    if topic == "limits":
        return ("Limitations: single split without external validation; probabilities are not guarantees; "
                "associations are not causal; socioeconomic coverage = " + (", ".join(schema["socioeconomic"]) or "none")
                + "." + synth)
    return ("I can answer questions about the selected model, metrics, feature importance, socioeconomic and academic "
            "evidence, dataset balance, data quality, fairness and limitations — using only locally computed results.")
