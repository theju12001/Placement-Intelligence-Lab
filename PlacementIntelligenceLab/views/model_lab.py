"""Panel C — Model performance, comparison, explainability."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.analytics import viz3d
from src.reporting import reports
from src.utils import ui

METRIC_HELP = {
    "Accuracy": "Share of correct labels at the 0.5 threshold; misleading when classes are imbalanced.",
    "Precision": "Of students predicted placed, the share who were. Depends on the threshold.",
    "Recall": "Of students actually placed, the share the model found. Depends on the threshold.",
    "F1": "Harmonic mean of precision and recall; ignores true negatives.",
    "ROC-AUC": "Chance a random placed student scores above a random unplaced one; threshold-free, but optimistic when classes are skewed.",
    "PR-AUC": "Area under precision–recall curve; focuses on the positive class, sensitive to the placement rate.",
    "Brier": "Mean squared error of probabilities (lower is better); measures calibration + sharpness.",
}


def render():
    c = ui.ctx()
    ui.page_header("Model Lab", "Compare models, inspect curves, calibration and explanations.")
    if not ui.require_model(c):
        return
    ui.synthetic_banner(c)
    m = c["metrics"]
    comp = reports.comparison_table(m)
    sel = m["selected"]
    fin = m["final_calibrated"]["metrics"]
    ui.tiles([("Selected", sel, m["selection_metric"]), ("ROC-AUC", f"{fin['roc_auc']:.3f}", "calibrated, test"),
              ("PR-AUC", f"{fin['pr_auc']:.3f}", "test"), ("F1", f"{fin['f1']:.3f}", "test"),
              ("Brier", f"{fin['brier']:.3f}", "lower is better")])
    t1, t2, t3, t4, t5 = st.tabs(["Ranking", "ROC & PR", "Confusion & calibration", "Explainability", "Metric guide"])
    with t1:
        st.dataframe(comp.round(4), width="stretch", hide_index=True)
        melt = comp.melt(id_vars="model", value_vars=["cv_roc_auc", "test_roc_auc", "test_pr_auc", "f1"])
        fig = px.bar(melt, x="model", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[config.COLORS[k] for k in ("cyan", "purple", "blue", "magenta")])
        st.plotly_chart(viz3d.style(fig, 380, "Model comparison"), width="stretch")
        st.caption("Selection used cross-validated ROC-AUC on the training split only; test metrics are held out. "
                   "Differences of ~0.01 AUC are within noise — see the CV std column.")
        st.download_button("Download ranking CSV", comp.to_csv(index=False), "model_ranking.csv", "text/csv")
    with t2:
        a, b = st.columns(2)
        a.plotly_chart(viz3d.style(viz3d.roc_fig(m["models"], sel), 380, "ROC curves"), width="stretch")
        b.plotly_chart(viz3d.style(viz3d.pr_fig(m["models"], sel), 380, "Precision–recall curves"), width="stretch")
    with t3:
        a, b = st.columns(2)
        pick = a.selectbox("Model", list(m["models"]), index=list(m["models"]).index(sel))
        a.plotly_chart(viz3d.confusion_fig(m["models"][pick]["curves"]["confusion"], f"{pick} (uncalibrated)"),
                       width="stretch")
        b.plotly_chart(viz3d.style(viz3d.calibration_fig(m["final_calibrated"]["curves"], f"{sel} (calibrated)"),
                                   340, "Calibration — selected model"), width="stretch")
        st.caption("Closer to the diagonal = predicted probabilities match observed rates. 10 quantile bins on a small test set are noisy.")
    with t4:
        imp = ui.cached_importance()
        st.caption(imp["method"])
        st.plotly_chart(viz3d.importance_3d(imp["table"], imp["method"]), width="stretch")
        st.dataframe(imp["table"].round(4), width="stretch", hide_index=True)
        st.markdown("<div class='disclaimer'>Importance = how the <i>model</i> uses a feature (association). It is not "
                    "a causal effect, and correlated features share credit.</div>", unsafe_allow_html=True)
    with t5:
        for k, v in METRIC_HELP.items():
            st.markdown(f"**{k}** — {v}")
