"""Panel A — Student prediction."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.analytics import viz3d
from src.explainability import explain
from src.models import predict
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Student Prediction", "Estimate a placement probability from academic and socioeconomic inputs.")
    if not ui.require_model(c):
        return
    ui.synthetic_banner(c)
    schema = c["schema"]
    left, right = st.columns([5, 6])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        ui.panel("Student profile")
        with st.form("predict_form"):
            vals = ui.input_form_fields(schema, "pred", cols=2)
            thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                            help="Probability above which the label is 'Placed'.")
            go_ = st.form_submit_button("Predict placement", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    if go_:
        st.session_state["last_pred"] = (vals, thr)
    with right:
        if "last_pred" not in st.session_state:
            st.markdown("<div class='glass'><div class='panel-sub'>Fill in the profile and press "
                        "<b>Predict placement</b>.</div></div>", unsafe_allow_html=True)
            return
        vals, thr = st.session_state["last_pred"]
        res = predict.predict_one(vals, thr)
        if not res["ok"]:
            for e in res["errors"]:
                st.error(e)
            return
        p = res["probability"]
        cls = "pos" if p >= thr else "neg"
        st.markdown(f"<div class='result {cls}'><div class='panel-title'>Predicted outcome</div>"
                    f"<div class='big'>{res['label']}</div>"
                    f"<div>Estimated placement probability: <b>{p:.1%}</b> "
                    f"(model spread across CV folds: {res['range_low']:.1%}–{res['range_high']:.1%})</div></div>",
                    unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": config.COLORS["cyan"]},
                                            "threshold": {"line": {"color": config.COLORS["magenta"], "width": 4},
                                                          "value": thr * 100},
                                            "bgcolor": "rgba(255,255,255,0.05)"}))
        st.plotly_chart(viz3d.style(fig, 230), width="stretch")
        ui.tiles([("Model", res["model"], "calibrated (sigmoid)"), ("Version", res["version"], ""),
                  ("Inference", f"{res['inference_ms']:.0f} ms", "single row")])
        with st.spinner("Explaining…"):
            loc = explain.local_contributions(vals)
        t = loc["table"].copy()
        bar = go.Figure(go.Bar(x=t["contribution"][::-1], y=t["feature"][::-1], orientation="h",
                               marker_color=[config.COLORS["cyan"] if v > 0 else config.COLORS["magenta"]
                                             for v in t["contribution"][::-1]]))
        bar.update_layout(xaxis_title="contribution to probability (vs. average student)")
        st.caption(loc["method"])
        st.plotly_chart(viz3d.style(bar, 320, "Main contributing factors"), width="stretch")
        strengths, gaps = explain.strengths_and_gaps(vals, t)
        a, b = st.columns(2)
        a.markdown("**Student strengths (model view)**\n\n" + ("\n".join(f"- {s}" for s in strengths) or "- none identified"))
        b.markdown("**Possible areas for improvement**\n\n" + ("\n".join(f"- {g}" for g in gaps) or "- none identified"))
        st.markdown("<div class='disclaimer'>Probability is a calibrated model estimate, not a promise. Contributions "
                    "describe associations learned from the training data, not causes. The range shown is the spread "
                    "of the cross-validated sub-models, not a formal confidence interval. Use for educational "
                    "research only.</div>", unsafe_allow_html=True)
