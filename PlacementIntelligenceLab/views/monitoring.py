"""Panel E — Monitoring of uploaded data against the training reference."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.analytics import viz3d
from src.monitoring import monitor
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Model Monitoring", "Drift and data-quality checks for an uploaded dataset vs the training reference.")
    st.markdown("<div class='disclaimer'>No live production system is connected. Monitoring runs only on data you upload "
                "here and compares it with the training reference.</div>", unsafe_allow_html=True)
    if not ui.require_model(c):
        return
    with st.expander("Thresholds", expanded=False):
        a, b, cc = st.columns(3)
        thr = {"psi": a.number_input("PSI drift threshold", 0.05, 1.0, 0.2, 0.05),
               "ks_p": b.number_input("KS/chi² p-value", 0.0001, 0.2, 0.01, format="%.4f"),
               "missing_delta": cc.number_input("Missing-rate increase (fraction)", 0.0, 0.5, 0.05, 0.01)}
    up = st.file_uploader("Upload current/new student data (CSV)", type=["csv"], key="mon_up")
    label_col = st.text_input("Label column (optional, enables performance tracking)", c["schema"]["target"])
    if up is not None:
        cur = pd.read_csv(up)
        cur.columns = [x.strip() for x in cur.columns]
        res = monitor.run_monitor(cur, thr, label_col if label_col in cur.columns else None)
        st.session_state["mon_res"] = res
    res = st.session_state.get("mon_res")
    if res is None:
        st.info("Upload a CSV with the same feature columns to run drift and quality checks. "
                "Tip: export part of your dataset from `data/` and edit values to see how drift is flagged.")
    else:
        s = res["summary"]
        ui.tiles([("Rows compared", str(res["n_current"]), f"vs {res['n_reference']} reference"),
                  ("Drifted features", f"{s['drifted_features']}/{s['n_features']}", f"PSI > {res['thresholds']['psi']}"),
                  ("Quality flags", str(s["quality_flags"]), "missing-rate increases")])
        d = res["drift_table"]
        fig = px.bar(d, x="feature", y="psi", color="drift", color_discrete_map={True: "#ec4899", False: "#22d3ee"})
        fig.add_hline(y=res["thresholds"]["psi"], line_dash="dash", line_color="#fbbf24")
        st.plotly_chart(viz3d.style(fig, 340, "Feature drift (PSI)"), width="stretch")
        st.dataframe(d, width="stretch", hide_index=True)
        st.dataframe(res["quality_table"], width="stretch", hide_index=True)
        pdr = res.get("prediction_drift")
        if pdr:
            ui.tiles([("Mean prob (ref)", f"{pdr['ref_mean']:.3f}", ""), ("Mean prob (current)", f"{pdr['cur_mean']:.3f}", ""),
                      ("Prediction PSI", f"{pdr['psi']:.3f}", f"KS p={pdr['ks_p']:.3g}")])
            h = go.Figure()
            h.add_histogram(x=pdr["ref_proba"], name="Reference", opacity=.6, marker_color=config.COLORS["blue"])
            h.add_histogram(x=pdr["cur_proba"], name="Current", opacity=.6, marker_color=config.COLORS["magenta"])
            h.update_layout(barmode="overlay", xaxis_title="predicted probability")
            st.plotly_chart(viz3d.style(h, 320, "Prediction drift"), width="stretch")
        elif res.get("prediction_note"):
            st.warning(res["prediction_note"])
        perf = res.get("performance")
        if perf:
            cur_m, ref_m = perf["current"], perf["reference_test"]
            ui.tiles([("AUC (uploaded)", f"{cur_m['roc_auc']:.3f}", f"reference test {ref_m.get('roc_auc', float('nan')):.3f}"),
                      ("Accuracy", f"{cur_m['accuracy']:.3f}", f"reference {ref_m.get('accuracy', float('nan')):.3f}"),
                      ("AUC drop", f"{perf['auc_drop']:+.3f}", "positive = worse")])
        elif res.get("performance_note"):
            st.warning(res["performance_note"])
        else:
            st.caption("No usable label column found, so model performance is not tracked for this upload.")
    hist = monitor.load_history()
    st.markdown("**Monitoring history**")
    if hist.empty:
        st.caption("No monitoring runs recorded yet.")
    else:
        st.dataframe(hist, width="stretch", hide_index=True)
        if len(hist) > 1:
            st.plotly_chart(viz3d.style(px.line(hist, x="timestamp", y="drifted_features", markers=True), 260,
                                        "Drifted features per check"), width="stretch")
    st.caption("Evidently is not bundled; this page uses PSI, KS and chi-square tests implemented in "
               "`src/monitoring/monitor.py`.")
