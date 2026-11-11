"""Batch CSV analysis."""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import viz3d
from src.models import predict
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Batch Analysis", "Upload a CSV of students and score every valid record.")
    if not ui.require_model(c):
        return
    schema = c["schema"]
    st.markdown(f"**Required columns:** `{', '.join(schema['features'])}`. A target column, if present, is ignored.")
    template = pd.DataFrame([{f: (schema["numeric"][f]["median"] if f in schema["numeric"]
                                  else schema["categorical"][f]["options"][0]) for f in schema["features"]}])
    st.download_button("Download CSV template", template.to_csv(index=False), "batch_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="batch_thr")
    if up is None:
        st.info("Upload a CSV to begin. No data leaves this machine.")
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    df.columns = [x.strip() for x in df.columns]
    st.markdown(f"**Preview** — {len(df)} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), width="stretch")
    valid, rejected, warns = predict.validate_batch(df, schema)
    for w in warns:
        st.warning(w)
    ui.tiles([("Uploaded", str(len(df)), "records"), ("Processed", str(len(valid)), "valid"),
              ("Rejected", str(len(rejected)), "see reasons below")])
    if len(rejected):
        with st.expander(f"Rejected records ({len(rejected)}) — never silently dropped", expanded=True):
            st.dataframe(rejected, width="stretch")
            st.download_button("Download rejected records", rejected.to_csv(index=False), "rejected.csv", "text/csv")
    if not len(valid):
        st.error("No valid records to score.")
        return
    out = predict.predict_batch(valid, thr)
    a, b = st.columns(2)
    with a:
        fig = px.histogram(out, x="placement_probability", nbins=20, color="predicted_outcome",
                           color_discrete_map={"Placed": "#22d3ee", "Not Placed": "#ec4899"})
        st.plotly_chart(viz3d.style(fig, 320, "Probability distribution"), width="stretch")
    with b:
        cnt = out["predicted_outcome"].value_counts().reset_index()
        fig = px.pie(cnt, names="predicted_outcome", values="count", hole=.55,
                     color="predicted_outcome", color_discrete_map={"Placed": "#22d3ee", "Not Placed": "#ec4899"})
        st.plotly_chart(viz3d.style(fig, 320, "Predicted outcomes"), width="stretch")
    f1, f2 = st.columns(2)
    outcome = f1.multiselect("Filter outcome", ["Placed", "Not Placed"], ["Placed", "Not Placed"])
    lo, hi = f2.slider("Probability range", 0.0, 1.0, (0.0, 1.0), 0.01)
    flt = out[out["predicted_outcome"].isin(outcome) & out["placement_probability"].between(lo, hi)]
    st.dataframe(flt, width="stretch")
    st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
    summary = (f"Batch report\nUploaded: {len(df)}\nProcessed: {len(valid)}\nRejected: {len(rejected)}\n"
               f"Predicted placed: {(out['predicted_outcome'] == 'Placed').sum()}\n"
               f"Mean probability: {out['placement_probability'].mean():.3f}\nThreshold: {thr}\n"
               f"Model: {c['meta']['selected_model']} {c['meta']['version']}\n"
               "Predictions are model estimates, not guarantees.\n")
    st.download_button("Download batch report (txt)", summary, "batch_report.txt")
