"""Panel F — Experiment tracking (real runs only)."""
import json

import plotly.express as px
import streamlit as st

from src.analytics import viz3d
from src.tracking import tracker
from src.utils import ui


def render():
    ui.page_header("Experiments", "Every training run recorded by this project — no fabricated runs.")
    st.markdown(ui.chip(f"Active backend: {tracker.active_backend()}", "ok" if tracker.HAS_MLFLOW else "warn"),
                unsafe_allow_html=True)
    runs = tracker.load_runs()
    if runs.empty:
        st.info("No runs yet. Run `python train_model.py` (or Settings → Retrain) to create the first recorded run.")
        return
    stages = runs.copy()
    ui.tiles([("Runs", str(len(runs)), "recorded"), ("Models", str(runs["model"].nunique()), "distinct"),
              ("Latest version", runs.iloc[-1]["model_version"], runs.iloc[-1]["timestamp"])])
    cols = ["run_name", "model", "model_version", "timestamp", "train_seconds"] + [c for c in runs if c.startswith("metric_")]
    st.dataframe(runs[cols].sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
    metric_cols = [c for c in runs if c.startswith("metric_") and runs[c].notna().any()]
    a, b = st.columns(2)
    m1 = a.selectbox("Metric", metric_cols, index=metric_cols.index("metric_roc_auc") if "metric_roc_auc" in metric_cols else 0)
    fig = px.scatter(runs, x="timestamp", y=m1, color="model", hover_data=["run_name", "model_version"])
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(viz3d.style(fig, 340, f"{m1} over runs"), width="stretch")
    pick = st.selectbox("Inspect a run", runs["run_id"] + " — " + runs["run_name"])
    row = runs[runs["run_id"] == pick.split(" — ")[0]].iloc[0]
    st.json({"params": json.loads(row["params"]), "dataset": json.loads(row["dataset"]), "artifact": row["artifact"]})
    st.download_button("Download runs CSV", runs.to_csv(index=False), "experiment_runs.csv", "text/csv")
    if not tracker.HAS_MLFLOW:
        st.caption("MLflow is optional: `pip install mlflow` then retrain to mirror runs into a local `mlruns/` store "
                   "(view with `mlflow ui`). Until then the JSONL log above is the source of truth.")
