"""Settings: dataset status, retraining, data setup, About."""
import streamlit as st

import config
from src.data import loader
from src.tracking import tracker
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Settings", "Dataset, training and project information.")
    t1, t2, t3 = st.tabs(["Data & training", "Environment", "About"])
    with t1:
        info = c["info"]
        st.markdown(f"**Active dataset:** `{info['file']}` — {info['n_rows']} rows, target `{info['target']}` · "
                    + ("**SYNTHETIC demo data**" if info["synthetic"] else "real data"))
        st.markdown("To use your own data, place a CSV in `data/raw/` (target column named e.g. `placed` or `placement_status`; "
                    "optional `data/raw/column_map.json` with `{\"target\": \"...\", \"rename\": {...}}`), then retrain. "
                    "Sensitive columns (gender, caste, religion, …) are automatically kept out of the predictors.")
        up = st.file_uploader("Upload a dataset CSV into data/raw/", type=["csv"])
        if up is not None and st.button("Save uploaded CSV to data/raw/"):
            (config.DATA_RAW / up.name).write_bytes(up.getvalue())
            st.cache_data.clear()
            st.success(f"Saved {up.name}. Now press Retrain.")
        if st.button("Retrain all models", width="stretch"):
            from src.models.train import train_all
            with st.spinner("Training… this takes under a minute for small datasets."):
                try:
                    df, info2 = loader.load_dataset()
                    train_all(df, info2, verbose=False)
                    st.cache_data.clear()
                    st.success("Training finished. Reload pages to see the new model.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    with t2:
        import importlib
        rows = []
        for name in ["streamlit", "sklearn", "plotly", "scipy", "statsmodels", "shap", "mlflow", "reportlab",
                     "xgboost", "lightgbm", "catboost", "evidently"]:
            try:
                m = importlib.import_module(name)
                rows.append((name, getattr(m, "__version__", "installed")))
            except Exception:
                rows.append((name, "not installed (optional)" if name in ("shap", "mlflow", "reportlab", "xgboost", "lightgbm", "catboost", "evidently") else "MISSING"))
        st.dataframe(rows, column_config={0: "package", 1: "status"}, width="stretch", hide_index=True)
        st.markdown(f"**Experiment tracking backend:** {tracker.active_backend()}")
    with t3:
        st.markdown(f"### {config.APP_NAME}  \n*{config.TAGLINE}*")
        st.markdown(f"**Project title:** {config.ACADEMIC_TITLE}")
        st.markdown(f"**Executed by:** {' and '.join(config.EXECUTORS)}  \n**Version:** {config.APP_VERSION}")
        st.markdown("This platform supports educational research on how academic and socioeconomic factors relate to "
                    "student placement outcomes. It is not a hiring tool and must not be the sole basis for any decision.")
