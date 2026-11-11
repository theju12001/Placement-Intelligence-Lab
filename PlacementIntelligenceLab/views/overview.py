"""Overview: hero, live metrics, 3D analytics studio, mini panels."""
import pandas as pd
import streamlit as st

import config
from src.analytics import viz3d
from src.utils import ui


def render():
    c = ui.ctx()
    df, schema, info = c["df"], c["schema"], c["info"]
    from src.data import loader
    q = loader.validate_dataset(df, schema["target"])

    left, right = st.columns([5, 6])
    with left:
        st.markdown("<div class='glass'><div class='panel-title'>"
                    f"{config.ACADEMIC_TITLE}</div>"
                    "<div class='hero-title'>From Education<br>To Opportunities.</div>"
                    "<div class='hero-sub'>AI-Powered Insights for Student Placement Success</div>"
                    "<div class='hero-desc'>Use statistical modeling, machine learning, and responsible analytics to "
                    "investigate student placement outcomes and understand the influence of academic and "
                    "socioeconomic factors.</div></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 1])
        goto = st.session_state["goto"]
        b1.button("Try Prediction", on_click=goto, args=("Predict",), width="stretch")
        b2.button("Explore Insights", on_click=goto, args=("Statistical Evidence",), width="stretch")
    with right:
        st.plotly_chart(viz3d.hero_network(), width="stretch", config={"displayModeBar": False})

    ui.synthetic_banner(c)
    items = [("Students (rows)", f"{q['rows']:,}", f"{info['file']}"),
             ("Placement rate", ui.fmt_pct(q["placement_rate"]), f"{q['class_counts']['placed']} placed"),
             ("Features used", str(len(schema["features"])), f"{len(schema['socioeconomic'])} socioeconomic")]
    if c["trained"]:
        fin = c["metrics"]["final_calibrated"]["metrics"]
        items += [("Test ROC-AUC", f"{fin['roc_auc']:.3f}", c["metrics"]["selected"]),
                  ("Test accuracy", ui.fmt_pct(fin["accuracy"]), "held-out set"),
                  ("Models compared", str(len(c["metrics"]["models"])), "measured, not assumed")]
    else:
        items.append(("Model", "not trained", "run train_model.py"))
    ui.tiles(items)

    # ---- Panel B: 3D analytics studio ----
    st.markdown("<div class='panel-title'>3D analytics studio</div>", unsafe_allow_html=True)
    num = list(schema["numeric"])
    tabs = st.tabs(["Feature space", "Rate surface", "Importance", "Probabilities", "Cohorts", "Scenario"])
    with tabs[0]:
        a, b, cc = st.columns(3)
        x = a.selectbox("X", num, 0, key="fs_x")
        y = b.selectbox("Y", num, min(1, len(num) - 1), key="fs_y")
        z = cc.selectbox("Z", num, min(3, len(num) - 1), key="fs_z")
        st.plotly_chart(viz3d.feature_space(df, schema["target"], x, y, z), width="stretch")
    with tabs[1]:
        a, b = st.columns(2)
        x = a.selectbox("X axis", num, 0, key="rs_x")
        y = b.selectbox("Y axis", num, min(3, len(num) - 1), key="rs_y")
        if x != y:
            st.plotly_chart(viz3d.rate_surface(df, schema["target"], x, y), width="stretch")
        else:
            st.info("Choose two different features.")
    with tabs[2]:
        if ui.require_model(c):
            imp = ui.cached_importance()
            st.caption(imp["method"])
            st.plotly_chart(viz3d.importance_scatter3d(imp["table"], schema), width="stretch")
    with tabs[3]:
        if ui.require_model(c):
            test = pd.read_csv(config.TEST_PRED_PATH)
            st.plotly_chart(viz3d.probability_distribution_3d(test, st.selectbox("Feature on X", num, key="pd_x")),
                            width="stretch")
    with tabs[4]:
        if len(num) >= 2 and schema["categorical"]:
            a, b = st.columns(2)
            x = a.selectbox("Feature A", num, 0, key="co_x")
            y = b.selectbox("Feature B", num, min(1, len(num) - 1), key="co_y")
            st.caption(f"Cohorts = categories of '{list(schema['categorical'])[0]}'. Small cohorts are noisy.")
            st.plotly_chart(viz3d.cohort_comparison_3d(df, schema["target"], schema, x, y), width="stretch")
        else:
            st.info("Needs at least one categorical feature.")
    with tabs[5]:
        st.info("Open **Scenario Simulator** to compare baseline vs scenario predictions in 3D "
                "(uses the real trained model).")
        st.button("Open Scenario Simulator", on_click=st.session_state["goto"], args=("Scenario Simulator",))

    # ---- mini panels (D–G) ----
    st.markdown("<div class='panel-title'>Workspace panels</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    nav = [("Scenario Simulator", "What-if predictions from the trained model"),
           ("Model Monitoring", "Drift & data-quality checks on uploaded data"),
           ("Experiments (MLflow)", "Real training runs, metrics and versions"),
           ("Research Reports", "Report, model card, CSV/HTML/PDF exports")]
    for col, (name, desc) in zip(cols, nav):
        with col:
            st.markdown(f"<div class='glass'><div class='panel-title'>{name}</div><div class='panel-sub'>{desc}</div></div>",
                        unsafe_allow_html=True)
            st.button(f"Open", key=f"open_{name}", on_click=st.session_state["goto"], args=(name,))
