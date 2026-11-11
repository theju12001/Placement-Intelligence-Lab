"""Panel D — What-if scenario simulator (uses the real trained model)."""
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
    ui.page_header("Scenario Simulator", "Compare a baseline student with a modified scenario — recalculated by the real model.")
    if not ui.require_model(c):
        return
    schema = c["schema"]
    base_col, scen_col = st.columns(2)
    with base_col:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        ui.panel("Baseline")
        base = ui.input_form_fields(schema, "base", cols=2)
        st.markdown("</div>", unsafe_allow_html=True)
    with scen_col:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        ui.panel("Scenario", "Change any value below; unchanged values mirror the baseline.")
        change = st.multiselect("Features to change", schema["features"], key="scen_feats")
        scen = dict(base)
        for f in change:
            label = f.replace("_", " ").title()
            if f in schema["numeric"]:
                s = schema["numeric"][f]
                scen[f] = st.slider(f"{label} (scenario)", float(s["min"]), float(s["max"]), float(base[f]),
                                    1.0 if s["integer"] else 0.1, key=f"scen_{f}")
            else:
                opts = schema["categorical"][f]["options"]
                scen[f] = st.selectbox(f"{label} (scenario)", opts, index=opts.index(base[f]), key=f"scen_{f}")
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.button("Run scenario", width="stretch"):
        return
    rb, rs = predict.predict_one(base), predict.predict_one(scen)
    for r in (rb, rs):
        if not r["ok"]:
            for e in r["errors"]:
                st.error(e)
            return
    delta = rs["probability"] - rb["probability"]
    ui.tiles([("Baseline", f"{rb['probability']:.1%}", rb["label"]), ("Scenario", f"{rs['probability']:.1%}", rs["label"]),
              ("Difference", f"{delta * 100:+.1f} pp", "percentage points")])
    changed = [(f, base[f], scen[f]) for f in schema["features"] if base[f] != scen[f]]
    if not changed:
        st.info("No feature was changed, so both predictions are identical.")
    else:
        st.dataframe(pd.DataFrame(changed, columns=["feature", "baseline", "scenario"]), width="stretch")
        lb, ls = explain.local_contributions(base), explain.local_contributions(scen)
        j = lb["table"].set_index("feature")[["contribution"]].join(
            ls["table"].set_index("feature")[["contribution"]], lsuffix="_baseline", rsuffix="_scenario")
        j["change"] = j["contribution_scenario"] - j["contribution_baseline"]
        j = j.loc[j["change"].abs().sort_values(ascending=False).index]
        fig = go.Figure()
        fig.add_bar(x=j.index, y=j["contribution_baseline"], name="Baseline", marker_color=config.COLORS["blue"])
        fig.add_bar(x=j.index, y=j["contribution_scenario"], name="Scenario", marker_color=config.COLORS["magenta"])
        fig.update_layout(barmode="group", yaxis_title="contribution to probability")
        st.caption(lb["method"])
        st.plotly_chart(viz3d.style(fig, 340, "Feature contributions: before vs after"), width="stretch")
        nums = [f for f, _, _ in changed if f in schema["numeric"]]
        ax = nums[0] if nums else None
        ay = nums[1] if len(nums) > 1 else None
        if ax:
            x2 = [float(base[ax]), float(scen[ax])]
            y2 = [float(base[ay]), float(scen[ay])] if ay else [0.0, 0.0]
            st.plotly_chart(viz3d.scenario_3d(rb["probability"], rs["probability"], x2, y2, ax, ay or "(single feature)"),
                            width="stretch")
    st.markdown("<div class='disclaimer'>These are model-based scenarios. A higher predicted probability does not "
                "mean that changing the feature would causally improve a real student's placement chances.</div>",
                unsafe_allow_html=True)
