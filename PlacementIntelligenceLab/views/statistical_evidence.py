"""Statistical evidence module."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.analytics import stats, viz3d
from src.data import loader
from src.utils import ui


@st.cache_data(show_spinner="Running tests…")
def _tests(path_key, n):
    c = ui.ctx()
    return stats.run_all_tests(c["df"], c["schema"], c["schema"]["target"])


def render():
    c = ui.ctx()
    df, schema, tgt = c["df"], c["schema"], c["schema"]["target"]
    ui.page_header("Statistical Evidence", "Descriptive statistics, hypothesis tests, effect sizes and odds ratios.")
    ui.synthetic_banner(c)
    if not schema["socioeconomic"]:
        st.warning("No socioeconomic columns were detected in this dataset — socioeconomic analysis is unavailable "
                   "(nothing is invented). Add valid columns to data/raw/ to enable it.")
    tabs = st.tabs(["Descriptive", "Distributions", "Correlation", "Placement rates", "Hypothesis tests", "Odds ratios"])
    with tabs[0]:
        st.dataframe(stats.descriptive(df, schema), width="stretch")
        miss = stats.missingness(df)
        st.markdown("**Missing values**")
        st.dataframe(miss, width="stretch") if len(miss) else st.caption("No missing values.")
        st.markdown("**Outliers (1.5×IQR rule; flagged, not removed)**")
        st.dataframe(stats.outliers(df, schema), width="stretch", hide_index=True)
    with tabs[1]:
        f = st.selectbox("Feature", schema["features"], key="dist_f")
        d = df.copy()
        d["Outcome"] = loader.encode_target(d[tgt]).map({1: "Placed", 0: "Not placed"})
        if f in schema["numeric"]:
            fig = px.histogram(d, x=f, color="Outcome", barmode="overlay", opacity=.65, marginal="box",
                               color_discrete_map={"Placed": "#22d3ee", "Not placed": "#ec4899"})
        else:
            fig = px.histogram(d, x=f, color="Outcome", barmode="group",
                               color_discrete_map={"Placed": "#22d3ee", "Not placed": "#ec4899"})
        st.plotly_chart(viz3d.style(fig, 400), width="stretch")
    with tabs[2]:
        corr = stats.correlation(df, schema, tgt)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#ec4899"], [.5, "#0b1020"], [1, "#22d3ee"]],
                        zmin=-1, zmax=1)
        st.plotly_chart(viz3d.style(fig, 480, "Spearman correlation (placed = 1)"), width="stretch")
        st.caption("Correlation does not establish causation.")
    with tabs[3]:
        f = st.selectbox("Group by", schema["features"], key="rate_f")
        g = stats.placement_rate_by(df, f, tgt)
        fig = go.Figure(go.Bar(x=g["group"], y=g["rate"], marker_color=config.COLORS["purple"],
                               error_y=dict(type="data", array=g["ci_high"] - g["rate"], arrayminus=g["rate"] - g["ci_low"])))
        fig.update_layout(yaxis_title="placement rate (95% Wilson CI)", yaxis_range=[0, 1])
        st.plotly_chart(viz3d.style(fig, 380), width="stretch")
        st.dataframe(g.round(3), width="stretch", hide_index=True)
        if g["small_sample"].any():
            st.warning("Some groups have n < 30; their rates and intervals are unreliable.")
        if f in schema["categorical"]:
            st.markdown("**Cross-tabulation**")
            st.dataframe(__import__("pandas").crosstab(df[f], df[tgt]), width="stretch")
    with tabs[4]:
        res = _tests(c["info"]["path"], len(df))
        st.dataframe(res.drop(columns=["H0", "interpretation"], errors="ignore").round(5), width="stretch", hide_index=True)
        st.caption("p_adj_BH = Benjamini–Hochberg adjusted p-value across all tests shown (controls false discovery rate).")
        pick = st.selectbox("Interpret", res["feature"])
        r = res[res["feature"] == pick].iloc[0]
        st.markdown(f"**H₀:** {r['H0']}  \n**Test:** {r['test']}  \n**p-value:** {r['p_value']:.4g}  \n"
                    f"**Effect size ({r['effect_name']}):** {r['effect_size']:.3f}")
        st.info(r["interpretation"])
    with tabs[5]:
        orr = stats.logistic_odds_ratios(df, schema, tgt)
        if orr is None:
            st.warning("Logistic regression did not converge (possible separation or too little data).")
        else:
            st.caption(f"Multivariable logistic regression, n={orr.attrs['n']} complete cases, McFadden pseudo-R²="
                       f"{orr.attrs['pseudo_r2']:.3f}. Numeric features standardised: OR is per +1 SD.")
            fig = go.Figure(go.Scatter(x=orr["odds_ratio"], y=orr["term"], mode="markers",
                                       error_x=dict(type="data", symmetric=False, array=orr["ci_high"] - orr["odds_ratio"],
                                                    arrayminus=orr["odds_ratio"] - orr["ci_low"]),
                                       marker=dict(color=config.COLORS["cyan"], size=9)))
            fig.add_vline(x=1, line_dash="dash", line_color=config.COLORS["amber"])
            fig.update_layout(xaxis_title="adjusted odds ratio (95% CI)", xaxis_type="log")
            st.plotly_chart(viz3d.style(fig, 420), width="stretch")
            st.dataframe(orr.round(4), width="stretch", hide_index=True)
            st.caption("Odds ratios are adjusted associations, not causal effects.")
