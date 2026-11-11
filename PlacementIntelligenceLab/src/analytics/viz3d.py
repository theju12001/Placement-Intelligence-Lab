"""Plotly figures: dark neon theme, 3D analytics, model-evaluation charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config

C = config.COLORS
FONT = dict(family="Inter, Segoe UI, sans-serif", color=C["text"])


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=FONT, height=height, margin=dict(l=10, r=10, t=44 if title else 12, b=10),
        title=dict(text=title, font=dict(size=15, color=C["text"])) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _scene(xt: str, yt: str, zt: str) -> dict:
    ax = dict(backgroundcolor="rgba(5,8,22,0.0)", gridcolor="rgba(99,102,241,0.25)",
              zerolinecolor="rgba(99,102,241,0.25)", showbackground=False)
    return dict(xaxis=dict(title=xt, **ax), yaxis=dict(title=yt, **ax), zaxis=dict(title=zt, **ax),
                bgcolor="rgba(0,0,0,0)")


def hero_network() -> go.Figure:
    """Central holographic education visual: concept nodes around a core, built
    from the application's real analytical concepts."""
    nodes = ["Placement\nPrediction", "Academic\nPerformance", "Socioeconomic\nFactors",
             "Skills &\nCertifications", "Projects &\nExperience", "Career\nRecommendations"]
    cols = [C["magenta"], C["cyan"], C["purple"], C["blue"], C["green"], C["amber"]]
    rng = np.random.default_rng(3)
    pts = [(0, 0, 0)]
    for i in range(5):
        a = 2 * np.pi * i / 5
        pts.append((2.2 * np.cos(a), 2.2 * np.sin(a), 0.9 * np.sin(2 * a + 1)))
    # order: concept 0 is the core
    P = np.array(pts)
    fig = go.Figure()
    for i in range(1, 6):
        fig.add_trace(go.Scatter3d(x=[0, P[i, 0]], y=[0, P[i, 1]], z=[0, P[i, 2]], mode="lines",
                                   line=dict(color=cols[i], width=6), hoverinfo="skip", showlegend=False))
    for i in range(1, 6):
        j = i % 5 + 1
        fig.add_trace(go.Scatter3d(x=[P[i, 0], P[j, 0]], y=[P[i, 1], P[j, 1]], z=[P[i, 2], P[j, 2]],
                                   mode="lines", line=dict(color="rgba(139,147,199,0.35)", width=2),
                                   hoverinfo="skip", showlegend=False))
    # ring / halo
    t = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter3d(x=3.0 * np.cos(t), y=3.0 * np.sin(t), z=-0.8 + 0 * t, mode="lines",
                               line=dict(color="rgba(34,211,238,0.55)", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=2.6 * np.cos(t), y=2.6 * np.sin(t), z=-0.8 + 0 * t, mode="lines",
                               line=dict(color="rgba(168,85,247,0.55)", width=2), hoverinfo="skip", showlegend=False))
    size = [30, 20, 20, 20, 20, 20]
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(color=C["text"], size=11), hovertext=nodes, hoverinfo="text",
        marker=dict(size=size, color=cols, opacity=0.95, line=dict(color="white", width=1)), showlegend=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))))
    return style(fig, 430)


# ----------------------------------------------------------------------
def _placed(df: pd.DataFrame, target: str) -> pd.Series:
    from src.data import loader
    return loader.encode_target(df[target]).map({1: "Placed", 0: "Not placed"})


def feature_space(df, target, x, y, z, sample: int = 800, color_by: str | None = None) -> go.Figure:
    d = df.dropna(subset=[x, y, z]).copy()
    d["_o"] = _placed(d, target)
    d = d.sample(min(sample, len(d)), random_state=1)
    fig = go.Figure()
    for lab, col in (("Placed", C["cyan"]), ("Not placed", C["magenta"])):
        s = d[d["_o"] == lab]
        fig.add_trace(go.Scatter3d(x=s[x], y=s[y], z=s[z], mode="markers", name=lab,
                                   marker=dict(size=3.2, color=col, opacity=0.75),
                                   hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<extra>{lab}</extra>"))
    fig.update_layout(scene=_scene(x, y, z))
    return style(fig, 460)


def rate_surface(df, target, x, y, bins: int = 7) -> go.Figure:
    """Placement-rate surface over two binned numeric features."""
    from src.data import loader
    d = df[[x, y]].copy()
    d["y"] = loader.encode_target(df[target])
    d = d.dropna()
    d["bx"] = pd.qcut(d[x], bins, duplicates="drop")
    d["by"] = pd.qcut(d[y], bins, duplicates="drop")
    g = d.groupby(["by", "bx"], observed=True)["y"].mean().unstack()
    xs = [round(i.mid, 2) for i in g.columns]
    ys = [round(i.mid, 2) for i in g.index]
    fig = go.Figure(go.Surface(x=xs, y=ys, z=g.to_numpy(), colorscale=[[0, "#1e1b4b"], [0.5, "#a855f7"], [1, "#22d3ee"]],
                               colorbar=dict(title="Placement<br>rate"), opacity=0.95))
    fig.update_layout(scene=_scene(x, y, "placement rate"))
    return style(fig, 460)


def importance_3d(table: pd.DataFrame, method: str) -> go.Figure:
    t = table.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                           marker=dict(color=t["importance"], colorscale=[[0, C["blue"]], [1, C["magenta"]]])))
    fig.update_layout(xaxis_title="importance", margin=dict(l=10, r=10, t=10, b=10))
    return style(fig, 360)


def importance_scatter3d(table: pd.DataFrame, schema: dict) -> go.Figure:
    t = table.copy()
    t["group"] = t["feature"].map(lambda f: "Socioeconomic" if f in schema["socioeconomic"] else "Academic")
    t["rank"] = np.arange(len(t))
    gi = t["group"].map({"Academic": 0, "Socioeconomic": 1})
    fig = go.Figure()
    for grp, col in (("Academic", C["cyan"]), ("Socioeconomic", C["purple"])):
        s = t[t["group"] == grp]
        fig.add_trace(go.Scatter3d(x=s["rank"], y=[0 if grp == "Academic" else 1] * len(s), z=s["importance"],
                                   mode="markers+text", text=s["feature"], name=grp, textfont=dict(size=9),
                                   marker=dict(size=7 + 40 * s["importance"] / max(t["importance"].max(), 1e-9),
                                               color=col, opacity=0.85),
                                   hovertemplate="%{text}<br>importance=%{z:.4f}<extra></extra>"))
        for _, r in s.iterrows():
            fig.add_trace(go.Scatter3d(x=[r["rank"]] * 2, y=[0 if grp == "Academic" else 1] * 2,
                                       z=[0, r["importance"]], mode="lines",
                                       line=dict(color=col, width=3), showlegend=False, hoverinfo="skip"))
    fig.update_layout(scene=_scene("importance rank", "group (0=academic, 1=socio)", "importance"))
    return style(fig, 460)


def probability_distribution_3d(test: pd.DataFrame, x_feat: str) -> go.Figure:
    d = test.copy()
    d["Outcome"] = d["y_true"].map({1: "Placed", 0: "Not placed"})
    fig = go.Figure()
    for lab, col in (("Placed", C["cyan"]), ("Not placed", C["magenta"])):
        s = d[d["Outcome"] == lab]
        fig.add_trace(go.Scatter3d(x=s[x_feat], y=s["proba"], z=s["y_true"] + np.random.default_rng(1).normal(0, 0.03, len(s)),
                                   mode="markers", name=lab, marker=dict(size=3, color=col, opacity=0.7)))
    fig.update_layout(scene=_scene(x_feat, "predicted probability", "actual outcome (jittered)"))
    return style(fig, 460)


def cohort_comparison_3d(df, target, schema, a: str, b: str) -> go.Figure:
    """Mean of two numeric features per cohort of the first categorical feature."""
    cats = list(schema["categorical"])
    if not cats:
        return style(go.Figure(), 300, "No categorical feature available for cohorts")
    cat = cats[0]
    from src.data import loader
    d = df[[cat, a, b]].copy()
    d["y"] = loader.encode_target(df[target])
    g = d.groupby(cat).agg(x=(a, "mean"), y=(b, "mean"), z=("y", "mean"), n=("y", "size")).reset_index()
    fig = go.Figure(go.Scatter3d(x=g["x"], y=g["y"], z=g["z"], mode="markers+text", text=g[cat],
                                 marker=dict(size=8 + 22 * g["n"] / g["n"].max(), color=g["z"],
                                             colorscale=[[0, C["magenta"]], [1, C["cyan"]]], showscale=True,
                                             colorbar=dict(title="rate")),
                                 hovertemplate=cat + ": %{text}<br>n=%{customdata}<extra></extra>", customdata=g["n"]))
    fig.update_layout(scene=_scene(f"mean {a}", f"mean {b}", "placement rate"))
    return style(fig, 460)


def scenario_3d(baseline: float, scenario: float, cur_x, cur_y, label_x, label_y) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[cur_x[0]], y=[cur_y[0]], z=[baseline], mode="markers+text", text=["Baseline"],
                               marker=dict(size=10, color=C["blue"]), name="Baseline"))
    fig.add_trace(go.Scatter3d(x=[cur_x[1]], y=[cur_y[1]], z=[scenario], mode="markers+text", text=["Scenario"],
                               marker=dict(size=10, color=C["magenta"]), name="Scenario"))
    fig.add_trace(go.Scatter3d(x=cur_x, y=cur_y, z=[baseline, scenario], mode="lines",
                               line=dict(color=C["cyan"], width=5), showlegend=False))
    fig.update_layout(scene=_scene(label_x, label_y, "placement probability"))
    return style(fig, 380)


# ----------------------------------------------------------------------
def roc_fig(models: dict, selected: str | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=C["muted"]), name="chance"))
    for n, r in models.items():
        c = r["curves"]
        if "fpr" in c:
            fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{n} (AUC {r['metrics']['roc_auc']:.3f})",
                                     line=dict(width=3 if n == selected else 1.5)))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style(fig, 380)


def pr_fig(models: dict, selected: str | None = None) -> go.Figure:
    fig = go.Figure()
    for n, r in models.items():
        c = r["curves"]
        if "precision" in c:
            fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines",
                                     name=f"{n} (AP {r['metrics']['pr_auc']:.3f})",
                                     line=dict(width=3 if n == selected else 1.5)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return style(fig, 380)


def calibration_fig(curves_: dict, name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=C["muted"]), name="perfect"))
    if "cal_mean" in curves_:
        fig.add_trace(go.Scatter(x=curves_["cal_mean"], y=curves_["cal_frac"], mode="lines+markers", name=name,
                                 line=dict(color=C["cyan"], width=3)))
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed placement rate")
    return style(fig, 340)


def confusion_fig(cm: list, title: str = "") -> go.Figure:
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(z=z, x=["Pred: Not placed", "Pred: Placed"], y=["Actual: Not placed", "Actual: Placed"],
                               colorscale=[[0, "#1e1b4b"], [1, C["purple"]]], text=z, texttemplate="%{text}",
                               textfont=dict(size=18), showscale=False))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 320, title or None)
