"""Shared UI components and cached context for all pages."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import config
from src.data import loader
from src.models import predict


def inject_css() -> None:
    css = (config.ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _dataset(mtime: float, path_key: str):
    df, info = loader.load_dataset()
    return df, info, loader.infer_schema(df, info["target"])


def dataset_ctx():
    path, _ = loader.find_dataset_path()
    return _dataset(path.stat().st_mtime, str(path))


def ctx() -> dict:
    """Everything a page needs. ``trained`` False → pages show an honest empty state."""
    df, info, schema = dataset_ctx()
    trained = predict.model_available()
    return {"df": df, "info": info, "schema": schema, "trained": trained,
            "metrics": predict.load_metrics() if trained else None,
            "meta": predict.load_metadata() if trained else None}


def require_model(c: dict) -> bool:
    if not c["trained"]:
        st.warning("No trained model found. Open **Settings → Retrain** or run `python train_model.py` in a terminal.")
        return False
    return True


def tiles(items: list[tuple[str, str, str]]) -> None:
    h = "".join(f"<div class='tile'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div>"
                f"<div class='s'>{html.escape(s)}</div></div>" for k, v, s in items)
    st.markdown(f"<div class='tiles'>{h}</div>", unsafe_allow_html=True)


def panel(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='panel-title'>{html.escape(title)}</div>"
                + (f"<div class='panel-sub'>{html.escape(sub)}</div>" if sub else ""), unsafe_allow_html=True)


def chip(text: str, kind: str = "") -> str:
    return f"<span class='chip {kind}'>{html.escape(text)}</span>"


def page_header(title: str, sub: str = "") -> None:
    st.markdown(f"<h2 style='margin-bottom:.1rem'>{html.escape(title)}</h2>"
                f"<div class='panel-sub' style='margin-bottom:.8rem'>{html.escape(sub)}</div>", unsafe_allow_html=True)


def synthetic_banner(c: dict) -> None:
    if c["info"]["synthetic"]:
        st.markdown("<div class='disclaimer'><b>Synthetic demo data.</b> No real student dataset was found in "
                    "<code>data/raw/</code>. All numbers below come from generated data whose relationships are "
                    "developer assumptions — not real-world validation.</div>", unsafe_allow_html=True)


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.1f}%"


def input_form_fields(schema: dict, key: str, defaults: dict | None = None, cols: int = 3) -> dict:
    """Render inputs derived from the real schema; returns the raw values."""
    vals: dict = {}
    defaults = defaults or {}
    layout = st.columns(cols)
    i = 0
    groups = [("Academic & skills", schema["academic"]), ("Socioeconomic", schema["socioeconomic"])]
    for gname, feats in groups:
        if not feats:
            continue
        st.caption(gname.upper())
        layout = st.columns(cols)
        for j, f in enumerate(feats):
            with layout[j % cols]:
                label = f.replace("_", " ").title()
                if f in schema["numeric"]:
                    s = schema["numeric"][f]
                    d = defaults.get(f, s["median"])
                    if s["integer"]:
                        vals[f] = st.number_input(label, float(s["min"]), float(s["max"]), float(round(d)), 1.0,
                                                  key=f"{key}_{f}", format="%d")
                    else:
                        step = 0.1 if (s["max"] - s["min"]) <= 20 else 1.0
                        vals[f] = st.number_input(label, float(s["min"]), float(s["max"]), float(round(d, 2)), step,
                                                  key=f"{key}_{f}")
                else:
                    opts = schema["categorical"][f]["options"]
                    d = defaults.get(f, opts[0])
                    vals[f] = st.selectbox(label, opts, index=opts.index(d) if d in opts else 0, key=f"{key}_{f}")
    return vals


@st.cache_data(show_spinner="Computing feature importance…")
def _importance_cached(version: str):
    from src.explainability import explain
    g = explain.global_importance()
    return {"method": g["method"], "table": g["table"]}


def cached_importance() -> dict:
    """Global importance (SHAP if available, else permutation) cached per model version."""
    return _importance_cached(predict.load_bundle()["version"])
