"""Placement Intelligence Lab — Streamlit entry point.

Run:  python -m streamlit run app.py
"""
import importlib

import streamlit as st

import config
from src.models import predict
from src.utils import ui

st.set_page_config(page_title=f"{config.APP_NAME} · {config.TAGLINE}", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

LABELS = [f"{icon}  {name}" for name, _, icon in config.NAV_ITEMS]
LABEL_TO_MODULE = {f"{icon}  {name}": mod for name, mod, icon in config.NAV_ITEMS}
NAME_TO_LABEL = {name.lower(): f"{icon}  {name}" for name, _, icon in config.NAV_ITEMS}

if "nav" not in st.session_state:
    st.session_state["nav"] = LABELS[0]


def goto(name: str) -> None:
    """Callback used by hero buttons and search."""
    st.session_state["nav"] = NAME_TO_LABEL[name.lower()]


def _search_cb() -> None:
    q = st.session_state.get("page_search", "").strip().lower()
    if q:
        hit = next((n for n in NAME_TO_LABEL if q in n), None)
        if hit:
            st.session_state["nav"] = NAME_TO_LABEL[hit]
        st.session_state["page_search"] = ""


st.session_state["goto"] = goto  # exposed to views

with st.sidebar:
    st.markdown(f"<div class='brand'><div class='brand-logo'>P</div><div><div class='brand-name'>{config.APP_NAME}</div>"
                f"<div class='brand-tag'>{config.TAGLINE}</div></div></div>", unsafe_allow_html=True)
    st.radio("Navigate", LABELS, key="nav", label_visibility="collapsed")
    st.markdown(f"<div class='side-foot'><b>Project executors</b><br>{' · '.join(config.EXECUTORS)}<br><br>"
                f"v{config.APP_VERSION}<br>Educational research tool. Not for sole use in employment decisions.</div>",
                unsafe_allow_html=True)

# ---- top bar: search, status, profile ----
c1, c2, c3 = st.columns([3, 5, 3])
with c1:
    st.text_input("Search pages", key="page_search", placeholder="Search pages…  (e.g. monitoring)",
                  label_visibility="collapsed", on_change=_search_cb)
with c2:
    meta = predict.load_metadata() if predict.model_available() else None
    if meta:
        synth = meta["dataset"].get("synthetic")
        st.markdown(ui.chip(f"Model: {meta['selected_model']} · {meta['version']}", "ok")
                    + ui.chip("Synthetic demo data" if synth else "Real data", "warn" if synth else "ok"),
                    unsafe_allow_html=True)
    else:
        st.markdown(ui.chip("No trained model — run train_model.py", "bad"), unsafe_allow_html=True)
with c3:
    ini = "".join(n[0] for n in config.EXECUTORS)
    st.markdown(f"<div class='topbar'><span class='pulse'></span><div class='userbox'><b>{config.EXECUTORS[0]}</b><br>"
                f"& {config.EXECUTORS[1]}</div><div class='avatar'>{ini}</div></div>", unsafe_allow_html=True)

mod = importlib.import_module(f"views.{LABEL_TO_MODULE[st.session_state['nav']]}")
try:
    mod.render()
except predict.ModelNotTrained as e:
    st.warning(str(e))
