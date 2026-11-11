"""Local deterministic research assistant (no LLM / API key)."""
import streamlit as st

from src.analytics import assistant
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Research Assistant", "Ask about metrics, features, statistical findings and limitations. "
                   "Answers use locally computed results only — no external AI service.")
    if not ui.require_model(c):
        return
    imp = ui.cached_importance()
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    ex = ["Which model was selected?", "What are the model metrics?", "Which features matter most?",
          "Does household income affect placement?", "What are the limitations?", "How is fairness handled?"]
    cols = st.columns(3)
    for i, q in enumerate(ex):
        if cols[i % 3].button(q, key=f"ex_{i}", width="stretch"):
            st.session_state["chat"] += [("user", q), ("assistant", assistant.answer(q, imp["table"]))]
    for role, msg in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Ask a question about the results…")
    if q:
        st.session_state["chat"] += [("user", q), ("assistant", assistant.answer(q, imp["table"]))]
        st.rerun()
