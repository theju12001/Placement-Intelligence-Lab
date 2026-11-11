"""Panel G — Research reports and model card."""
import streamlit as st

import config
from src.reporting import reports
from src.utils import ui


def render():
    c = ui.ctx()
    ui.page_header("Research Reports & Model Card", "Downloadable documents built from the project's actual results.")
    if not ui.require_model(c):
        return
    ui.synthetic_banner(c)
    imp = ui.cached_importance()
    comp = reports.comparison_table(c["metrics"])
    html_doc = reports.build_html(imp["table"], imp["method"], comp)
    card = reports.build_model_card(imp["table"], imp["method"])
    a, b, cc, d = st.columns(4)
    a.download_button("Research report (HTML)", html_doc, "research_report.html", "text/html", width="stretch")
    b.download_button("Model card (Markdown)", card, "model_card.md", "text/markdown", width="stretch")
    cc.download_button("Model comparison (CSV)", comp.to_csv(index=False), "model_comparison.csv", "text/csv", width="stretch")
    try:
        pdf = reports.build_pdf(imp["table"], imp["method"])
        d.download_button("Research report (PDF)", pdf, "research_report.pdf", "application/pdf", width="stretch")
    except Exception as e:  # reportlab missing or failed
        d.caption(f"PDF unavailable ({type(e).__name__}); install reportlab.")
    t1, t2 = st.tabs(["Report preview", "Model card"])
    with t1:
        if hasattr(st, "iframe"):          # newer Streamlit
            st.iframe(html_doc, height=700)
        else:                               # older Streamlit
            import streamlit.components.v1 as components
            components.html(html_doc, height=700, scrolling=True)
    with t2:
        st.markdown(card)
    if st.button("Save copies to outputs/reports/"):
        (config.REPORTS_DIR / "research_report.html").write_text(html_doc, encoding="utf-8")
        (config.REPORTS_DIR / "model_card.md").write_text(card, encoding="utf-8")
        comp.to_csv(config.EXPORTS_DIR / "model_comparison.csv", index=False)
        st.success("Saved to outputs/reports/ and outputs/exports/.")
