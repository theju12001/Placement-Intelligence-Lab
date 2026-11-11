"""Navigation / UI smoke tests using Streamlit's AppTest harness."""
import importlib
from pathlib import Path

import pytest

import config

APP = str(Path(config.ROOT / "app.py"))


def test_all_view_modules_import():
    for _, mod, _ in config.NAV_ITEMS:
        m = importlib.import_module(f"views.{mod}")
        assert callable(m.render)


@pytest.mark.parametrize("name,mod,icon", config.NAV_ITEMS)
def test_every_page_renders(trained, name, mod, icon):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["nav"] = f"{icon}  {name}"
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_predict_form_submission(trained):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["nav"] = "◎  Predict"
    at.run()
    at.button[0].click().run() if at.button else None
    assert not at.exception, [e.value for e in at.exception]
    assert any("Predicted outcome" in m.value for m in at.markdown)


def test_hero_button_navigates(trained):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.button[0].click().run()          # "Try Prediction"
    assert at.session_state["nav"].endswith("Predict")
    assert not at.exception


def test_pages_show_branding_and_title(trained):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    text = " ".join(m.value for m in at.markdown)
    assert config.APP_NAME in text and config.TAGLINE in text
    assert config.ACADEMIC_TITLE in text
    assert "Theju Gangadhar" in text and "Karthik TG" in text
