"""Block 13 checks: every app page runs without exceptions; no teammate names or P1-P7 labels in the app."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py"] + sorted((APP / "pages").glob("*.py"))
NAMES = ["Aryan", "Gunneet", "Shreyash", "Amithav", "Vedang", "Krish", "Nikhil", "Palakollu", "Danny", "Chinhalli"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs(page):
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]


def test_no_names_or_role_labels():
    for f in list(APP.rglob("*.py")):
        text = f.read_text(encoding="utf-8")
        for n in NAMES:
            assert n not in text, (f.name, n)
        assert not re.search(r"\bP[1-7]\b", text), f.name
