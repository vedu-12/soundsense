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


@pytest.mark.parametrize("page,state", [
    ("4_Similar_Songs.py", {"sim_metric": "Cosine"}),
    ("7_Mood_Playlists.py", {"mood_pick": "Sad", "mood_popular": False}),
    ("7_Mood_Playlists.py", {"mood_pick": "Party"}),
    ("7_Mood_Playlists.py", {"mood_pick": "Intense"}),
    ("6_Song_Lab_and_Model_Arena.py", {"arena_task": "Hit"}),
    ("6_Song_Lab_and_Model_Arena.py", {"arena_task": "Happiness"}),
    ("1_Song_Map.py", {"map_colour": "Mood", "map_odd": False}),
    ("3_Genre_Detector.py", {"genre_cm_model": "Gaussian Naive Bayes"}),
    ("5_Hit_Predictor.py", {"song_id": 0}),
])
def test_page_interactions(page, state):
    at = AppTest.from_file(str(APP / "pages" / page), default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_song_lab_slider_moves_towards_metal():
    at = AppTest.from_file(str(APP / "pages" / "6_Song_Lab_and_Model_Arena.py"), default_timeout=120).run()
    at.slider(key="lab_energy").set_value(1.0)
    at.slider(key="lab_loudness").set_value(-2.0)
    at.slider(key="lab_acousticness").set_value(0.0)
    at.run()
    assert not at.exception
    says = [m.value for m in at.markdown if "The network says" in m.value]
    assert says and "heavy-metal" in says[0]
