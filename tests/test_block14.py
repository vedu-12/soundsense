"""Block 14 checks: final docs, brief coverage, repo hygiene, every notebook saved cleanly."""
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src import evaluate
from nb_utils import check_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["P1_data_song_map.ipynb", "P2_happiness_meter.ipynb", "P3_genre_detector.ipynb", "P4_similar_songs.ipynb",
             "P5_hit_predictor.ipynb", "P6_neural_net_arena.ipynb", "P7_mood_playlists.ipynb"]


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_all_notebooks_clean(nb):
    check_notebook(nb)


def test_docs_exist():
    for f in ["README.md", "docs/BUILD_LOG.md", "docs/BRIEF_CHECKLIST.md", "docs/DEMO_SCRIPT.md"]:
        assert (ROOT / f).is_file(), f
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for s in ["streamlit run app/Home.py", "pip install -r requirements.txt", "Results summary", "P1", "P7"]:
        assert s in readme


def test_checklist_all_yes():
    rows = [l for l in (ROOT / "docs/BRIEF_CHECKLIST.md").read_text(encoding="utf-8").splitlines()
            if l.startswith("| ") and l.rstrip().endswith("|") and l.count("|") == 5 and not l.startswith("| Type")
            and not l.startswith("|---")]
    assert len(rows) >= 130
    assert all(r.rstrip("| ").endswith("yes") for r in rows)


def test_no_large_files():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    for f in filter(None, files):
        p = ROOT / f
        if p.exists():
            assert p.stat().st_size < 50 * 1024 * 1024, f


def test_predictions_match_scores():
    """The saved test predictions reproduce the logged scores (no silent mismatch)."""
    s = evaluate.load_scores()
    pr = evaluate.load_predictions("genre")
    for model in [c for c in pr.columns if c != "_true"]:
        acc = (pr[model] == pr["_true"]).mean()
        logged = s[(s.task == "genre") & (s.model == model)].accuracy.iloc[0]
        assert abs(acc - logged) < 1e-9, model
