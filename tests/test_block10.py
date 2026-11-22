"""Block 10 checks: P7 Mood Playlists."""
import json
import joblib
import numpy as np
import pandas as pd
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p7_moods"


def test_notebook():
    check_notebook("P7_mood_playlists.ipynb")


def test_moods_saved():
    moods = pd.read_csv(M / "moods.csv", index_col=0)
    songs, _ = data.load_all_songs()
    names = json.loads((M / "mood_names.json").read_text())["moods"]
    assert len(moods) == len(songs) and set(moods.mood) == set(names)
    p = moods[[f"p_{m}" for m in names]]
    assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
    for f in ["kmeans.joblib", "gmm.joblib"]:
        b = joblib.load(M / f)
        assert b["features"] == data.MOOD_FEATURES     # genre never used


def test_scores_logged():
    s = evaluate.load_scores()
    m = s[(s.component == "P7 Mood Playlists") & (s.task == "mood")]
    assert len(m) == 3 and (m.silhouette > 0).all() and m.ari.between(-0.1, 0.9).all()
