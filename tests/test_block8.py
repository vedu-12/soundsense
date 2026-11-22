"""Block 8 checks: P4 Similar Songs."""
import joblib
import pandas as pd
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p4_similar"


def test_notebook():
    check_notebook("P4_similar_songs.ipynb")


def test_neighbours_table():
    nb = pd.read_csv(M / "neighbours.csv")
    songs, _ = data.load_all_songs()
    assert set(nb.metric) == {"euclidean", "cosine"}
    assert len(nb) == len(songs) * 2 * 5
    assert (nb.song_id != nb.neighbour_id).all()          # a song is never its own recommendation


def test_classifiers():
    X_tr, X_te, y_tr, y_te, _ = data.load_splits()
    for f in ["svm_best.joblib", "knn_best.joblib"]:
        b = joblib.load(M / f)
        acc = (b["model"].predict(X_te[b["features"]]) == y_te["genre"]).mean()
        assert 0.6 < acc < 0.95


def test_scores_logged():
    s = evaluate.load_scores()
    p4 = s[s.component == "P4 Similar Songs"]
    assert (p4.task == "genre").sum() == 3
    rec = p4[p4.task == "recommender"]
    assert len(rec) == 2 and (rec.same_genre_top5 > 0.4).all()
