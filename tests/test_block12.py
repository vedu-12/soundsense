"""Block 12 checks: from-scratch algorithms match scikit-learn."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src import data, scratch
from nb_utils import check_notebook

X_tr, X_te, y_tr, y_te, _ = data.load_splits()
A = data.AUDIO_FEATURES


def test_knn_within_2pp():
    sk = KNeighborsClassifier(13).fit(X_tr[A], y_tr["genre"]).score(X_te[A], y_te["genre"])
    s = (scratch.KNNClassifierScratch(13).fit(X_tr[A], y_tr["genre"]).predict(X_te[A]) == y_te["genre"]).mean()
    assert abs(sk - s) <= 0.02


def test_naive_bayes_identical():
    sk = GaussianNB(var_smoothing=0.1).fit(X_tr[A], y_tr["genre"])
    s = scratch.GaussianNBScratch(0.1).fit(X_tr[A], y_tr["genre"])
    assert (sk.predict(X_te[A]) == s.predict(X_te[A])).all()
    assert np.allclose(sk.predict_proba(X_te[A]), s.predict_proba(X_te[A]), atol=1e-6)


def test_linear_regression_normal_and_gd():
    Xv, Xtv = X_tr.drop(columns="valence"), X_te.drop(columns="valence")
    sk = LinearRegression().fit(Xv, y_tr["valence"])
    for method in ["normal", "gd"]:
        s = scratch.LinearRegressionScratch(method).fit(Xv, y_tr["valence"])
        assert np.allclose(s.coef_, sk.coef_, atol=1e-4)
        assert abs(r2_score(y_te["valence"], s.predict(Xtv)) - r2_score(y_te["valence"], sk.predict(Xtv))) < 1e-4


def test_kmeans_inertia_close():
    _, X = data.load_all_songs()
    M = X[data.MOOD_FEATURES].values
    sk = KMeans(4, n_init=10, random_state=42).fit(M).inertia_
    s = scratch.KMeansScratch(4).fit(M).inertia_
    assert abs(s - sk) / sk < 0.02


def test_notebooks_still_ok():
    for nb in ["P2_happiness_meter.ipynb", "P3_genre_detector.ipynb", "P4_similar_songs.ipynb", "P7_mood_playlists.ipynb"]:
        check_notebook(nb)
