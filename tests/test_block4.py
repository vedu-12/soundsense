"""Block 4 checks: shared evaluation helpers and score logging."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src import data, evaluate


def test_dummy_genre_and_logging(tmp_path):
    X_tr, X_te, y_tr, y_te, _ = data.load_splits()
    dummy = DummyClassifier(strategy="uniform", random_state=42).fit(X_tr, y_tr["genre"])
    m = evaluate.classification_metrics(y_te["genre"], dummy.predict(X_te))
    assert 0.12 < m["accuracy"] < 0.22          # ~1/6
    p = tmp_path / "scores.csv"
    evaluate.log_score("baseline", "Dummy (uniform)", "genre", m, path=p)
    evaluate.log_score("baseline", "Dummy (uniform)", "genre", m, path=p)   # rerun
    assert len(pd.read_csv(p)) == 1
    evaluate.log_score("baseline", "Other", "genre", m, path=p)
    assert len(pd.read_csv(p)) == 2


def test_baseline_row_logged_once():
    df = evaluate.load_scores()
    rows = df[(df.component == "baseline") & (df.task == "genre")]
    assert len(rows) == 1
    assert not df.duplicated(evaluate.KEY_COLS).any()


def test_metric_helpers():
    m = evaluate.classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert m["accuracy"] == 0.75 and m["roc_auc"] == 1.0 and "f1" in m
    r = evaluate.regression_metrics([1, 2, 3], [1, 2, 3])
    assert r["rmse"] == 0 and r["r2"] == 1
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    c = evaluate.clustering_metrics(X, [0, 0, 1, 1], [0, 0, 1, 1])
    assert c["silhouette"] > 0.8 and c["ari"] == 1.0
