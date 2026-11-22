"""Block 7 checks: P3 Genre Detector."""
import joblib
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p3_genre"


def test_notebook():
    check_notebook("P3_genre_detector.ipynb")


def test_models_saved_and_reasonable():
    X_tr, X_te, y_tr, y_te, _ = data.load_splits()
    for f, lo in [("logistic_regression.joblib", 0.6), ("naive_bayes.joblib", 0.5)]:
        b = joblib.load(M / f)
        acc = (b["model"].predict(X_te[b["features"]]) == y_te["genre"]).mean()
        assert lo < acc < 0.95, (f, acc)        # well above 1/6, not suspiciously perfect


def test_scores_logged():
    s = evaluate.load_scores()
    p3 = s[(s.component == "P3 Genre Detector") & (s.task == "genre")]
    assert set(p3.model) == {"Logistic regression", "Gaussian Naive Bayes"}
    base = s[(s.component == "baseline") & (s.task == "genre")].accuracy.iloc[0]
    assert (p3.accuracy > base + 0.3).all()
