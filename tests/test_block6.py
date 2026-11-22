"""Block 6 checks: P2 Happiness Meter."""
import joblib
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p2_happiness"


def test_notebook():
    check_notebook("P2_happiness_meter.ipynb")


def test_valence_model_has_no_valence_input():
    for f in ["valence_best.joblib", "valence_ridge.joblib"]:
        b = joblib.load(M / f)
        assert "valence" not in b["features"] and "popularity" not in b["features"]
        X_tr, X_te, y_tr, y_te, _ = data.load_splits()
        r2 = evaluate.regression_metrics(y_te["valence"], b["model"].predict(X_te[b["features"]]))["r2"]
        assert 0.1 < r2 < 0.8          # beats the mean baseline, not suspiciously perfect


def test_scores_logged_and_beat_baseline():
    s = evaluate.load_scores()
    v = s[(s.task == "valence")]
    base = v[v.component == "baseline"].r2.iloc[0]
    p2 = v[v.component == "P2 Happiness Meter"]
    assert len(p2) == 4 and (p2.r2.max() > base + 0.1)
    assert len(s[(s.task == "popularity") & (s.component == "P2 Happiness Meter")]) == 4
