"""Block 9 checks: P5 Hit Predictor."""
import json
import joblib
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p5_hit"


def test_notebook():
    check_notebook("P5_hit_predictor.ipynb")


def test_hit_model():
    b = joblib.load(M / "hit_model.joblib")
    assert "popularity" not in b["features"]
    X_tr, X_te, y_tr, y_te, _ = data.load_splits()
    p = b["model"].predict_proba(X_te[b["features"]])[:, 1]
    m = evaluate.classification_metrics(y_te["hit"], (p >= b["threshold"]).astype(int), p)
    assert 0.6 < m["roc_auc"] < 0.95          # better than chance, not suspicious
    assert abs(p.mean() - y_tr["hit"].mean()) < 0.05   # calibrated: average chance ≈ hit rate
    prof = json.loads((M / "hit_profile.json").read_text())
    assert set(prof) >= {"importance", "hit_means", "nonhit_means"}


def test_scores_logged_beat_baseline():
    s = evaluate.load_scores()
    h = s[s.task == "hit"]
    base = h[h.component == "baseline"].iloc[0]
    p5 = h[h.component == "P5 Hit Predictor"]
    assert len(p5) == 4
    assert (p5.roc_auc > base.roc_auc + 0.2).all() and (p5.f1 > base.f1).all()
