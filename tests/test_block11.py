"""Block 11 checks: P6 Neural Network & Model Arena."""
import joblib
import pandas as pd
from src import data, evaluate
from nb_utils import check_notebook

M = data.ROOT / "models" / "p6_neural"


def test_notebook():
    check_notebook("P6_neural_net_arena.ipynb")


def test_mlps_saved():
    X_tr, X_te, y_tr, y_te, _ = data.load_splits()
    g = joblib.load(M / "mlp_genre.joblib")
    assert 0.6 < (g["model"].predict(X_te[g["features"]]) == y_te["genre"]).mean() < 0.95
    v = joblib.load(M / "mlp_valence.joblib")
    assert "valence" not in v["features"]
    h = joblib.load(M / "mlp_hit.joblib")
    assert "popularity" not in h["features"]


def test_arena_complete():
    s = evaluate.load_scores()
    assert not s.duplicated(evaluate.KEY_COLS).any()
    for task, n_min in [("genre", 7), ("hit", 6), ("valence", 6)]:
        assert (s.task == task).sum() >= n_min
    for task in ["genre", "hit", "valence"]:
        p = evaluate.load_predictions(task)
        assert "_true" in p.columns and len(p) == 854
    b = pd.read_csv(data.ROOT / "results" / "bootstrap_ci.csv")
    assert (b["95% CI low"] <= b["test score"]).all() and (b["test score"] <= b["95% CI high"]).all()
