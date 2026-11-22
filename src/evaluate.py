"""Shared evaluation: the same metrics for every model, and one scores table (results/scores.csv)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = ROOT / "results" / "scores.csv"
FIGURES = ROOT / "results" / "figures"
KEY_COLS = ["component", "model", "task"]


def classification_metrics(y_true, y_pred, y_proba=None):
    """Accuracy, macro F1, precision, recall (+ hit-class F1 and ROC-AUC for binary tasks).

    Multi-class: precision/recall are macro averages (every genre counts equally).
    Binary (hit): precision/recall/F1 are for the positive class (hit = 1), and ROC-AUC uses
    y_proba = predicted probability of class 1.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    binary = len(np.unique(y_true)) == 2
    avg = "binary" if binary else "macro"
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if binary:
        m["f1"] = f1_score(y_true, y_pred, zero_division=0)
        if y_proba is not None:
            m["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: float(v) for k, v in m.items()}


def regression_metrics(y_true, y_pred):
    """RMSE (typical error size), MAE (average absolute error), R² (share of variation explained)."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clustering_metrics(X, labels, true_labels=None):
    """Silhouette (how well songs fit their own cluster vs the next) and ARI vs real labels."""
    m = {"silhouette": float(silhouette_score(X, labels))}
    if true_labels is not None:
        m["ari"] = float(adjusted_rand_score(true_labels, labels))
    return m


def log_score(component, model_name, task, metrics_dict, path=SCORES_CSV):
    """Append one row to scores.csv, replacing any row with the same component+model+task."""
    path = Path(path)
    row = pd.DataFrame([{"component": component, "model": model_name, "task": task, **metrics_dict}])
    if path.exists():
        old = pd.read_csv(path)
        same = ((old.component == component) & (old.model == model_name) & (old.task == task))
        df = pd.concat([old[~same], row], ignore_index=True)
    else:
        df = row
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_scores(path=SCORES_CSV):
    return pd.read_csv(path)


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion matrix", save_path=None,
                          normalize=False, ax=None, cmap="Blues"):
    """Draw a confusion matrix (rows = true, columns = predicted) and optionally save it."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted genre" if len(labels) > 2 else "Predicted")
    ax.set_ylabel("True genre" if len(labels) > 2 else "True")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, txt, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, random_state=42):
    """Resample the test set with replacement n_boot times; return (mean, low, high) of the metric."""
    rng = np.random.default_rng(random_state)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    return float(stats.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def log_baselines():
    """Dummy baselines ('guessing') for every task, so real models can be compared with them."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from src.data import load_splits
    X_tr, X_te, y_tr, y_te, _ = load_splits()
    d = DummyClassifier(strategy="uniform", random_state=42).fit(X_tr, y_tr["genre"])
    log_score("baseline", "Dummy (uniform guess)", "genre",
              classification_metrics(y_te["genre"], d.predict(X_te)))
    d = DummyClassifier(strategy="stratified", random_state=42).fit(X_tr, y_tr["hit"])
    log_score("baseline", "Dummy (random by hit rate)", "hit",
              classification_metrics(y_te["hit"], d.predict(X_te), d.predict_proba(X_te)[:, 1]))
    for task in ("valence", "popularity"):
        d = DummyRegressor(strategy="mean").fit(X_tr, y_tr[task])
        log_score("baseline", "Dummy (predict the mean)", task,
                  regression_metrics(y_te[task], d.predict(X_te)))
    return load_scores()


if __name__ == "__main__":
    print(log_baselines().to_string())
