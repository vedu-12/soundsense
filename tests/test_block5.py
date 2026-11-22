"""Block 5 checks: P1 notebook and its saved outputs."""
import json
import joblib
import pandas as pd

from src import data
from nb_utils import check_notebook

M = data.ROOT / "models" / "p1_song_map"


def test_notebook():
    check_notebook("P1_data_song_map.ipynb")


def test_song_map_saved():
    sm = pd.read_csv(M / "song_map.csv", index_col=0)
    songs, _ = data.load_all_songs()
    assert len(sm) == len(songs) and set(sm.columns) >= {"pc1", "pc2", "is_outlier"}
    assert 0 < sm.is_outlier.mean() < 0.10
    info = json.loads((M / "song_map_info.json").read_text())
    assert 2 <= info["n_components_90"] <= 10
    pca = joblib.load(M / "pca.joblib")
    assert abs(sum(pca.explained_variance_ratio_) - 1) < 1e-6


def test_figures_saved():
    f = data.ROOT / "results" / "figures" / "p1_song_map"
    for n in ["correlation_heatmap.png", "scree_plot.png", "song_map_genre.png", "song_map_outliers.png"]:
        assert (f / n).exists(), n
