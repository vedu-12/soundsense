"""Shared colours so every notebook and app page shows genres and moods the same way.

The six genre colours are a colour-blind-checked categorical palette (validated for
both light and dark backgrounds). Moods reuse the same palette in order.
"""
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#8e6cd8", "#e66767", "#9085e9"]

GENRE_COLORS = {
    "classical": "#3987e5",
    "country": "#d95926",
    "edm": "#199e70",
    "heavy-metal": "#c98500",
    "hip-hop": "#d55181",
    "jazz": "#8e6cd8",
}

ACCENT = "#1DB954"      # app accent (Spotify-like green)
OUTLIER = "#e34948"     # unusual songs (DBSCAN noise)
NEUTRAL = "#8a8a86"


def setup_notebook_style():
    """Clean, readable matplotlib defaults for the notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.alpha": 0.4, "axes.titleweight": "bold"})
