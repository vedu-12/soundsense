"""Block 2 checks: the chosen genres exist and the hit rate at the threshold is >= 5%."""
from pathlib import Path
import pandas as pd
import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "spotify_tracks.csv"
GENRES = ["classical", "country", "edm", "heavy-metal", "hip-hop", "jazz"]
HIT_THRESHOLD = 70


@pytest.mark.skipif(not RAW.exists(), reason="raw CSV not downloaded")
def test_genres_and_hit_rate():
    df = pd.read_csv(RAW, index_col=0)
    s = df[df.track_genre.isin(GENRES)]
    assert set(s.track_genre.unique()) == set(GENRES)
    assert (s.track_genre.value_counts() == 1000).all()
    multi = s.groupby("track_id").track_genre.nunique()
    assert (multi > 1).sum() < 50  # little overlap between the chosen genres
    assert (s.drop_duplicates("track_id").popularity >= HIT_THRESHOLD).mean() >= 0.05
