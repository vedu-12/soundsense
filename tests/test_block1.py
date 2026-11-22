"""Block 1 checks: raw dataset downloaded and has the expected shape."""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "spotify_tracks.csv"


def test_raw_csv_shape():
    if not RAW.exists():
        import pytest
        pytest.skip("raw CSV not downloaded (data/raw is git-ignored)")
    df = pd.read_csv(RAW, index_col=0)
    assert len(df) == 114000
    for c in ["track_id", "track_genre", "popularity", "valence", "danceability", "energy", "tempo"]:
        assert c in df.columns
