"""SoundSense data pipeline (FROZEN after Block 3).

Run `python -m src.data` to rebuild everything in data/processed/.

Steps in build_dataset():
  clean -> keep the 6 chosen genres -> drop tracks listed under more than one chosen genre
  -> drop duplicate track_ids -> drop duplicate songs (same track_name + artists)
  -> add the hit label -> encode categories -> stratified 80/20 split by genre
  -> fit StandardScaler on TRAIN only -> save to data/processed/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- constants
GENRES = ["classical", "country", "edm", "heavy-metal", "hip-hop", "jazz"]
HIT_THRESHOLD = 70          # hit = popularity >= 70 (hit rate 8.9% in our genres, so no fallback)
RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "spotify_tracks.csv"
PROCESSED = ROOT / "data" / "processed"

META_COLS = ["track_id", "track_name", "artists", "album_name"]   # display only, never inputs
# Continuous audio features (the "sound" of a song)
AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
MOOD_FEATURES = ["energy", "valence", "tempo", "acousticness", "danceability"]
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------- helpers
def _encode(df):
    """Turn the categorical columns into numbers.

    - explicit: True/False -> 1/0
    - mode: already 1 (major) / 0 (minor); a 2-category one-hot needs just this one column
    - key (0-11, a category, not a quantity): one-hot, dropping key 0 (C) as the baseline
      so the columns are not perfectly collinear (needed for the normal equation)
    - time_signature: one-hot of 3, 5, other (0/1 are rare/invalid); 4/4 is the baseline
    """
    out = df[AUDIO_FEATURES].astype(float).copy()
    out["explicit"] = df["explicit"].astype(int)
    out["mode"] = df["mode"].astype(int)
    for k in range(1, 12):
        out[f"key_{KEY_NAMES[k]}"] = (df["key"] == k).astype(int)
    ts = df["time_signature"]
    out["time_sig_3"] = (ts == 3).astype(int)
    out["time_sig_5"] = (ts == 5).astype(int)
    out["time_sig_other"] = (~ts.isin([3, 4, 5])).astype(int)
    return out


def build_dataset(verbose=True):
    """Build and save the processed dataset. Returns a summary dict."""
    raw = pd.read_csv(RAW_CSV, index_col=0)
    summary = {"raw_rows": len(raw), "raw_genres": int(raw.track_genre.nunique())}

    df = raw.dropna(subset=META_COLS)                          # 1 row has no name/artist
    df = df[df.track_genre.isin(GENRES)]
    summary["rows_in_chosen_genres"] = len(df)

    n_genres = df.groupby("track_id").track_genre.nunique()
    multi = n_genres[n_genres > 1].index
    df = df[~df.track_id.isin(multi)]                          # ambiguous genre -> remove
    summary["tracks_in_multiple_genres_removed"] = len(multi)

    before = len(df)
    df = df.drop_duplicates("track_id")
    summary["duplicate_track_ids_removed"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(["track_name", "artists"])         # same song, different release
    summary["duplicate_name_artist_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    df.index.name = "song_id"
    df["hit"] = (df["popularity"] >= HIT_THRESHOLD).astype(int)

    X = _encode(df)
    feature_names = list(X.columns)

    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df["track_genre"], random_state=RANDOM_STATE)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    scaler = StandardScaler().fit(X.loc[train_idx])            # TRAIN ONLY: no test leakage

    keep = META_COLS + ["track_genre", "popularity", "hit", "split", "key", "time_signature"]
    songs = pd.concat([df[keep], X], axis=1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    songs.to_csv(PROCESSED / "songs.csv")
    joblib.dump(scaler, PROCESSED / "scaler.joblib")
    (PROCESSED / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (PROCESSED / "config.json").write_text(json.dumps({
        "genres": GENRES, "hit_threshold": HIT_THRESHOLD, "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE, "audio_features": AUDIO_FEATURES, "mood_features": MOOD_FEATURES,
    }, indent=2))

    summary.update({
        "songs_kept": len(df), "n_features": len(feature_names),
        "train_size": len(train_idx), "test_size": len(test_idx),
        "songs_per_genre": df.track_genre.value_counts().to_dict(),
        "hit_rate_train": float(df.loc[train_idx, "hit"].mean()),
        "hit_rate_test": float(df.loc[test_idx, "hit"].mean()),
    })
    if verbose:
        for k, v in summary.items():
            print(f"{k:38s} {v}")
    return summary


# ---------------------------------------------------------------- loaders
def _load_songs():
    path = PROCESSED / "songs.csv"
    if not path.exists():
        raise FileNotFoundError("Processed data missing. Run `python -m src.data` first.")
    return pd.read_csv(path, index_col="song_id")


def load_feature_names():
    return json.loads((PROCESSED / "feature_names.json").read_text())


def load_scaler():
    return joblib.load(PROCESSED / "scaler.joblib")


def _scaled(songs, feature_names, scaler):
    return pd.DataFrame(scaler.transform(songs[feature_names]), index=songs.index, columns=feature_names)


def load_splits():
    """Return X_train, X_test (scaled DataFrames), y_train, y_test (dicts), feature_names.

    y dicts have keys: genre, valence, hit, popularity.
    Remember: drop 'valence' from X when valence is the target (use X.drop(columns='valence')).
    """
    songs = _load_songs()
    feature_names = load_feature_names()
    X = _scaled(songs, feature_names, load_scaler())
    tr, te = songs.split == "train", songs.split == "test"

    def ys(mask):
        s = songs[mask]
        return {"genre": s.track_genre, "valence": s.valence, "hit": s.hit, "popularity": s.popularity}

    return X[tr], X[te], ys(tr), ys(te), feature_names


def load_all_songs():
    """Return (songs, X_all): the full kept table (metadata, labels, original feature values, split)
    and the scaled features for every song (same index, scaled with the train-fitted scaler)."""
    songs = _load_songs()
    feature_names = load_feature_names()
    return songs, _scaled(songs, feature_names, load_scaler())


def scale_raw_features(raw_df):
    """Scale a DataFrame of unscaled encoded features (used by the app's Song Lab sliders)."""
    feature_names = load_feature_names()
    return pd.DataFrame(load_scaler().transform(raw_df[feature_names]), columns=feature_names)


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    build_dataset()
