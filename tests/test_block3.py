"""Block 3 checks: the frozen data pipeline."""
import numpy as np
import pytest

from src import data


@pytest.fixture(scope="module")
def splits():
    return data.load_splits()


def test_popularity_not_a_feature(splits):
    names = splits[4]
    assert "popularity" not in names and "hit" not in names
    for c in data.META_COLS + ["track_genre"]:
        assert c not in names


def test_sizes(splits):
    X_tr, X_te, *_ = splits
    n = len(X_tr) + len(X_te)
    assert 3500 < n < 7000
    assert abs(len(X_te) / n - 0.2) < 0.01


def test_genre_proportions_match(splits):
    _, _, y_tr, y_te, _ = splits
    p_tr = y_tr["genre"].value_counts(normalize=True)
    p_te = y_te["genre"].value_counts(normalize=True)
    assert set(p_tr.index) == set(data.GENRES)
    assert (p_tr - p_te).abs().max() < 0.01


def test_no_nans(splits):
    X_tr, X_te, y_tr, y_te, _ = splits
    assert not X_tr.isna().any().any() and not X_te.isna().any().any()
    for y in (y_tr, y_te):
        for v in y.values():
            assert not v.isna().any()


def test_train_scaled(splits):
    X_tr = splits[0]
    assert np.allclose(X_tr.mean(), 0, atol=1e-8)
    assert np.allclose(X_tr.std(ddof=0), 1, atol=1e-6)


def test_hit_label(splits):
    _, _, y_tr, y_te, _ = splits
    assert ((y_tr["popularity"] >= data.HIT_THRESHOLD) == (y_tr["hit"] == 1)).all()
    assert 0.05 <= y_tr["hit"].mean() <= 0.2
    assert abs(y_tr["hit"].mean() - y_te["hit"].mean()) < 0.03


def test_no_duplicate_songs():
    songs, X_all = data.load_all_songs()
    assert not songs.track_id.duplicated().any()
    assert not songs.duplicated(["track_name", "artists"]).any()
    assert len(songs) == len(X_all)
    assert (songs.index == X_all.index).all()


def test_train_test_disjoint(splits):
    X_tr, X_te, *_ = splits
    assert len(set(X_tr.index) & set(X_te.index)) == 0
