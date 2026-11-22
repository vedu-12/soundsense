"""Hit Predictor: calibrated hit chance from a Random Forest, with plain-English reasons."""
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find app/ui.py
import ui  # noqa: E402

ui.setup_page("Hit Predictor", "📈")
S, X = ui.load_songs()
P = ui.all_predictions()
prof = ui.load_json("models/p5_hit/hit_profile.json")

ui.page_header("📈", "Hit Predictor", "Does it sound like a hit? A hit = Spotify popularity of 70 or more.")
sid = ui.song_picker("Pick a song", key="hit_picker", default_label="Without Me – Eminem")
song = S.loc[sid]
chance = P["hit"].loc[sid]
base = prof["hit_rate"]

UNITS = {"loudness": lambda v: f"{v:.1f} dB", "tempo": lambda v: f"{v:.0f} BPM",
         "duration_ms": lambda v: f"{int(v // 60000)}:{int(v % 60000 // 1000):02d} min"}
NICE = {"duration_ms": "length", "speechiness": "spoken words", "acousticness": "acousticness",
        "instrumentalness": "instrumental", "liveness": "live feel", "valence": "happiness"}


def fmt(f, v):
    return UNITS.get(f, lambda x: f"{x:.2f}")(v)


left, mid, right = st.columns([1, 1.2, 1.7], gap="large")
with left:
    ui.song_card(sid, key="hit_card", radar_height=200)
with mid:
    st.plotly_chart(ui.gauge(100 * chance, "Hit chance", color="#c98500"), config={"displayModeBar": False}, key="hit_gauge")
    ratio = chance / base
    verdict = "Sounds like a hit" if chance >= prof["threshold"] else "Doesn't sound like a typical hit"
    ui.metric_card("Verdict", verdict, f"{ratio:.1f}× the average song's chance ({base:.0%})")
    st.caption(f"Real popularity: {song.popularity} → {'a hit ✅' if song.hit else 'not a hit'}")
with right:
    st.markdown("**Why? The features that matter most for hits**")
    feats = [f for f in sorted(prof["importance"], key=prof["importance"].get, reverse=True) if f in prof["hit_means"]][:5]
    for f in feats:
        v, h, n = song[f], prof["hit_means"][f], prof["nonhit_means"][f]
        like_hit = abs(v - h) < abs(v - n)
        high = v > (h + n) / 2
        icon = "✅" if like_hit else "⚠️"
        tail = "like most hits" if like_hit else "more like non-hits"
        WORDS = {"duration_ms": ("Long song", "Short song"), "loudness": ("Loud", "Quiet"), "tempo": ("Fast", "Slow")}
        phrase = WORDS[f][0 if high else 1] if f in WORDS else f"{'High' if high else 'Low'} {NICE.get(f, f)}"
        st.markdown(f"<div class='ss-reason'>{icon} <b>{phrase}</b> ({fmt(f, v)}): {tail} "
                    f"<small>· hits avg {fmt(f, h)}, others {fmt(f, n)}</small></div>", unsafe_allow_html=True)

ui.how_it_works(
    ["Label: **hit = popularity ≥ 70**. Only about 9% of songs are hits, so the classes are imbalanced.",
     "A **decision tree** asks yes/no questions (\"danceability > 0.55?\"); each split makes the groups as pure as possible (Gini impurity).",
     "A **Random Forest** averages 300 trees trained on random samples of songs and features (bagging, less variance); **AdaBoost** adds small trees that focus on past mistakes (boosting).",
     "Class weights make a missed hit cost more; models are judged by **F1 and ROC-AUC**, not accuracy. Probabilities are **calibrated** so \"30%\" really means about 30%.",
     "Reasons come from the forest's most important features, comparing this song with the average hit and non-hit."],
    ["Decision tree", "Gini impurity", "Random Forest", "AdaBoost", "Class imbalance", "ROC-AUC", "Calibration"])

ui.behind_the_scenes(
    "Random Forest (300 trees, class-weighted) with isotonic calibration; decision threshold chosen on training data.",
    [("Test ROC-AUC", f"{ui.score_of('Hit Predictor', 'Random Forest (calibrated)', 'hit', 'roc_auc'):.2f}"),
     ("Test F1", f"{ui.score_of('Hit Predictor', 'Random Forest (calibrated)', 'hit', 'f1'):.2f}"),
     ("Hits caught (recall)", f"{ui.score_of('Hit Predictor', 'Random Forest (calibrated)', 'hit', 'recall'):.0%}"),
     ("Single tree ROC-AUC", f"{ui.score_of('Hit Predictor', 'Decision tree', 'hit', 'roc_auc'):.2f}")],
    ui.FIGS / "p5_hit" / "feature_importance.png", "What the forest relies on most",
    note="Honest limit: sound explains only part of a hit. Fame, marketing and luck aren't in the audio, "
         "and importance is not causation.")
