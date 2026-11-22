"""Genre Detector: logistic regression vs Naive Bayes, side by side, plus the confusion matrix."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find app/ui.py
import ui  # noqa: E402

ui.setup_page("Genre Detector", "🎸")
S, X = ui.load_songs()
P = ui.all_predictions()

ui.page_header("🎸", "Genre Detector", "Which genre is it? Two models guess from the sound alone.")
sid = ui.song_picker("Pick a song", key="genre_picker", default_label="Without Me – Eminem")
true_genre = S.loc[sid, "track_genre"]

left, a, b = st.columns([1, 1.35, 1.35], gap="large")
with left:
    ui.song_card(sid, key="genre_card", radar_height=200)
for col, key, title in [(a, "lr", "Logistic regression"), (b, "nb", "Naive Bayes")]:
    with col:
        probs = P[key].loc[sid]
        guess = probs.idxmax()
        ok = "✅ correct" if guess == true_genre else f"❌ really {true_genre}"
        ui.metric_card(title, guess, f"{probs.max():.0%} sure · {ok}")
        st.plotly_chart(ui.genre_prob_bar(probs, height=270), config={"displayModeBar": False}, key=f"genre_bar_{key}")

ui.section("Where do the models get confused?")
c1, c2 = st.columns([1, 2.2], vertical_alignment="top")
with c1:
    model = st.segmented_control("Model", ["Logistic regression", "Gaussian Naive Bayes"], default="Logistic regression",
                                 key="genre_cm_model") or "Logistic regression"
    st.markdown("Rows = the **true** genre, columns = the **predicted** genre, for the 854 test songs the models never saw. "
                "The diagonal is correct; everything else is a mix-up.")
    st.markdown("Most mix-ups are **hip-hop ↔ edm** (both danceable and electronic) and **jazz ↔ country** (mellow and acoustic).")
with c2:
    pr = ui.load_predictions("genre")
    G = ui.D.GENRES
    cm = np.array([[((pr["_true"] == t) & (pr[model] == p)).sum() for p in G] for t in G])
    fig = go.Figure(go.Heatmap(z=cm, x=G, y=G, colorscale=[[0, "#1E1E1E"], [1, "#137a3a"]], showscale=False,
                               text=cm, texttemplate="%{text}", textfont=dict(size=14, color="#F2F2F0"),
                               hovertemplate="true %{y} → predicted %{x}: %{z} songs<extra></extra>"))
    ui.base_layout(fig, height=400, legend=False)
    fig.update_yaxes(autorange="reversed", title="true genre", showgrid=False)
    fig.update_xaxes(title="predicted genre", showgrid=False, side="bottom")
    st.plotly_chart(fig, config={"displayModeBar": False}, key="genre_cm")

ui.how_it_works(
    ["Both models learn from the same 3,413 training songs and are tested on the same 854 unseen songs.",
     "**Logistic regression** (discriminative) learns a weighted score per genre and turns the scores into probabilities with **softmax**: it learns the boundaries between genres directly.",
     "**Naive Bayes** (generative) learns what a typical song of each genre looks like (a bell curve per feature) and uses **Bayes' rule** to ask which genre most likely produced this song.",
     "\"Naive\" = it assumes features are independent within a genre, which isn't true (energy and loudness rise together), so it's usually less accurate.",
     "We compare accuracy, macro F1 (every genre counts equally) and the confusion matrix."],
    ["Logistic regression", "Softmax", "Naive Bayes", "Bayes' rule", "Generative vs discriminative", "Confusion matrix", "Macro F1"])

ui.behind_the_scenes(
    "Logistic regression (C tuned by 5-fold cross-validation) vs Gaussian Naive Bayes (10 continuous audio features).",
    [("Logistic regression accuracy", f"{ui.score_of('Genre Detector', 'Logistic regression', 'genre', 'accuracy'):.0%}"),
     ("Logistic regression macro F1", f"{ui.score_of('Genre Detector', 'Logistic regression', 'genre', 'macro_f1'):.2f}"),
     ("Naive Bayes accuracy", f"{ui.score_of('Genre Detector', 'Gaussian Naive Bayes', 'genre', 'accuracy'):.0%}"),
     ("Random guessing", "18%")],
    ui.FIGS / "p3_genre" / "learning_curves.png",
    "Naive Bayes learns faster from very few songs; logistic regression wins with more data")
