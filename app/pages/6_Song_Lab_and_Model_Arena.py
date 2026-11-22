"""Song Lab (design a song with sliders, the neural network guesses its genre) and Model Arena (leaderboards)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find app/ui.py
import ui  # noqa: E402

ui.setup_page("Song Lab & Model Arena", "🧪")
S, X = ui.load_songs()
M = ui.load_models()
lab = ui.load_json("models/p6_neural/song_lab.json")

ui.page_header("🧪", "Song Lab & Model Arena", "Design a song and watch the neural network guess its genre, then see which model wins.")
tab_lab, tab_arena = st.tabs(["🎛️ Song Lab", "🏆 Model Arena"])

# ------------------------------------------------------------------ Song Lab
with tab_lab:
    SLIDERS = [("energy", "Energy", 0.0, 1.0, 0.01), ("danceability", "Danceability", 0.0, 1.0, 0.01),
               ("valence", "Happiness (valence)", 0.0, 1.0, 0.01), ("acousticness", "Acousticness", 0.0, 1.0, 0.01),
               ("loudness", "Loudness (dB)", -35.0, 0.0, 0.5), ("tempo", "Tempo (BPM)", 60.0, 200.0, 1.0),
               ("speechiness", "Spoken words", 0.0, 0.6, 0.01), ("instrumentalness", "Instrumental", 0.0, 1.0, 0.01)]
    base = dict(lab["base_song"])
    base["tempo"], base["loudness"] = round(base["tempo"]), round(base["loudness"] * 2) / 2
    if st.button("Reset to a typical song", key="lab_reset"):
        for f, *_ in SLIDERS:
            st.session_state[f"lab_{f}"] = float(round(base[f], 2))
    sl, out = st.columns([1.3, 1], gap="large")
    with sl:
        st.markdown("**Move the sliders**")
        c1, c2 = st.columns(2)
        values = {}
        for i, (f, label, lo, hi, step) in enumerate(SLIDERS):
            with (c1 if i % 2 == 0 else c2):
                values[f] = st.slider(label, lo, hi, float(np.clip(round(base[f], 2), lo, hi)), step, key=f"lab_{f}")
    song = pd.DataFrame([base])
    for f, v in values.items():
        song[f] = v
    b = M["mlp_genre"]
    probs = pd.Series(b["model"].predict_proba(ui.D.scale_raw_features(song)[b["features"]])[0], index=b["model"].classes_)
    with out:
        guess = probs.idxmax()
        ui.metric_card("The network says", guess, f"{probs.max():.0%} sure")
        st.plotly_chart(ui.genre_prob_bar(probs, height=300), config={"displayModeBar": False}, key="lab_bar")
        st.caption("Try it: push energy and loudness up and watch heavy-metal take over; turn them down and add acousticness for classical or jazz.")

# ------------------------------------------------------------------ Model Arena
with tab_arena:
    scores = ui.load_scores()
    TASKS = {"Genre": ("genre", ["accuracy", "macro_f1"], "accuracy"),
             "Hit": ("hit", ["roc_auc", "f1", "precision", "recall"], "roc_auc"),
             "Happiness": ("valence", ["r2", "rmse", "mae"], "r2")}
    LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "roc_auc": "ROC-AUC", "f1": "F1", "precision": "Precision",
              "recall": "Recall", "r2": "R²", "rmse": "RMSE", "mae": "MAE"}
    task_name = st.segmented_control("Task", list(TASKS), default="Genre", key="arena_task") or "Genre"
    task, cols, key = TASKS[task_name]
    t = scores[scores.task == task][["model", "component"] + cols].sort_values(key, ascending=False).reset_index(drop=True)
    t.index = t.index + 1
    winner = t.iloc[0]
    boot = pd.read_csv(ui.ROOT / "results" / "bootstrap_winners.csv")
    brow = boot[boot.task == task].iloc[0]

    w1, w2, w3 = st.columns(3)
    with w1:
        ui.metric_card("🏆 Winner", winner.model, f"{LABELS[key]} {winner[key]:.3f}")
    with w2:
        ui.metric_card("Runner-up", t.iloc[1].model, f"{LABELS[key]} {t.iloc[1][key]:.3f}")
    with w3:
        clear = "Yes" if str(brow["clearly better?"]).startswith("yes") else "Not clearly"
        ui.metric_card("Is the win real?", clear, f"bootstrap 95% range of the gap {brow['95% CI of difference']}")

    def highlight(row):
        return ["background-color: rgba(29,185,84,0.22); font-weight: 700" if row.name == 1 else "" for _ in row]
    shown = t.rename(columns={**LABELS, "model": "Model", "component": "Part of SoundSense"})
    st.dataframe(shown.style.apply(highlight, axis=1).format({LABELS[c]: "{:.3f}" for c in cols}),
                 width="stretch", height=min(42 + 35 * len(t), 420))
    st.caption("Same test set (854 songs) and the same metric code for every model. Baseline = random guessing / predicting the average.")

    ui.section("Inspect a model")
    pr = ui.load_predictions(task)
    options = [m for m in t.model if m in pr.columns]
    m1, m2 = st.columns([1, 2.2], gap="large")
    with m1:
        chosen = st.selectbox("Model", options, key=f"arena_model_{task}")
        row = t[t.model == chosen].iloc[0]
        for c in cols:
            ui.metric_card(LABELS[c], f"{row[c]:.3f}")
            st.write("")
    with m2:
        if task in ("genre", "hit"):
            labels = ui.D.GENRES if task == "genre" else [0, 1]
            names = ui.D.GENRES if task == "genre" else ["not hit", "hit"]
            yt = pr["_true"] if task == "genre" else pr["_true"].astype(int)
            yp = pr[chosen] if task == "genre" else pr[chosen].astype(int)
            cm = np.array([[((yt == a) & (yp == b_)).sum() for b_ in labels] for a in labels])
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, colorscale=[[0, "#1E1E1E"], [1, "#137a3a"]], showscale=False,
                                       text=cm, texttemplate="%{text}", textfont=dict(size=14, color="#F2F2F0"),
                                       hovertemplate="true %{y} → predicted %{x}: %{z}<extra></extra>"))
            ui.base_layout(fig, height=420, legend=False)
            fig.update_yaxes(autorange="reversed", title="true", showgrid=False)
            fig.update_xaxes(title="predicted", showgrid=False)
            fig.update_layout(title=dict(text=f"Confusion matrix · {chosen}", font=dict(size=14, color=ui.INK_2)))
        else:
            fig = go.Figure(go.Scattergl(x=pr["_true"], y=pr[chosen], mode="markers", marker=dict(size=5, color="#3987e5", opacity=0.6),
                                         hovertemplate="real %{x:.2f} → predicted %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#8a8a86", dash="dash"), hoverinfo="skip"))
            ui.base_layout(fig, height=420, legend=False)
            fig.update_xaxes(title="real happiness (valence)", range=[0, 1])
            fig.update_yaxes(title="predicted", range=[-0.05, 1.05])
            fig.update_layout(title=dict(text=f"Predicted vs real · {chosen}", font=dict(size=14, color=ui.INK_2)))
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"arena_plot_{task}")

ui.how_it_works(
    ["The **neural network** (MLP): audio features → a hidden layer of ReLU neurons → softmax over the 6 genres.",
     "Training: a forward pass makes a guess, the loss measures the error, **backpropagation** works out each weight's share of the blame, and Adam nudges the weights. Repeat for many epochs.",
     "**Early stopping** and a small weight penalty stop it from memorising the training songs; layer size, penalty and learning rate were tuned by cross-validation.",
     "**Model Arena**: every model in SoundSense is scored on the same 854 test songs with the same metrics.",
     "**Bootstrap**: resample the test set 1,000 times to see if the winner's lead is real or could be luck."],
    ["Neural network (MLP)", "ReLU", "Softmax", "Backpropagation", "Early stopping", "Cross-validation", "Bootstrap CI"])

ui.behind_the_scenes(
    f"MLPClassifier, hidden layer {M['mlp_genre']['params']['hidden_layer_sizes']}, alpha {M['mlp_genre']['params']['alpha']}, early stopping.",
    [("Network genre accuracy", f"{ui.score_of('Neural Network', 'MLP (neural network)', 'genre', 'accuracy'):.1%}"),
     ("Network happiness R²", f"{ui.score_of('Neural Network', 'MLP (neural network)', 'valence', 'r2'):.2f}"),
     ("Network hit ROC-AUC", f"{ui.score_of('Neural Network', 'MLP (neural network)', 'hit', 'roc_auc'):.2f}")],
    ui.FIGS / "p6_arena" / "loss_curve.png", "Training vs validation loss: without early stopping the network overfits")
