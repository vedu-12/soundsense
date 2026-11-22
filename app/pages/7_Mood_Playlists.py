"""Mood Playlists: moods discovered by clustering (no labels); each song shows its GMM mood mix."""
import plotly.graph_objects as go
import streamlit as st

import ui

ui.setup_page("Mood Playlists", "🌈")
S, X = ui.load_songs()
info = ui.load_json("models/p7_moods/mood_names.json")
MOODS = info["moods"]

ui.page_header("🌈", "Mood Playlists", "Pick a mood, get a playlist. The moods were discovered by the computer, with no labels.")

c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
with c1:
    mood = st.pills("Mood", MOODS, default="Chill", format_func=lambda m: f"{info['emoji'][m]} {m}", key="mood_pick") or "Chill"
with c2:
    popular = st.toggle("Popular songs first", value=True, key="mood_popular")

d = S[S.mood == mood]
avg = d[["energy", "valence", "tempo", "acousticness", "danceability"]].mean()
m1, m2, m3, m4 = st.columns(4)
with m1:
    ui.metric_card("Songs in this mood", f"{len(d):,}", f"{len(d) / len(S):.0%} of all songs")
with m2:
    ui.metric_card("Energy", f"{avg.energy:.2f}", "0 = calm, 1 = intense")
with m3:
    ui.metric_card("Happiness", f"{avg.valence:.2f}", "0 = sad, 1 = cheerful")
with m4:
    ui.metric_card("Tempo", f"{avg.tempo:.0f} BPM", f"acousticness {avg.acousticness:.2f}")

st.write("")
play, mapc = st.columns([1.35, 1], gap="large")
with play:
    st.markdown(f"**{info['emoji'][mood]} Your {mood} playlist** · each bar shows the song's mood mix")
    # the most typical songs: highest probability for this mood; optionally favour popular ones
    d = d.assign(score=d[f"p_{mood}"] + (d.popularity / 100 if popular else 0))
    picks = d.sort_values(["score", "popularity"], ascending=False).head(10)
    cols = st.columns(2)
    for i, sid in enumerate(picks.index):
        with cols[i % 2]:
            ui.song_card(int(sid), show_radar=False, show_mood=True, key=f"mood_{sid}")
with mapc:
    st.markdown("**Moods on the Song Map**")
    fig = go.Figure()
    for m in MOODS:
        dm = S[S.mood == m]
        sel = m == mood
        fig.add_trace(go.Scattergl(x=dm.pc1, y=dm.pc2, mode="markers", name=f"{info['emoji'][m]} {m}",
                                   marker=dict(size=6 if sel else 4, color=info["colors"][m], opacity=0.85 if sel else 0.12),
                                   text=dm.track_name + "<br>" + dm.artists_display, hovertemplate="%{text}<extra>" + m + "</extra>"))
    ui.base_layout(fig, height=430)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(title="quiet → intense", showticklabels=False, zeroline=False)
    fig.update_yaxes(title="long & live → danceable & happy", showticklabels=False, zeroline=False, range=[-6.5, 3.5])
    st.plotly_chart(fig, config={"displayModeBar": False}, key="mood_map")
    st.caption("Moods cut across genres: country songs land in all four moods.")

ui.how_it_works(
    ["Use only 5 \"feel\" features: energy, happiness (valence), tempo, acousticness and danceability. Genre is never used.",
     "**k-means** places 4 centres, assigns each song to the nearest, moves each centre to the average of its songs, and repeats. k = 4 was chosen with the elbow method and silhouette score.",
     "**Hierarchical clustering** (Ward) on a 1,500-song sample builds a tree of merges (dendrogram) and confirms 4 natural groups.",
     "A **Gaussian Mixture Model** trained with **EM** gives every song a probability for each mood (E-step: probabilities; M-step: update the moods). That's the mood mix bar.",
     "Each cluster is named from its averages: Party (energetic, happy, danceable), Chill (calm, acoustic, positive), Sad (quiet, slow, low happiness), Intense (loud, fast, dark)."],
    ["Unsupervised learning", "k-means", "Elbow method", "Silhouette score", "Hierarchical clustering", "GMM", "EM algorithm", "Adjusted Rand Index"])

ui.behind_the_scenes(
    "k-means (k = 4) for the playlists; Gaussian Mixture Model (spherical, EM) for each song's mood mix.",
    [("k-means silhouette", f"{ui.score_of('Mood Playlists', 'k-means (k=4)', 'mood', 'silhouette'):.2f}"),
     ("GMM silhouette", f"{ui.score_of('Mood Playlists', 'GMM via EM (k=4)', 'mood', 'silhouette'):.2f}"),
     ("Moods vs genres (ARI)", f"{ui.score_of('Mood Playlists', 'k-means (k=4)', 'mood', 'ari'):.2f}")],
    ui.FIGS / "p7_moods" / "moods_vs_genres.png", "Share of each genre's songs in each mood",
    note="ARI = 1 would mean moods copy genres exactly; 0 = no relation. 0.29 means moods only partly follow genres.")
