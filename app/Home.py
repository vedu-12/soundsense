"""SoundSense home page: analyze any song, then explore every feature."""
import streamlit as st

import ui

ui.setup_page("Home", "🎵")
S, X = ui.load_songs()
P = ui.all_predictions()
moods = ui.load_json("models/p7_moods/mood_names.json")
nbrs = ui.load_neighbours()

st.markdown("<div class='ss-hero'>Sound<span>Sense</span></div>", unsafe_allow_html=True)
st.markdown("<div class='ss-purpose'>Genre, happiness, hit chance, mood and look-alike songs, all worked out "
            "from how a song <b>sounds</b>.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ analyze a song
ui.section("🔎 Analyze a song")
sid = ui.song_picker("Pick any of 4,267 songs", key="home_picker", default_label="HUMBLE. – Kendrick Lamar")
song = S.loc[sid]

left, right = st.columns([1, 2.1], gap="large")
with left:
    ui.song_card(sid, key="home_card", radar_height=230)
with right:
    g = P["mlp_genre"].loc[sid]
    top_genre = g.idxmax()
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.metric_card("Genre", top_genre, f"{g.max():.0%} sure · neural network")
        st.page_link("pages/3_Genre_Detector.py", label="see how", icon=":material/arrow_forward:")
    with c2:
        m = song.mood
        ui.metric_card("Mood", f"{moods['emoji'][m]} {m}", f"{song[f'p_{m}']:.0%} of its mood mix")
        st.page_link("pages/7_Mood_Playlists.py", label="see how", icon=":material/arrow_forward:")
    with c3:
        ui.metric_card("Popularity", f"{song.popularity}", "on Spotify (0–100)")
        st.page_link("pages/1_Song_Map.py", label="see it on the map", icon=":material/arrow_forward:")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(ui.gauge(100 * P["valence"].loc[sid], "Happiness (predicted)", reference=100 * song.valence),
                        config={"displayModeBar": False}, key="home_happy")
        st.caption(f"White mark = Spotify's real value ({100 * song.valence:.0f}%)")
        st.page_link("pages/2_Happiness_Meter.py", label="see how", icon=":material/arrow_forward:")
    with g2:
        st.plotly_chart(ui.gauge(100 * P["hit"].loc[sid], "Hit chance", color="#c98500"),
                        config={"displayModeBar": False}, key="home_hit")
        st.caption(f"Real popularity {song.popularity} → {'a hit' if song.hit else 'not a hit'} (hit = 70+)")
        st.page_link("pages/5_Hit_Predictor.py", label="see how", icon=":material/arrow_forward:")

st.markdown("**🎧 5 songs that sound like it**")
ids = nbrs[(nbrs.song_id == sid) & (nbrs.metric == "euclidean")].sort_values("rank").neighbour_id.tolist()
cols = st.columns(5)
for c, n in zip(cols, ids):
    with c:
        ui.song_card(n, show_radar=False)
st.page_link("pages/4_Similar_Songs.py", label="see how", icon=":material/arrow_forward:")

# ------------------------------------------------------------------ feature grid
ui.section("✨ Explore SoundSense")
FEATURES = [
    ("🗺️", "Song Map", "Every song as a dot. Spot clusters and the odd ones out.", "pages/1_Song_Map.py"),
    ("😊", "Happiness Meter", "How happy does a song sound, and why?", "pages/2_Happiness_Meter.py"),
    ("🎸", "Genre Detector", "Guess the genre from sound alone.", "pages/3_Genre_Detector.py"),
    ("🎧", "Similar Songs", "Find songs that sound like the one you love.", "pages/4_Similar_Songs.py"),
    ("📈", "Hit Predictor", "Does it sound like a hit? See the reasons.", "pages/5_Hit_Predictor.py"),
    ("🧪", "Song Lab & Model Arena", "Design a song with sliders; see which model wins.", "pages/6_Song_Lab_and_Model_Arena.py"),
    ("🌈", "Mood Playlists", "Pick a mood, get a playlist.", "pages/7_Mood_Playlists.py"),
]
for row in [FEATURES[:4], FEATURES[4:]]:
    cols = st.columns(4)
    for c, (icon, name, desc, page) in zip(cols, row):
        with c:
            st.markdown(f"<div class='ss-feature'><div class='ic'>{icon}</div><div class='nm'>{name}</div>"
                        f"<div class='ds'>{desc}</div></div>", unsafe_allow_html=True)
            st.page_link(page, label=f"Open {name}", icon=":material/arrow_forward:")

st.caption("Data: Spotify Tracks Dataset (Hugging Face, BSD licence): 4,267 songs from 6 genres. " + ui.feature_glossary())
