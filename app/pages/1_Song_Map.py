"""Song Map: every song in 2D (PCA), coloured by genre or mood, unusual songs (DBSCAN) highlighted."""
import plotly.graph_objects as go
import streamlit as st

import ui

ui.setup_page("Song Map", "🗺️")
S, X = ui.load_songs()
moods = ui.load_json("models/p7_moods/mood_names.json")
info = ui.load_json("models/p1_song_map/song_map_info.json")

ui.page_header("🗺️", "Song Map", "Every song as a dot: songs that sound alike sit close together.")

c1, c2, c3 = st.columns([2.2, 1.2, 1.2], vertical_alignment="bottom")
with c1:
    sid = ui.song_picker("Find a song on the map", key="map_picker", default_label="Without Me – Eminem")
with c2:
    colour_by = st.segmented_control("Colour by", ["Genre", "Mood"], default="Genre", key="map_colour") or "Genre"
with c3:
    show_odd = st.toggle("Highlight unusual songs", value=True, key="map_odd")

fig = go.Figure()
groups = ui.D.GENRES if colour_by == "Genre" else moods["moods"]
col = "track_genre" if colour_by == "Genre" else "mood"
for g in groups:
    d = S[S[col] == g]
    color = ui.GENRE_COLORS[g] if colour_by == "Genre" else moods["colors"][g]
    name = g if colour_by == "Genre" else f"{moods['emoji'][g]} {g}"
    fig.add_trace(go.Scattergl(x=d.pc1, y=d.pc2, mode="markers", name=name, customdata=d.index,
                               marker=dict(size=5, color=color, opacity=0.65),
                               text=d.track_name + "<br>" + d.artists_display + "<br>" + d.track_genre,
                               hovertemplate="%{text}<extra></extra>"))
if show_odd:
    d = S[S.is_outlier == 1]
    fig.add_trace(go.Scattergl(x=d.pc1, y=d.pc2, mode="markers", name="unusual song", customdata=d.index,
                               marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=ui.OUTLIER, width=1.6)),
                               text=d.track_name + "<br>" + d.artists_display + "<br>unusual: far from every crowd",
                               hovertemplate="%{text}<extra></extra>"))
s = S.loc[sid]
fig.add_trace(go.Scatter(x=[s.pc1], y=[s.pc2], mode="markers", name="your song", customdata=[sid],
                         marker=dict(size=20, symbol="star", color="white", line=dict(color="#121212", width=2)),
                         hovertemplate=f"{s.track_name}<br>{s.artists_display}<extra>your song</extra>"))
fig.add_annotation(x=s.pc1, y=s.pc2, text=f"<b>{s.track_name[:30]}</b>", showarrow=True, arrowhead=0, arrowcolor="white",
                   ax=60, ay=-45, bgcolor="#121212", bordercolor="white", borderpad=4, font=dict(color="white", size=12))
ui.base_layout(fig, height=600)
fig.update_layout(legend=dict(orientation="h", y=-0.16, x=0), dragmode="pan", margin=dict(l=10, r=10, t=10, b=10))
fig.update_xaxes(title="PC1 · quiet & acoustic  →  loud & energetic", zeroline=False)
fig.update_yaxes(title="PC2 · long & live  →  danceable & happy", zeroline=False)

map_col, side = st.columns([3, 1.05], gap="medium")
with map_col:
    event = st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": True},
                            on_select="rerun", selection_mode="points", key="song_map")
    st.caption("Hover for details · scroll to zoom · drag to pan · double-click to reset · click a dot to inspect it")
with side:
    picked = sid
    try:
        pts = event.selection.points if event else []
        if pts and pts[0].get("customdata") is not None:
            cd = pts[0]["customdata"]
            picked = int(cd[0] if isinstance(cd, (list, tuple)) else cd)
    except Exception:
        pass
    st.markdown("**Selected song**")
    extra = "<div class='ss-mixlab' style='margin-top:6px'>⚠️ unusual song (DBSCAN noise)</div>" if S.loc[picked, "is_outlier"] else ""
    ui.song_card(picked, extra_html=extra, key=f"map_card_{picked}", radar_height=200)
    ui.metric_card("Unusual songs", f"{int(S.is_outlier.sum())}", f"{S.is_outlier.mean():.1%} of all songs")

ui.how_it_works(
    ["Clean the data: remove duplicate songs and keep 6 clearly different genres (4,267 songs).",
     "Turn categories (key, mode) into numbers with one-hot encoding, and standard-scale every feature (scaler fit on the training songs only).",
     "**PCA** finds the directions where songs differ most; the first two (PC1 ≈ intensity, PC2 ≈ danceable & happy) give this map.",
     "**DBSCAN** groups songs that sit in dense crowds; songs with too few neighbours within a small radius are marked as *unusual*.",
     "The map keeps about half of the information, so two dots that look close can still differ in hidden directions."],
    ["Data cleaning", "One-hot encoding", "Standard scaling", "Train/test split", "PCA", "Explained variance", "DBSCAN"])

ev = info["explained_variance_ratio"]
ui.behind_the_scenes(
    f"PCA on 10 scaled audio features (fit on training songs) · DBSCAN with eps = {info['dbscan_eps']}, min_samples = {info['dbscan_min_samples']}.",
    [("Variance kept by this 2D map", f"{ev[0] + ev[1]:.0%}"), ("Components for 90%", f"{info['n_components_90']} of 10"),
     ("Unusual songs found", f"{info['n_outliers']}")],
    ui.FIGS / "p1_song_map" / "scree_plot.png", "Scree plot: how much each principal component explains")
