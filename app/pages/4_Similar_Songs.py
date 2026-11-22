"""Similar Songs: k-nearest-neighbour recommendations, Euclidean vs cosine."""
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find app/ui.py
import ui  # noqa: E402

ui.setup_page("Similar Songs", "🎧")
S, X = ui.load_songs()
nbrs = ui.load_neighbours()

ui.page_header("🎧", "Similar Songs", "Find songs that sound like the one you love.")

c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
with c1:
    sid = ui.song_picker("Pick a song", key="sim_picker", default_label="HUMBLE. – Kendrick Lamar")
with c2:
    metric = st.segmented_control("Measure similarity by", ["Euclidean", "Cosine"], default="Euclidean", key="sim_metric") or "Euclidean"

rows = nbrs[(nbrs.song_id == sid) & (nbrs.metric == metric.lower())].sort_values("rank")
seed, chart = st.columns([1, 2], gap="large")
with seed:
    st.markdown("**Your song**")
    ui.song_card(sid, key="sim_seed", radar_height=210)
with chart:
    st.markdown(f"**How close are the picks?** · {metric.lower()} {'similarity (1 = same direction)' if metric == 'Cosine' else 'distance (0 = identical)'}")
    names = [S.loc[n, "track_name"][:28] for n in rows.neighbour_id][::-1]
    vals = [(1 - d) if metric == "Cosine" else d for d in rows.distance][::-1]
    cols_ = [ui.GENRE_COLORS[S.loc[n, "track_genre"]] for n in rows.neighbour_id][::-1]
    fig = ui.go.Figure(ui.go.Bar(x=vals, y=names, orientation="h", marker=dict(color=cols_, cornerradius=4),
                                 text=[f"{v:.2f}" for v in vals], textposition="outside", textfont=dict(color=ui.INK),
                                 hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    ui.base_layout(fig, height=300, legend=False)
    fig.update_xaxes(range=[0, max(vals) * 1.2], showgrid=False, visible=False)
    fig.update_layout(bargap=0.4)
    st.plotly_chart(fig, config={"displayModeBar": False}, key=f"sim_bar_{metric}")
    same = (S.loc[rows.neighbour_id, "track_genre"] == S.loc[sid, "track_genre"]).sum()
    st.caption(f"{same} of 5 share the genre ({S.loc[sid, 'track_genre']}). Switch the measure above to see the list change.")

st.markdown("**🎧 Your 5 recommendations** · dotted grey outline = your song")
cols = st.columns(5)
for c, (_, r) in zip(cols, rows.iterrows()):
    with c:
        ui.song_card(int(r.neighbour_id), key=f"sim_{metric}_{r.neighbour_id}", radar_height=170,
                     compare_id=sid, radar_labels=False)

ui.how_it_works(
    ["Every song becomes a point in a 10-dimensional space: one axis per audio feature (energy, danceability, …).",
     "Features are standardised first, so tempo (≈120) can't drown out danceability (0–1).",
     "**Euclidean** distance = the straight-line gap between two songs; **cosine** = the angle between them "
     "(same *shape* of song, even if one is more intense).",
     "The 5 closest songs (k-nearest neighbours) are your recommendations.",
     "The same geometry also classifies genres: kNN votes among neighbours; an SVM draws the widest-margin boundary."],
    ["kNN", "Euclidean distance", "Cosine similarity", "Feature scaling", "SVM", "Kernel trick"])

share_e = ui.score_of("Similar Songs", "kNN recommender (euclidean)", "recommender", "same_genre_top5")
share_c = ui.score_of("Similar Songs", "kNN recommender (cosine)", "recommender", "same_genre_top5")
svm_acc = ui.score_of("Similar Songs", "SVM (linear)", "genre", "accuracy")
ui.behind_the_scenes(
    "k-nearest neighbours (scikit-learn `NearestNeighbors`) on 10 scaled audio features, all 4,267 songs.",
    [("Same genre in top 5 (Euclidean)", f"{share_e:.0%}"), ("Same genre in top 5 (cosine)", f"{share_c:.0%}"),
     ("Random picks would give", "18%"), ("Linear SVM genre accuracy", f"{svm_acc:.0%}")],
    ui.FIGS / "p4_similar" / "knn_accuracy_vs_k.png", "kNN genre classifier: accuracy vs k (cross-validated)")
