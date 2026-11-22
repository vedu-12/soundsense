"""Shared pieces for every SoundSense page: data/model loading, page layout, song cards, charts."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as D            # noqa: E402
from src.style import ACCENT, GENRE_COLORS, OUTLIER, PALETTE   # noqa: E402

MODELS = ROOT / "models"
FIGS = ROOT / "results" / "figures"
RADAR_FEATURES = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness", "instrumentalness"]
RADAR_LABELS = ["dance", "energy", "happy", "acoustic", "speech", "live", "instrum."]
INK, INK_2, MUTED, SURFACE, BORDER = "#F2F2F0", "#C9C9C4", "#8E8E89", "#1E1E1E", "#2E2E2C"

REQUIRED = [
    D.PROCESSED / "songs.csv", MODELS / "p1_song_map" / "song_map.csv", MODELS / "p2_happiness" / "valence_best.joblib",
    MODELS / "p3_genre" / "logistic_regression.joblib", MODELS / "p4_similar" / "neighbours.csv",
    MODELS / "p5_hit" / "hit_model.joblib", MODELS / "p6_neural" / "mlp_genre.joblib", MODELS / "p7_moods" / "moods.csv",
    ROOT / "results" / "scores.csv",
]

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.ss-purpose {{color: {INK_2}; font-size: 1.08rem; margin-top: -0.6rem; margin-bottom: 1.2rem;}}
.ss-card {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px 10px 16px;}}
.ss-title {{font-weight: 700; font-size: 1.02rem; color: {INK}; line-height: 1.25; overflow: hidden; min-height: 2.5em;
           text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;}}
.ss-artist {{color: {INK_2}; font-size: 0.9rem; margin: 2px 0 8px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.ss-chip {{display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
          background: #2A2A28; color: {INK}; font-size: 0.8rem; margin-right: 4px; border: 1px solid {BORDER};}}
.ss-dot {{width: 9px; height: 9px; border-radius: 50%; display: inline-block;}}
.ss-concept {{display: inline-block; padding: 3px 11px; border-radius: 999px; background: rgba(29,185,84,0.14);
             color: {INK}; border: 1px solid rgba(29,185,84,0.55); font-size: 0.82rem; margin: 2px 4px 2px 0;}}
.ss-metric {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; height: 100%;}}
.ss-metric .lab {{color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;}}
.ss-metric .val {{color: {INK}; font-size: 1.6rem; font-weight: 700; margin-top: 2px;}}
.ss-metric .sub {{color: {INK_2}; font-size: 0.85rem;}}
.ss-mix {{display: flex; height: 12px; border-radius: 6px; overflow: hidden; gap: 2px; margin: 8px 0 2px 0;}}
.ss-mixlab {{color: {INK_2}; font-size: 0.78rem;}}
.ss-reason {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; color: {INK};}}
.ss-reason small {{color: {INK_2};}}
.ss-feature {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px; height: 100%;}}
.ss-feature .ic {{font-size: 1.7rem;}}
.ss-feature .nm {{font-weight: 700; font-size: 1.05rem; margin: 6px 0 2px 0; color: {INK};}}
.ss-feature .ds {{color: {INK_2}; font-size: 0.9rem; min-height: 2.6em;}}
.ss-hero {{font-size: 3.2rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0;}}
.ss-hero span {{color: {ACCENT};}}
.ss-section {{margin-top: 1.8rem; margin-bottom: 0.4rem; font-size: 1.3rem; font-weight: 700;}}
div[data-testid="stExpander"] details {{border-radius: 12px;}}
</style>
"""


# ------------------------------------------------------------------ page scaffolding
def setup_page(title, icon):
    st.set_page_config(page_title=f"{title} · SoundSense", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        st.error("Some models or processed data are missing. **Run the notebooks first: see README.** "
                 f"(missing: {', '.join(str(p.relative_to(ROOT)) for p in missing[:3])} …)")
        st.stop()


def page_header(icon, title, purpose):
    st.markdown(f"# {icon} {title}")
    st.markdown(f"<div class='ss-purpose'>{purpose}</div>", unsafe_allow_html=True)


def how_it_works(steps, concepts):
    with st.expander("How it works", expanded=False):
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
        st.markdown(" ".join(f"<span class='ss-concept'>{c}</span>" for c in concepts), unsafe_allow_html=True)


def behind_the_scenes(model_text, score_rows, figure=None, caption=None, note=None):
    """score_rows: list of (label, value) shown as metric cards."""
    with st.expander("Behind the scenes", expanded=False):
        st.markdown(f"**Model:** {model_text}")
        if score_rows:
            cols = st.columns(len(score_rows))
            for c, (lab, val) in zip(cols, score_rows):
                with c:
                    metric_card(lab, val)
        if note:
            st.caption(note)
        if figure is not None and Path(figure).exists():
            st.image(str(figure), caption=caption, width="stretch")


def metric_card(label, value, sub=""):
    st.markdown(f"<div class='ss-metric'><div class='lab'>{label}</div><div class='val'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def section(title):
    st.markdown(f"<div class='ss-section'>{title}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ data + models (cached)
@st.cache_data(show_spinner=False)
def load_songs():
    songs, X = D.load_all_songs()
    sm = pd.read_csv(MODELS / "p1_song_map" / "song_map.csv", index_col=0)
    moods = pd.read_csv(MODELS / "p7_moods" / "moods.csv", index_col=0)
    S = songs.join(sm[["pc1", "pc2", "is_outlier"]]).join(moods)
    S["artists_display"] = S.artists.str.replace(";", ", ", regex=False)
    S["label"] = S.track_name + " – " + S.artists_display
    return S, X


@st.cache_resource(show_spinner=False)
def load_models():
    m = {}
    for key, path in {"valence": "p2_happiness/valence_best.joblib", "valence_ridge": "p2_happiness/valence_ridge.joblib",
                      "lr": "p3_genre/logistic_regression.joblib", "nb": "p3_genre/naive_bayes.joblib",
                      "hit": "p5_hit/hit_model.joblib", "mlp_genre": "p6_neural/mlp_genre.joblib"}.items():
        m[key] = joblib.load(MODELS / path)
    return m


@st.cache_data(show_spinner=False)
def load_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_scores():
    """Scores table with plain component names (the internal role prefix is removed for display)."""
    s = pd.read_csv(ROOT / "results" / "scores.csv")
    s["component"] = s.component.str.replace(r"^P\d\s+", "", regex=True).replace({"baseline": "Baseline (guessing)"})
    return s


@st.cache_data(show_spinner=False)
def load_neighbours():
    return pd.read_csv(MODELS / "p4_similar" / "neighbours.csv")


@st.cache_data(show_spinner=False)
def load_predictions(task):
    return pd.read_csv(ROOT / "results" / "predictions" / f"{task}.csv", index_col="song_id")


@st.cache_data(show_spinner=False)
def all_predictions():
    """Predict everything for every song once, so pages are instant."""
    S, X = load_songs()
    M = load_models()
    out = {}
    for key in ["lr", "nb", "mlp_genre"]:
        b = M[key]
        out[key] = pd.DataFrame(b["model"].predict_proba(X[b["features"]]), index=S.index, columns=b["model"].classes_)
    b = M["valence"]
    out["valence"] = pd.Series(np.clip(b["model"].predict(X[b["features"]]), 0, 1), index=S.index)
    r = M["valence_ridge"]
    out["valence_contrib"] = X[r["features"]] * r["model"].coef_
    b = M["hit"]
    out["hit"] = pd.Series(b["model"].predict_proba(X[b["features"]])[:, 1], index=S.index)
    return out


def score_of(component, model, task, metric):
    s = load_scores()
    row = s[(s.component == component) & (s.model == model) & (s.task == task)]
    return float(row[metric].iloc[0]) if len(row) else float("nan")


# ------------------------------------------------------------------ song picker + cards
def song_picker(label="Search a song", key="picker", default_label=None):
    """Searchable select box over 'song – artist' (most popular first). Remembers the choice across pages."""
    S, _ = load_songs()
    order = S.sort_values("popularity", ascending=False)
    ids = list(order.index)
    labels = dict(zip(order.index, order.label))
    if "song_id" not in st.session_state:
        if default_label is not None and (S.label == default_label).any():
            st.session_state.song_id = int(S.index[S.label == default_label][0])
        else:
            st.session_state.song_id = int(ids[0])
    idx = ids.index(st.session_state.song_id) if st.session_state.song_id in ids else 0
    sid = st.selectbox(label, ids, index=idx, format_func=lambda i: labels[i], key=key,
                       help="Type to search by song or artist name.")
    st.session_state.song_id = int(sid)
    return int(sid)


def genre_chip(genre):
    return f"<span class='ss-chip'><span class='ss-dot' style='background:{GENRE_COLORS[genre]}'></span>{genre}</span>"


def mood_mix_html(song_id):
    S, _ = load_songs()
    info = load_json("models/p7_moods/mood_names.json")
    row = S.loc[song_id]
    parts, labs = [], []
    for m in info["moods"]:
        p = float(row[f"p_{m}"])
        if p >= 0.01:
            parts.append(f"<div title='{m} {p:.0%}' style='width:{p*100:.1f}%;background:{info['colors'][m]}'></div>")
        if p >= 0.10:
            labs.append(f"{info['emoji'][m]} {m} {p:.0%}")
    return f"<div class='ss-mix'>{''.join(parts)}</div><div class='ss-mixlab'>{' · '.join(labs)}</div>"


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def radar(song_id, height=190, compare_id=None, labels=True):
    """Small radar chart of a song's 0-1 audio features. compare_id draws another song as a grey outline."""
    S, _ = load_songs()
    feats = RADAR_FEATURES + [RADAR_FEATURES[0]]
    theta = RADAR_LABELS + [RADAR_LABELS[0]]
    fig = go.Figure()
    if compare_id is not None:
        c = S.loc[compare_id]
        fig.add_trace(go.Scatterpolar(r=[c[f] for f in feats], theta=theta, line=dict(color="#9a9a95", width=1.5, dash="dot"),
                                      name="your song", hovertemplate="your song · %{theta}: %{r:.2f}<extra></extra>"))
    row = S.loc[song_id]
    color = GENRE_COLORS[row.track_genre]
    fig.add_trace(go.Scatterpolar(r=[row[f] for f in feats], theta=theta, fill="toself", line=dict(color=color, width=2),
                                  fillcolor=rgba(color, 0.25), name=row.track_name[:30],
                                  hovertemplate="%{theta}: %{r:.2f}<extra></extra>"))
    m = 42 if labels else 8
    fig.update_layout(height=height, margin=dict(l=m, r=m, t=22 if labels else 8, b=18 if labels else 8), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#3a3a38", linecolor="#3a3a38"),
                                 angularaxis=dict(gridcolor="#3a3a38", linecolor="#3a3a38", showticklabels=labels,
                                                  tickfont=dict(size=10, color=INK_2))),
                      font=dict(color=INK))
    return fig


def song_card(song_id, extra_html="", show_radar=True, show_mood=False, key=None, radar_height=180,
              compare_id=None, radar_labels=True, show_split=None):
    S, _ = load_songs()
    row = S.loc[song_id]
    if show_split is None:                       # main cards say whether the models saw this song in training
        show_split = show_radar and radar_labels and compare_id is None
    split_html = ""
    if show_split:
        split_html = ("<span class='ss-chip' title='The models never saw this song while training'>🆕 unseen test song</span>"
                      if row.split == "test" else
                      "<span class='ss-chip' title='The models learned from this song'>📚 training song</span>")
    with st.container(border=True):
        st.markdown(f"<div class='ss-title' title=\"{row.track_name}\">{row.track_name}</div>"
                    f"<div class='ss-artist'>{row.artists_display}</div>{genre_chip(row.track_genre)}{split_html}{extra_html}",
                    unsafe_allow_html=True)
        if show_mood:
            st.markdown(mood_mix_html(song_id), unsafe_allow_html=True)
        if show_radar:
            st.plotly_chart(radar(song_id, height=radar_height, compare_id=compare_id, labels=radar_labels),
                            config={"displayModeBar": False},
                            key=key or f"radar_{song_id}")


# ------------------------------------------------------------------ charts
def gauge(value, title, suffix="%", color=ACCENT, max_value=100, height=230, reference=None, ref_label=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=suffix, valueformat=".0f", font=dict(size=40, color=INK)),
        title=dict(text=title, font=dict(size=15, color=INK_2)),
        gauge=dict(axis=dict(range=[0, max_value], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
                   bar=dict(color=color, thickness=0.32), bgcolor="#2A2A28", borderwidth=0,
                   threshold=dict(line=dict(color=INK, width=3), thickness=0.85, value=reference) if reference is not None else None)))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
    return fig


def base_layout(fig, height=420, legend=True):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, size=13), margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
                      hoverlabel=dict(bgcolor="#2A2A28", font_color=INK, bordercolor=BORDER))
    fig.update_xaxes(gridcolor="#2C2C2A", zerolinecolor="#3a3a38", color=INK_2)
    fig.update_yaxes(gridcolor="#2C2C2A", zerolinecolor="#3a3a38", color=INK_2)
    return fig


def genre_prob_bar(probs, title="", height=300):
    """Horizontal bar of genre probabilities (pd.Series indexed by genre)."""
    p = probs.reindex(D.GENRES)[::-1]
    fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h", marker=dict(color=[GENRE_COLORS[g] for g in p.index], cornerradius=4),
                           text=[f"{v:.0%}" for v in p.values], textposition="outside", textfont=dict(color=INK),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showgrid=False, visible=False)
    base_layout(fig, height=height, legend=False)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK_2)), bargap=0.35)
    return fig


def feature_glossary():
    return ("Audio features come from Spotify: danceability, energy, valence (happiness), acousticness, speechiness "
            "(spoken words), liveness (live audience) and instrumentalness are 0–1 scores; loudness is in dB; tempo in BPM.")
