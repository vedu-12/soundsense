"""Happiness Meter: predicted valence vs the real value, and which features push it up or down."""
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find app/ui.py
import ui  # noqa: E402

ui.setup_page("Happiness Meter", "😊")
S, X = ui.load_songs()
P = ui.all_predictions()
M = ui.load_models()

ui.page_header("😊", "Happiness Meter", "How happy does a song sound? Our model guesses from the sound alone.")
sid = ui.song_picker("Pick a song", key="happy_picker", default_label="Without Me – Eminem")
song = S.loc[sid]
pred, real = 100 * P["valence"].loc[sid], 100 * song.valence

left, mid, right = st.columns([1, 1.25, 1.6], gap="large")
with left:
    ui.song_card(sid, key="happy_card", radar_height=200)
with mid:
    st.plotly_chart(ui.gauge(pred, "Predicted happiness", reference=real), config={"displayModeBar": False}, key="happy_gauge")
    a, b = st.columns(2)
    with a:
        ui.metric_card("Real value", f"{real:.0f}%", "Spotify's valence")
    with b:
        ui.metric_card("Off by", f"{abs(pred - real):.0f} pts", "typical miss: 19")
    st.caption("White mark on the gauge = the real value.")
with right:
    contrib = P["valence_contrib"].loc[sid]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
    nice = [f.replace("_", " ") for f in top.index]
    colors = ["#3987e5" if v > 0 else "#d95926" for v in top.values]
    fig = go.Figure(go.Bar(x=top.values * 100, y=nice, orientation="h", marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{y}: %{x:+.1f} points<extra></extra>"))
    ui.base_layout(fig, height=330, legend=False)
    fig.update_layout(title=dict(text="What pushes this song's happiness <span style='color:#3987e5'>up</span> or "
                                      "<span style='color:#d95926'>down</span>", font=dict(size=14, color=ui.INK_2)))
    fig.update_xaxes(title="change in predicted happiness (points)", zeroline=True, zerolinecolor="#6a6a66")
    st.plotly_chart(fig, config={"displayModeBar": False}, key="happy_contrib")
    st.caption("Each bar = the feature's weight × how far this song is from the average song.")

ui.how_it_works(
    ["Inputs: every audio feature **except** valence (that's what we predict).",
     "**Linear regression**: happiness = w₁·danceability + w₂·energy + … + b. Each weight says how much a feature pushes happiness up or down.",
     "The weights are chosen by **least squares**: make the total squared error on the training songs as small as possible.",
     "**Ridge** and **Lasso** add a penalty on big weights; Lasso can drop useless features entirely. The penalty strength is picked by cross-validation.",
     "Result: sound explains about 30% of happiness; danceability and energy make songs sound happier."],
    ["Linear regression", "Least squares", "Polynomial features", "Ridge (L2)", "Lasso (L1)", "Bias–variance", "R²"])

name = M["valence"]["name"]
ui.behind_the_scenes(
    f"{name}, chosen by 5-fold cross-validation on the training songs (bars use the Ridge model's weights).",
    [("Test R²", f"{ui.score_of('Happiness Meter', name, 'valence', 'r2'):.2f}"),
     ("Test RMSE", f"{ui.score_of('Happiness Meter', name, 'valence', 'rmse'):.2f}"),
     ("Polynomial model R²", f"{ui.score_of('Happiness Meter', 'Polynomial (deg 2) + linear', 'valence', 'r2'):.2f}"),
     ("Popularity R² (bonus)", f"{ui.score_of('Happiness Meter', name, 'popularity', 'r2'):.2f}")],
    ui.FIGS / "p2_happiness" / "coefficients.png", "Model weights: which features make songs sound happier or sadder",
    note="R² = share of the differences in happiness the model explains (1 = perfect, 0 = guessing the average). "
         "Popularity is much harder to predict: it depends on fame and marketing, not just sound.")
