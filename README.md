# 🎵 SoundSense

**Genre detection, happiness prediction, hit prediction, mood playlists and song recommendations, using only how a song sounds.**

SoundSense is our BITS F464 Machine Learning project (BITS Pilani, K K Birla Goa Campus, 2026–27). It uses the audio features of the public [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) to:

1. detect a song's **genre** (classification),
2. predict how **happy** it sounds: Spotify's `valence`, 0–1 (regression),
3. predict whether it will be a **hit**: popularity ≥ 70 (imbalanced binary classification),
4. group songs into **mood playlists** without any labels (clustering),
5. recommend **similar songs** (k-nearest neighbours),

and shows everything in an interactive **Streamlit app**.

---

## Quick start

```bash
git clone https://github.com/vedu-12/soundsense.git
cd soundsense
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

The processed data, trained models and results are in the repo, so **the app runs straight after `pip install`**, with no training needed. It opens at http://localhost:8501.

Tested with Python 3.14 and the library versions in `requirements.txt`.

## Rebuild everything from scratch (optional)

The raw CSV is not stored on GitHub (20 MB). To reproduce every number:

```bash
# 1. download the raw data (≈20 MB)
curl -L -o data/raw/spotify_tracks.csv https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv

# 2. build the cleaned dataset + split + scaler           (data/processed/)
python -m src.data

# 3. log the "guessing" baselines                          (results/scores.csv)
python -m src.evaluate

# 4. run the notebooks in this order (P7 before P6: the arena uses the moods)
jupyter nbconvert --to notebook --execute --inplace notebooks/P1_data_song_map.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P2_happiness_meter.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P3_genre_detector.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P4_similar_songs.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P5_hit_predictor.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P7_mood_playlists.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/P6_neural_net_arena.ipynb

# 5. run all checks
pytest

# 6. launch the app
streamlit run app/Home.py
```

The whole rebuild takes about 10 minutes on a laptop. Every random step uses `random_state=42`, so results are identical on every run.

## Who owns what

Each teammate owns one component: one notebook and one app page. **Read your notebook top to bottom**: every notebook starts with *What this part does* and *Concepts in plain English*, explains every plot, and ends with a *Results summary* and *Questions you may be asked*.

| Role | Owner | Component | Notebook | App page |
|---|---|---|---|---|
| P1 | Aryan Danny | The Data & Song Map | `notebooks/P1_data_song_map.ipynb` | 🗺️ Song Map |
| P2 | Gunneet Singh | Happiness Meter | `notebooks/P2_happiness_meter.ipynb` | 😊 Happiness Meter |
| P3 | Shreyash Lakshmi Narayan | Genre Detector | `notebooks/P3_genre_detector.ipynb` | 🎸 Genre Detector |
| P4 | Amithav Sachin | Similar Songs | `notebooks/P4_similar_songs.ipynb` | 🎧 Similar Songs |
| P5 | Vedang Chinhalli | Hit Predictor | `notebooks/P5_hit_predictor.ipynb` | 📈 Hit Predictor |
| P6 | Krish Jai Ganesh | Neural Network & Model Arena | `notebooks/P6_neural_net_arena.ipynb` | 🧪 Song Lab & Model Arena |
| P7 | Palakollu Sai Nikhil Reddy | Mood Playlists | `notebooks/P7_mood_playlists.ipynb` | 🌈 Mood Playlists |

The live-demo plan is in [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## The data in one paragraph

114,000 tracks across 114 genres → keep 6 clearly different genres (**classical, country, edm, heavy-metal, hip-hop, jazz**) → remove tracks listed under two of them, duplicate track IDs and re-releases of the same song (same name + artist) → **4,267 unique songs**. Inputs: 26 features (10 continuous audio features, explicit, mode, one-hot key and time signature). `popularity` is **never** an input; it only defines the hit label (popularity ≥ 70, 8.9% of songs). One stratified 80/20 split (3,413 train / 854 test), shared by every notebook; the scaler is fit on the training songs only.

## Results summary (test set, 854 songs)

| Task | Best model | Score | Other models | Baseline |
|---|---|---|---|---|
| **Genre** (6 classes) | Neural network (MLP) | accuracy **0.728**, macro F1 0.693 | linear SVM 0.725 · logistic regression 0.715 · RBF SVM 0.714 · kNN 0.701 · Naive Bayes 0.649 | random 0.180 |
| **Hit** (popularity ≥ 70) | Random Forest | ROC-AUC **0.784** (calibrated: F1 0.266) | AdaBoost 0.757 · decision tree 0.729 · MLP 0.723 | random 0.465 |
| **Happiness** (valence) | Neural network (MLP) | R² **0.322**, RMSE 0.184 | Lasso / Ridge / linear 0.295 · polynomial 0.104 | mean 0.00 |
| Popularity (bonus) | Lasso | R² 0.107 | linear 0.103 · Ridge 0.106 | mean 0.00 |
| **Mood playlists** | k-means (k = 4) | silhouette 0.254, ARI vs genre 0.294 | GMM 0.245 · Ward 0.209 | |
| **Recommender** | kNN (cosine) | 62.4% of top-5 share the genre | Euclidean 61.5% | random 17.5% |

**Bootstrap check (1,000 resamples):** none of the three task winners is clearly better than its runner-up: their 95% intervals overlap. Several models are equally good; all clearly beat the baselines.

Key insights: hip-hop ↔ edm is the most common genre confusion; danceable, energetic, louder, less acoustic and shorter songs are more often hits (but sound explains only part of popularity); moods (Party, Chill, Sad, Intense) only partly follow genres.

## Folder guide

```
soundsense/
├── app/                    Streamlit app: Home.py, pages/ (one per component), ui.py (shared look)
├── data/
│   ├── raw/                original CSV (not in git; see "Rebuild")
│   └── processed/          songs.csv (clean songs + split), scaler.joblib, feature_names.json, config.json
├── docs/                   proposal, team briefs, BUILD_LOG.md, BRIEF_CHECKLIST.md, DEMO_SCRIPT.md, screenshots/
├── models/                 trained models, one folder per component (p1_song_map … p7_moods)
├── notebooks/              P1–P7 notebooks, saved with all outputs
├── results/
│   ├── scores.csv          every model's test scores (the Model Arena)
│   ├── predictions/        every model's test predictions (bootstrap + confusion matrices)
│   └── figures/            key plots, one folder per component
├── src/
│   ├── data.py             data pipeline: load_splits(), load_all_songs()
│   ├── evaluate.py         shared metrics, log_score(), confusion matrix, bootstrap
│   ├── scratch.py          kNN, Naive Bayes, linear regression, k-means written from scratch in NumPy
│   └── style.py            shared colours
├── tests/                  pytest checks, one file per build block; take_screenshots.py
└── requirements.txt
```

## Course concepts covered

Preprocessing, encoding, scaling, train/test split · PCA · DBSCAN · linear regression, least squares, polynomial features, Ridge, Lasso · logistic regression vs Naive Bayes (discriminative vs generative) · kNN, distance metrics, SVM (linear + RBF kernel) · decision trees, Random Forest, AdaBoost, class imbalance, ROC-AUC, calibration · neural networks (MLP), backpropagation, early stopping · k-means, hierarchical clustering, GMM via EM, silhouette, ARI · cross-validation everywhere, bootstrap confidence intervals · from-scratch NumPy implementations checked against scikit-learn.

## Notes on decisions

All decisions, numbers and differences from the original proposal are recorded in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md). The main ones:
- 4,267 songs instead of the proposal's "about 6,000": many songs appeared several times (re-releases, compilations) and were removed.
- The hit threshold stayed at **popularity ≥ 70** as in the proposal (8.9% hits).
- The test set was used once per final model; all tuning used 5-fold cross-validation on the training set.
- Known small imperfection: 13 test songs (1.5%) are the same recording as a training song under a slightly different title. Removing them changes every score by at most 0.65 points (checked in P6), so no conclusion depends on it.
- Popularity regression is the P2 bonus task (4 linear models, best R² 0.11); it is reported in `results/scores.csv` but not shown in the app's Model Arena, which covers the three main tasks.
- In the app, each main song card says whether the song is an **unseen test song** or a **training song** (the models learned from training songs, so their predictions there are optimistic). The demo script uses test songs only.

*Data: Spotify Tracks Dataset by maharshipandya (Hugging Face / Kaggle, BSD licence).*
