# SoundSense Build Log

## Block 0: Setup  (commit see git log: "Block 0: setup")
- Built: folder structure, `requirements.txt`, `.gitignore`, `README.md` stub, `docs/BRIEF_CHECKLIST.md`, `tests/test_block0.py`, `.venv` (Python 3.14.6), Playwright Chromium, GitHub CLI 2.105.0 (installed via winget).
- Checks run: `pytest tests/test_block0.py` (structure, docs, imports) passed. Library versions: pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, plotly 6.9.0, streamlit 1.67.0, joblib 1.6.0, scipy 1.18.1.
- Key numbers: n/a
- Decisions I made: Python 3.14 is the only interpreter on the machine; all libraries installed fine on it. Requirements pin major-version ranges. Repo private (default).
- Differences between the documents and CLAUDE.md (followed CLAUDE.md, recorded here):
  1. Briefs list inputs without `time_signature`; proposal and CLAUDE.md include it (one-hot, or drop if almost constant). Decided in Block 3.
  2. P4 brief says "plot **test** accuracy vs k"; CLAUDE.md says CV on train (never tune on test). The notebook plots **cross-validated** accuracy vs k and uses the test set once for the chosen k.
  3. P6 brief says tune hidden neurons and **learning rate**; CLAUDE.md says hidden size and **alpha**. The notebook tunes all three with a small CV grid.
  4. P1 brief says "duplicate track_ids are removed"; CLAUDE.md additionally removes tracks in more than one chosen genre and same name+artist duplicates. Both are done.
  5. Hit threshold "≥ 70" in briefs/proposal may change by the CLAUDE.md fallback rule if the hit rate is under 5% (decided in Block 2).
- Repo URL: https://github.com/vedu-12/soundsense (private)

## Block 1: Get the data  (commit see git log: "Block 1: get the data")
- Built: `data/raw/spotify_tracks.csv` (20.1 MB, from the Hugging Face URL in CLAUDE.md, first try), `tests/test_block1.py`.
- Checks run: all tests pass (6). Raw CSV has 114,000 rows.
- Key numbers: 114,000 rows; 20 columns (+ unnamed index): track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre. **114 unique genres** (briefs/proposal say 125; the real file has 114). Missing values: 1 each in artists, album_name, track_name (same row). Duplicate track_ids: 24,259.
- Decisions I made: None (no cleaning yet). Raw CSV is git-ignored; test skips if absent so a fresh clone still passes.

## Block 2: Choose the 6 genres  (commit see git log: "Block 2: choose the 6 genres")
- Built: `tests/test_block2.py`; analysis run as a script (results below; the same tables are reproduced in the P1 notebook).
- Checks run: all tests pass (7). Every one of the 114 genres has exactly 1,000 rows in the raw file.
- Candidates compared (after removing cross-genre tracks, duplicate track_ids and same name+artist duplicates):
  | set (metal variant) | tracks in >1 chosen genre | songs kept | hit rate ≥70 |
  |---|---|---|---|
  | heavy-metal | 16 | 4,267 | 8.9% |
  | black-metal | 16 | 4,273 | 8.9% |
  | death-metal | 16 | 4,256 | 8.9% |
  | metal | 28 | 4,085 | 13.7% |
- Mean features (heavy-metal set): energy classical 0.20 / jazz 0.37 / country 0.62 / hip-hop 0.69 / edm 0.75 / heavy-metal 0.87; acousticness classical 0.91 / jazz 0.67 / country 0.29 / hip-hop 0.22 / edm 0.12 / heavy-metal 0.02; speechiness highest for hip-hop (0.14); instrumentalness highest for classical (0.60). The genres clearly differ.
- **Chosen genres: classical, country, edm, heavy-metal, hip-hop, jazz.** Reason: the default list from CLAUDE.md; `heavy-metal` picked as the metal genre because it has the fewest overlaps and loses the fewest songs to duplicates (986 of 1,000 kept) while sounding typical "metal" (unlike black/death metal, which are more extreme sub-styles).
- **Hit threshold: popularity ≥ 70 (unchanged from the proposal)**: hit rate 8.9% ≥ 5%, so no fallback needed.
- Key numbers: ~4,267 songs after cleaning, **fewer than the proposal's "~6,000"**, because many songs appear several times with different track_ids (e.g. jazz has 519 same name+artist duplicates: Christmas songs on many compilations; "Frosty The Snowman" by Ella Fitzgerald appears 34 times). Removing them is required by CLAUDE.md and prevents the same song landing in both train and test (leakage).
- Decisions I made: heavy-metal as the metal genre; keep threshold 70.

## Block 3: Data pipeline src/data.py  (commit see git log: "Block 3: data pipeline")
- Built: `src/data.py` (`build_dataset`, `load_splits`, `load_all_songs`, `load_feature_names`, `load_scaler`, `scale_raw_features`; constants `GENRES`, `HIT_THRESHOLD=70`, `RANDOM_STATE=42`), `python -m src.data`, `data/processed/{songs.csv, scaler.joblib, feature_names.json, config.json}`, `tests/test_block3.py`, `pytest.ini`.
- Checks run: 15 tests pass: popularity/hit/text/genre not in features; 4,267 songs; test share 20%; genre proportions train vs test within 1%; no NaNs; train means = 0 and stds = 1; hit label = popularity ≥ 70; no duplicate track_ids or name+artist; train/test disjoint.
- Key numbers: 114,000 raw → 6,000 in chosen genres → −16 tracks in >1 chosen genre → −86 duplicate track_ids → −1,615 same name+artist duplicates → **4,267 songs** (heavy-metal 986, classical 806, hip-hop 727, edm 668, country 607, jazz 473). Train 3,413 / test 854. Hit rate train 9.2% / test 7.6% (split is stratified by genre only, as specified). 26 features.
- Decisions I made:
  - Features (26): 10 continuous audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) + explicit (0/1) + mode (0/1; a two-category one-hot needs one column) + key one-hot (11 columns, key C dropped as baseline, so the normal equation is not singular) + time_signature one-hot (time_sig_3, time_sig_5, time_sig_other; 4/4 is the baseline). **time_signature is kept, not dropped**: 4/4 is 88% overall but only 69% for classical, so it carries genre information; values 0 and 1 (1.2%) merged into "other".
  - All 26 columns are standard-scaled (incl. one-hot), fit on train only.
  - `AUDIO_FEATURES` (the 10 continuous ones) is also exported: distance-based **unsupervised** parts (PCA Song Map, DBSCAN, recommender) use these, because 11 scaled key dummies would otherwise dominate distances (two songs in different keys would look far apart regardless of sound). Supervised models use all 26 features.
  - `load_all_songs()` returns a tuple (songs table, scaled X) with the same index, since the table has both original and scaled values of the same feature names.
- **Pipeline is now FROZEN.**

## Block 4: Shared evaluation src/evaluate.py  (commit see git log: "Block 4: shared evaluation")
- Built: `src/evaluate.py` (`classification_metrics`, `regression_metrics`, `clustering_metrics`, `log_score`, `load_scores`, `plot_confusion_matrix`, `bootstrap_ci`, `log_baselines`; `python -m src.evaluate` logs the baselines), `results/scores.csv`, `tests/test_block4.py`.
- Checks run: 18 tests pass. Dummy genre accuracy 18.0% (≈ 1/6). Rerunning `python -m src.evaluate` keeps 4 rows (no duplicates); `log_score` replace-on-rerun tested on a temp file (1 row after two identical logs).
- Key numbers: genre dummy accuracy 0.180 / macro F1 0.178; hit dummy F1 0.027, ROC-AUC 0.465; valence dummy RMSE 0.224, R² −0.002; popularity dummy RMSE 26.4.
- Decisions I made: besides the genre dummy, I also logged dummy baselines for hit, valence and popularity (component "baseline"), so the Model Arena can show "beats guessing" for every task; so scores.csv has 4 rows (one per task) instead of 1. For binary tasks precision/recall/F1 are for the hit class; `macro_f1` is also stored. Added `bootstrap_ci` here for reuse in P6.

## Block 5: P1 notebook – The Data & Song Map  (commit see git log: "Block 5: P1 notebook")
- Built: `notebooks/P1_data_song_map.ipynb` (executed, outputs saved), `src/style.py` (shared colour-blind-checked genre palette + plot style), `models/p1_song_map/{pca.joblib, song_map.csv, song_map_info.json}`, `results/figures/p1_song_map/*.png` (8 figures), `tests/nb_utils.py`, `tests/test_block5.py`.
- Checks run: 21 tests pass (notebook saved with outputs and no errors, required sections present, song map has 4,267 rows, outlier share < 10%, figures exist). Looked at every plot: heatmap, distributions, scree, loadings, maps all render and make musical sense.
- Key numbers: corr(energy, loudness) = +0.85, corr(energy, acousticness) = −0.87. PCA: PC1 36.5%, PC2 14.4%; **7 components for ≥ 90%**. PC1 = loudness/energy vs acousticness/instrumentalness ("intensity"); PC2 = danceability/valence vs duration/liveness. DBSCAN (min_samples=20, eps=2.56 from automatic k-distance elbow): 1 dense cluster, **59 noise songs (1.4%)**, 35 of them classical (mostly long Indian devotional chants).
- Decisions I made: PCA and DBSCAN use the 10 continuous audio features (not the one-hot key/time-signature columns), see Block 3. **PCA is fit on training songs only** and applied to all songs, so P6 can use the components as model inputs without leakage. P1 reads the raw CSV once (only to show "before cleaning" counts); everything else uses `load_all_songs()`/`load_splits()`. Notebook test checks saved outputs; set `RUN_NOTEBOOKS=1` to re-execute notebooks inside pytest (slow), full re-execution is done in Block 14. eps chosen by the max-distance-to-diagonal elbow rule on the normalised k-distance curve.
- Brief coverage: all P1 items (7 steps, 5 concepts, 4 "show" items, 5 questions) covered; ticked in BRIEF_CHECKLIST.md.

## Block 6: P2 notebook – Happiness Meter  (commit see git log: "Block 6: P2 notebook")
- Built: `notebooks/P2_happiness_meter.ipynb`, `models/p2_happiness/{valence_best.joblib, valence_ridge.joblib}` (dicts: model, name, feature list), `results/figures/p2_happiness/*.png`, `tests/test_block6.py`.
- Checks run: 24 tests pass (notebook OK, valence not an input of saved models, test R² in a plausible range, 4 valence + 4 popularity rows logged, best beats the mean baseline by > 0.1 R²).
- Key numbers (test set): valence: Linear R² 0.295 / RMSE 0.188; Poly-2 R² 0.104 (train 0.436 → overfits); Ridge (alpha 5.18) 0.295; Lasso (alpha 0.00087) 0.295 (dropped key_F, key_B, time_sig_5). Popularity: best R² 0.107 (Lasso), RMSE ≈ 25.0. Strongest weights: energy +0.12, danceability +0.11, acousticness +0.10 (given energy).
- Decisions I made: "best" valence model chosen by 5-fold CV R² on the training set (Lasso, 0.297 vs 0.296 for linear/Ridge: a tie in practice), not by test score. Also saved the Ridge model so the app can show per-feature contributions (weight × scaled value). For popularity, valence is allowed as an input (it's a feature, not the target there).
- Brief coverage: all P2 items covered (normal equation / gradient descent are explained here and implemented from scratch in Block 12). Ticked.

## Block 7: P3 notebook – Genre Detector  (commit see git log: "Block 7: P3 notebook")
- Built: `notebooks/P3_genre_detector.ipynb`, `models/p3_genre/{logistic_regression.joblib, naive_bayes.joblib}` (dicts with model, name, feature list), `results/figures/p3_genre/*.png` (incl. separate confusion matrices for the app), `tests/test_block7.py`. Confusion-matrix helper: grid lines removed.
- Checks run: 27 tests pass. Accuracy well above baseline and below 95% (no leakage signs). Looked at the confusion matrices: errors are musically sensible.
- Key numbers (test): **Logistic regression (C=5.62, 5-fold CV on macro F1) accuracy 0.715, macro F1 0.685**; **Gaussian NB (10 audio features, var_smoothing=0.1) accuracy 0.649, macro F1 0.609**. Top confusions (LR): hip-hop→edm 32, edm→hip-hop 21, jazz→country 19. Learning curve: NB beats LR below ~250 training songs (30 songs: 0.534 vs 0.448), LR wins with more data.
- **Bug found and fixed:** Gaussian NB on all 26 features scored only 40.5% (CV) because some 0/1 columns never vary inside a genre (explicit in classical, time_sig_5 in edm), giving variance 1e-9 and exploding likelihoods. Decision: NB uses the 10 continuous audio features with var_smoothing tuned by CV (64.3% CV). Shown in the notebook as a teaching point; LR on the same 10 features (69.9% CV) is also reported, so the comparison is fair.
- Decisions I made: models saved as dicts with their feature list (NB and LR use different columns). `multi_class` arg no longer exists in scikit-learn 1.9; lbfgs LogisticRegression is multinomial (softmax) by default.
- Brief coverage: all P3 items covered (incl. learning-curve evidence for "When can NB beat LR?" and within-genre correlation check for "is the assumption true?"). Ticked.

## Block 8: P4 notebook – Similar Songs  (commit see git log: "Block 8: P4 notebook")
- Built: `notebooks/P4_similar_songs.ipynb`, `models/p4_similar/{recommender.joblib, neighbours.csv, svm_best.joblib, knn_best.joblib}`, `results/figures/p4_similar/*.png`, `tests/test_block8.py`.
- Checks run: 31 tests pass (neighbours table complete, no self-recommendations, classifiers 60–95%, scores logged). Inspected recommendations for 3 well-known songs: musically sensible (e.g. Eminem "Without Me" → other Eminem songs).
- Key numbers: top-5 same-genre share Euclidean **61.5%**, cosine **62.4%** (chance 17.5%; all 26 features 53.7% / 54.5%; unscaled 27.0%). kNN classifier: best k = 13 (10 audio features, CV 0.696), **test accuracy 0.701 / macro F1 0.671**. SVM linear (C=1, CV 0.720): **test 0.725 / 0.697** (best genre model so far). SVM RBF (C=10, gamma=0.01, CV 0.718): test 0.714 / 0.684. Tuning time: linear 7 s, RBF 11 s (20 combos × 5 folds), so no subsampling needed.
- Decisions I made: recommender and kNN classifier use the 10 audio features (chosen by evidence in the notebook: higher same-genre share and higher CV accuracy than with the 16 one-hot columns); SVM uses all 26. Accuracy vs k is plotted with **CV accuracy** (not test accuracy as the brief says; see Block 0 note). Example songs: Beast In Black "One Night in Tokyo", Yiruma "River Flows In You", Eminem "Without Me" (well-known; my first choice, the most popular song per genre, gave odd crossover examples). Neighbours are pre-computed into `neighbours.csv` so the app is instant. pandas 3 Arrow strings need `.to_numpy(dtype=str)` before 2-D indexing. joblib prints harmless "memmapping folder" KeyErrors on Windows at kernel shutdown (not in notebook outputs).
- Brief coverage: all P4 items covered. Ticked.

## Block 9: P5 notebook – Hit Predictor  (commit see git log: "Block 9: P5 notebook")
- Built: `notebooks/P5_hit_predictor.ipynb`, `models/p5_hit/{hit_model.joblib (calibrated RF + threshold), tree_depth3.joblib, hit_profile.json}`, `results/figures/p5_hit/*.png`, `tests/test_block9.py`.
- Checks run: 34 tests pass (popularity not an input; ROC-AUC 0.6–0.95; calibrated mean probability ≈ hit rate; all P5 models beat the dummy). Viewed tree plot, ROC curves and reliability plot.
- Key numbers (test, 65 hits of 854): Decision tree (Gini, depth 5; CV AUC 0.737): F1 0.256, ROC-AUC 0.729. **Random Forest** (300 trees, min_samples_leaf 10; CV AUC 0.790): F1 0.262, **ROC-AUC 0.784**. AdaBoost (200 stumps, lr 0.5; CV AUC 0.782): F1 0.241, ROC-AUC 0.757. Calibrated RF (isotonic, 5-fold) with threshold 0.18 (picked by F1 on training out-of-fold predictions): F1 0.266, precision 0.185, recall 0.477; Brier 0.110 → 0.066. "Always not-hit" accuracy 92.4%. Hit rate by genre: edm 23.7%, hip-hop 17.7%, country 10.9%, jazz 4.0%, classical 0.9%, heavy-metal 0%.
- Decisions I made: tuned by CV ROC-AUC (threshold-free, stable with 314 training hits). AdaBoost imbalance via balanced sample weights (AdaBoost has no class_weight). Best model for calibration chosen by CV AUC (RF). Decision threshold chosen on training CV predictions, not the test set. Hit threshold stays 70.
- Brief coverage: all P5 items covered, including the honest-limit note. Ticked.
