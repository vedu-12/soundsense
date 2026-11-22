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

## Block 10: P7 notebook – Mood Playlists  (commit see git log: "Block 10: P7 notebook")
- Built: `notebooks/P7_mood_playlists.ipynb`, `models/p7_moods/{kmeans.joblib, gmm.joblib, mood_names.json, moods.csv}` (mood label + GMM probabilities for all 4,267 songs), `results/figures/p7_moods/*.png`, `tests/test_block10.py`.
- Checks run: 37 tests pass (moods for every song, probabilities sum to 1, only mood features used, 3 clustering rows logged). Viewed dendrogram, mood map, moods-vs-genres heatmap: all sensible.
- Key numbers: k-means silhouette k=2..10: 0.364, 0.272, **0.254 (k=4)**, 0.246, …; no sharp elbow. Moods (k-means): Party 1,388, Chill 656, Sad 699, Intense 1,524. Silhouette / ARI vs genre: k-means 0.254 / 0.294; Ward (1,500 sample) 0.209 / 0.233; GMM 0.245 / 0.276. GMM agrees with k-means on 91.1% of songs; 23.2% of songs have a mixed mood (top prob < 0.8). Example: "River Flows In You" = 65% Chill / 35% Sad.
- Decisions I made: **k = 4** (silhouette best at k=2 but too coarse; little loss from 3→4; 4 nameable moods matching the brief's examples). Moods named automatically by matching centres to mood prototypes with the Hungarian algorithm (robust to cluster order). **GMM uses spherical covariance**: full covariance had the best BIC but silhouette 0.085, ARI 0.27 vs k-means and near-0/1 probabilities (useless mood mix); comparison of all four covariance types shown in the notebook. Ward chosen over average/complete linkage (average put 76% of songs in one cluster). Emoji only in the app (matplotlib fonts lack them).
- Brief coverage: all P7 items covered. Ticked.

## Block 11: P6 notebook – Neural Network & Model Arena  (commit see git log: "Block 11: P6 notebook")
- Built: `notebooks/P6_neural_net_arena.ipynb`, `models/p6_neural/{mlp_genre, mlp_hit, mlp_valence}.joblib`, `models/p6_neural/song_lab.json`, `results/bootstrap_ci.csv`, `results/bootstrap_winners.csv`, `results/predictions/{genre,hit,valence}.csv` (test predictions of every model, written by new `evaluate.save_predictions`), `results/figures/p6_arena/*.png`, `tests/test_block11.py`.
- Change to earlier notebooks: added `evaluate.save_predictions(...)` next to each `log_score` in P2–P5 (needed for bootstrap CIs and the app's per-model confusion matrices), re-ran P2–P5: **all logged scores identical to before (max abs diff 0.0)**. Data pipeline untouched.
- Checks run: 40 tests pass. Viewed network diagram, loss curve, leaderboard and bootstrap plots.
- Key numbers: MLP genre (64 ReLU, alpha 0.01, lr 0.01; CV macro F1 0.683): **test accuracy 0.728, macro F1 0.693**. MLP hit (64, alpha 0.1; CV AUC 0.765): test ROC-AUC 0.723, F1 0.236. MLP valence (32→16, alpha 0.1; CV R² 0.315): **test R² 0.322**, RMSE 0.184. Loss curve: val loss min at epoch 9 (0.77), 3.02 at epoch 300 (train 0.30). **Arena winners: genre MLP 0.728 (linear SVM 0.725), hit Random Forest AUC 0.784 (AdaBoost 0.757), valence MLP R² 0.322 (Lasso 0.295). Bootstrap (1,000 resamples): no winner is clearly better than its runner-up** (difference CIs: genre [−0.021, +0.027], hit [−0.002, +0.060], valence [−0.023, +0.078]). Ablations (linear SVM genre, CV acc): all 26 features 0.720, 7 PCs 0.676, 2 PCs 0.538, +mood 0.724; RF hit CV AUC 0.790 → 0.791 with mood.
- Decisions I made: grid tunes hidden sizes, alpha AND learning rate (brief + CLAUDE.md both satisfied). Hit MLP uses balanced sample weights (MLP has no class_weight). Bootstrap top-3 skips "Random Forest (calibrated)" (identical ranking to RF, so comparing them is meaningless). Ablation decisions use CV; test shown for information only. The mood feature for the ablation comes from k-means fit on all songs (unsupervised, no labels).
- Brief coverage: all P6 items covered. Ticked.

## Block 12: From-scratch algorithms src/scratch.py  (commit see git log: "Block 12: from-scratch algorithms")
- Built: `src/scratch.py` (`KNNClassifierScratch`, `GaussianNBScratch`, `LinearRegressionScratch` with normal equation + gradient descent, `KMeansScratch` with k-means++), "Built from scratch" sections in P2, P3, P4, P7 (re-executed), `tests/test_block12.py`.
- Checks run: 45 tests pass. All notebooks re-executed without errors; logged scores unchanged.
- Key numbers: kNN (k=13) test accuracy scratch 0.706 vs sklearn 0.701 (97.1% same predictions; differences = vote ties). Gaussian NB: identical predictions (0.649), probabilities equal to 1e-6. Linear regression: normal equation and gradient descent (lr 0.05, 5,000 steps) both R² 0.2947 = sklearn, weights equal to < 1e-4. k-means: scratch inertia 9,491 vs sklearn 9,541 (−0.5%), silhouette 0.253 vs 0.254.
- Finding: the scratch k-means found a slightly better local optimum (sklearn with n_init=50 finds it too) that groups songs differently (ARI 0.47): Chill+Sad merged, loud songs split by tempo. Decision: **keep the n_init=10 mood solution** (equal silhouette, inertia within 0.5%, separates by feel/valence, matches the brief's moods, GMM agrees 91%). Explained in P7 as a "k-means local optima" lesson.
- Brief coverage: adds the "normal equation / gradient descent" (P2) and implementation depth promised in the proposal.

## Block 13: Streamlit app  (commits "Block 13a…", "Block 13b…", "Block 13c: app polish")
- Built: `.streamlit/config.toml` (dark theme, accent #1DB954), `app/ui.py` (cached loaders, page header, "How it works" / "Behind the scenes" expanders, song card with radar, genre chip, GMM mood-mix bar, gauges, shared plot styling), `app/Home.py` (hero, "Analyze a song", "Explore SoundSense" grid of 7 feature cards), 7 pages (`Song Map`, `Happiness Meter`, `Genre Detector`, `Similar Songs`, `Hit Predictor`, `Song Lab and Model Arena` with two tabs, `Mood Playlists`), `tests/test_block13.py` (19 checks), `tests/take_screenshots.py`, `docs/screenshots/*.png` (9 screenshots incl. the Model Arena tab).
- Checks run: 64 tests pass. Every page runs with no exception in Streamlit's AppTest, incl. interactive states (cosine toggle, all 4 moods, arena tasks Hit/Happiness, map coloured by mood, Naive Bayes confusion matrix, a non-hit song); Song Lab test: energy 1.0 + loudness −2 dB + acousticness 0 → the network says heavy-metal. No teammate names or P1–P7 labels anywhere in `app/` (tested; the scores table's role prefixes are stripped for display). Reviewed screenshots of every page at 1440 px in 3 rounds.
- Fixed during review: 8-digit hex colours rejected by plotly (→ rgba); radar labels clipped (→ short labels, label-free mini radars with the seed song as a dotted outline); cramped recommendation cards (→ seed + "how close" bar chart, then a full-width row of 5); uneven card heights (fixed 2-line titles); gauge decimals; unexplained gauge marker (caption); map legend colliding with toolbar (legend below, toolbar hidden); song hard to find on the map (labelled arrow); low-contrast confusion-matrix text; Song Lab slider defaults rounded; reasons wording ("Long song", "Loud"); moods reused genre colours (→ separate meaning-based mood palette: Party amber, Chill teal, Sad indigo, Intense red, validated for colour-blind separation; P7 re-run so its figures match).
- Decisions I made: Home shows genre from the Model Arena winner (MLP), hit chance from the calibrated Random Forest (Hit Predictor's model) and happiness from P2's best model (Lasso), so Home and the Happiness Meter page agree (MLP vs Lasso valence difference is not significant by bootstrap). Song search remembers the chosen song across pages. Default example song: "Without Me – Eminem". The app only loads saved models/data (all per-song predictions computed once and cached).

## Block 14: Final verification + README  (commits "Block 14: final verification…" [tag v1.0] and "Block 14 fix: independent review findings" [tag v1.1])
- Built: full `README.md`, `docs/DEMO_SCRIPT.md`, final `docs/BRIEF_CHECKLIST.md` (136 items → notebook section → covered: all yes) with a proposal-coverage table, `tests/test_block14.py`.
- Fresh run: deleted `results/scores.csv`, `results/predictions/` and all `models/p*`, then `python -m src.data` → `python -m src.evaluate` → executed P1, P2, P3, P4, P5, P7, P6 with `jupyter nbconvert --execute --inplace` (15–116 s each, ≈5.5 min total): **all OK, every score identical to before (29 rows, max abs diff 0.0)**.
- Fresh clone check: `git clone` into a temp folder → new venv → `pip install -r requirements.txt` only → `streamlit run app/Home.py` served HTTP 200 and all 8 pages ran without exceptions (processed data + models come from the repo).
- Final leaderboard: genre MLP 0.728 (linear SVM 0.725, LR 0.715); hit Random Forest ROC-AUC 0.784 (AdaBoost 0.757); valence MLP R² 0.322 (Lasso 0.295). No winner clearly better than its runner-up (bootstrap).
- App launch: `streamlit run app/Home.py`.
- **Independent review** (separate subagent, fresh eyes): no must-fix issues; no leakage, test set never used for tuning, all notebooks follow the layout and match the briefs, screenshots clean, no names / P-labels, proposal fully delivered. Fixed everything it raised:
  1. 13 test songs are near-duplicates of training songs (identical audio features, slightly different title) → wording corrected in P1 (+ new detection cell), README note, and a P6 robustness check: removing them changes any score by ≤ 0.0065 (genre accuracies slightly *rise*). Pipeline not rebuilt (frozen; effect negligible); recorded here.
  2. App showed in-sample predictions for training songs without saying so → main song cards now show "🆕 unseen test song" / "📚 training song"; default song and all demo-script examples switched to test songs (HUMBLE., OUT OUT, In the Name of Love, Sexy Bitch, One Night in Tokyo).
  3. P7 k-means local optimum: already documented in §8; dendrogram text reworded to "consistent with" 4 clusters (the widest gap is at 2), Q1 updated.
  4. P2 Q3 wrongly said polynomial features "help a little" → corrected (they overfit).
  5. P6: added note that mood clusters were fit on all songs (label-free, negligible) in the ablation; P5 `hit_profile.json` and the hit/non-hit profile table now use training songs only (text numbers updated: −9.6 dB, ~18 s shorter).
  6. Minor: Home genre card names its model ("neural network"); P4 wording (section reference, "16 columns" description, results-table header); demo-script "mixed bar" wording; Chill emoji changed to 😌 (☕ rendered as a tiny glyph); README notes on popularity and train/test tags.
- Checks run: 75 tests pass; affected notebooks (P1, P2, P4, P5, P6, P7) re-executed; screenshots retaken.
- Decisions I made: tag `v1.0` was pushed before the review; the post-review final state is tagged `v1.1` (no force-push / tag rewrite).

## Extra (after Block 14): song preview button  (commit "App: add Spotify preview button")
- Built: a small speaker icon (🔊, borderless) next to the title on each page's main song card; clicking it opens a compact Spotify embed player (preview clip, chosen by Spotify) for that track_id.
- Checks run: 19 app tests pass; screenshot of the open player reviewed.
- Decisions I made: requested by the user, this **overrides the CLAUDE.md "no external images" rule** (the Spotify player shows the album cover, only after clicking). Needs internet; the rest of the app still works offline. Spotify decides which part of the song the preview plays; the start point can't be chosen.
