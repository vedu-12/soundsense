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
