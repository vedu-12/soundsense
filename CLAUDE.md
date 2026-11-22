# SoundSense – Build Spec for Claude Code

This file tells you (Claude Code) everything about this project. Read it fully before doing anything.

## 1. What we are building

**SoundSense** is a college Machine Learning project (BITS F464, BITS Pilani Goa). Using only a song's audio features from the public Spotify Tracks Dataset, it:

1. detects a song's **genre** (classification),
2. predicts how **happy** it sounds, i.e. `valence` 0–1 (regression),
3. predicts whether it will be a **hit**, i.e. popularity ≥ 70 (binary classification),
4. groups songs into **mood playlists** (clustering, no labels),
5. recommends **similar songs** (kNN).

Everything is shown in a **Streamlit web app**.

Seven teammates each own and present one component (P1–P7). **One person builds all the code with you.** The other six must learn their part from the notebooks, so **the notebooks must be very clearly explained in plain English.** That is as important as the code working.

### Reference documents (already in this folder)

Two documents sit in this project folder (moved to `docs/` in Block 0):

1. **The project proposal** (e.g. `SoundSense_Project_Proposal.pdf`): what we promised the course. The final project must deliver everything in its Objectives, Dataset, Syllabus Concepts Covered, Methodology and Evaluation Metrics sections.
2. **The team briefs** (e.g. `SoundSense – Team Briefs`, as PDF or other export): one section per teammate (P1 Aryan, P2 Gunneet, P3 Shreyash, P4 Amithav, P5 Vedang, P6 Krish, P7 Nikhil). Each section lists "How it works in our project", "Concepts you must understand", "What you will show" and "Questions you should be ready for".

How to use them:
- **Each notebook must cover everything in its owner's brief**: every step in "How it works", every concept in "Concepts you must understand", and every item in "What you will show" must appear in that notebook (as code, a plot or an explanation). The notebook's "Questions you may be asked" section must answer the questions listed in the brief.
- **Use the same wording and names** as the briefs (component names, app page names like "Song Lab" and "Model Arena", mood names style) so teammates recognise their part.
- If a document and this CLAUDE.md disagree, follow CLAUDE.md for code structure and rules, and **record the difference** in the build log.
- If a document can't be read, try another way first (e.g. install `pdfplumber` or `pypdf`, or use `pdftotext`). Only if all fail, stop and ask me to export it as text.

## 2. HOW YOU MUST WORK (most important rules)

**AUTONOMOUS MODE.** The person building this is very busy. Build the WHOLE project, Block 0 to Block 14, back to back, without waiting for approval between blocks. You do the checking and verification yourself.

1. **Only stop and ask me for these things:**
   - the GitHub login one-time code (Block 0), and git `user.name` / `user.email` if they are not set;
   - a real blocker you cannot solve yourself after 2–3 genuine attempts (explain what you tried);
   - nothing else. For every other decision, pick the sensible default given in this file, record it in the build log, and continue.
2. **Verify every block yourself before moving on:**
   - Write automated checks in `tests/test_block<N>.py` (pytest) for that block's must-be-true facts (e.g. `popularity` not in features, train/test genre proportions within 1%, no NaNs, scores above the dummy baseline, every model file exists, every notebook in `notebooks/` executes without errors). Run `pytest` (all tests so far, not just the new ones) and fix failures before continuing.
   - Run your own sanity checks on results (numbers in plausible ranges, plots render, results make musical sense). If something looks suspicious (e.g. accuracy > 95%, R² near 1, or below the dummy baseline), investigate for data leakage or bugs and fix it.
   - Check the notebook against its owner's team brief (see Reference documents) and add anything missing.
3. **After each block, append a short entry to `docs/BUILD_LOG.md`** (don't stop or wait):

```
## Block <N>: <name>  (commit <hash>)
- Built: <files>
- Checks run: <tests + sanity checks, all passed / what was fixed>
- Key numbers: <row counts, accuracy, R², etc.>
- Decisions I made: <defaults chosen and why, or "None">
- Brief coverage: <for notebook blocks: every brief item covered? list any gaps fixed>
```

4. **Run your own code before logging.** Execute notebooks/scripts and fix errors first. Log real numbers, never guesses.
5. **The data pipeline is FROZEN after Block 3.** If a later block truly needs a change to it, make the change, rebuild everything downstream, re-run all tests, and record it clearly in the build log.
6. **Default decisions (use these, don't ask):**
   - Genres: pick 6 that sound clearly different, using the dataset's actual names, preferring roughly classical, a metal genre, hip-hop, edm, jazz, country, each with ~1,000 songs and little overlap.
   - Hit threshold: popularity ≥ 70. If the hit rate within the 6 genres is below 5%, use ≥ 60 instead (and if still below 5%, ≥ 50), and record it prominently in the build log and final summary, because the proposal says 70.
   - Repo visibility: private.
   - Anything else: the simplest option that matches the proposal and team briefs.
7. **Keep it simple.** scikit-learn, pandas, numpy, matplotlib/seaborn, plotly (for the app only), streamlit, joblib, pytest, playwright (for app screenshots only). No deep learning frameworks (use scikit-learn's `MLPClassifier`/`MLPRegressor`). No unnecessary abstractions.
8. **Random seed = 42 everywhere** (`random_state=42`, `np.random.seed(42)`), so results are repeatable.
9. **Never use the test set for tuning.** Tune with cross-validation on the training set only. Test set is used once per final model.
10. **Commit and push to GitHub at the end of every block**, after its checks pass. Commit message: `Block <N>: <block name>` (fixes: `Block <N> fix: <what>`). Save notebooks WITH their outputs (plots, tables) so teammates can read them on GitHub without running anything. Never force-push.
11. **Long sessions:** if your context is getting full, re-read this CLAUDE.md and `docs/BUILD_LOG.md` before continuing, so you never lose the rules or progress. If the session ends midway, a new session should read both and continue from the next unfinished block.

## 3. Project structure

```
soundsense/
├── CLAUDE.md
├── README.md
├── docs/                               # proposal + team briefs (never edited), BUILD_LOG.md, BRIEF_CHECKLIST.md, DEMO_SCRIPT.md, screenshots/
├── tests/                              # pytest checks, one file per block
├── requirements.txt
├── data/
│   ├── raw/spotify_tracks.csv          # downloaded, never edited
│   └── processed/                      # created by src/data.py (Block 3)
├── src/
│   ├── data.py                         # cleaning, encoding, scaling, split (FROZEN after Block 3)
│   ├── evaluate.py                     # shared metrics + score logging (Block 4)
│   └── scratch.py                      # from-scratch NumPy algorithms (Block 12)
├── notebooks/
│   ├── P1_data_song_map.ipynb
│   ├── P2_happiness_meter.ipynb
│   ├── P3_genre_detector.ipynb
│   ├── P4_similar_songs.ipynb
│   ├── P5_hit_predictor.ipynb
│   ├── P6_neural_net_arena.ipynb
│   └── P7_mood_playlists.ipynb
├── models/                             # saved models (.joblib), one subfolder per component
├── results/
│   ├── scores.csv                      # every model's test scores (written via evaluate.log_score)
│   └── figures/                        # saved plots, one subfolder per component
└── app/
    ├── Home.py
    └── pages/
        ├── 1_Song_Map.py
        ├── 2_Happiness_Meter.py
        ├── 3_Genre_Detector.py
        ├── 4_Similar_Songs.py
        ├── 5_Hit_Predictor.py
        ├── 6_Song_Lab_and_Model_Arena.py
        └── 7_Mood_Playlists.py
```

## 4. Data rules (apply everywhere)

- **Dataset:** Spotify Tracks Dataset by maharshipandya (Hugging Face: `maharshipandya/spotify-tracks-dataset`, also on Kaggle). ~114,000 rows, 125 genres, one CSV.
- **Input features:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, explicit, mode, key (one-hot), time_signature (one-hot, or drop if almost constant – report which).
- **`popularity` is NEVER an input.** It only defines the hit label and the popularity-regression bonus task.
- **When predicting valence, remove valence from the inputs.**
- **Text columns** (track_id, track_name, artists, album_name) are never inputs; keep them for display in the app.
- **Genres:** keep about 6 clearly different genres (chosen with me in Block 2).
- **Hit label:** `hit = popularity >= 70`.
- **Split:** one stratified 80/20 train/test split by genre, `random_state=42`, saved to disk and shared by all notebooks.
- **Scaler:** StandardScaler fit on the TRAIN set only, then applied to train, test and the full song table.
- **Unsupervised parts** (clustering, recommender, song map) may use all kept songs (scaled with the train-fitted scaler), since they use no labels.

## 5. Notebook writing rules (teammates learn from these)

Every notebook must follow this layout:

1. **Title + "What this part does"**: 3–5 plain sentences: what the component does in the app and what question it answers.
2. **"Concepts in plain English"**: short explanation of each syllabus concept used in this notebook, before any code. No heavy math; one formula max per concept, explained in words.
3. **Code sections**, each preceded by a markdown cell saying what the next cell does and why.
4. **After every plot or result**: a markdown cell interpreting it in 1–3 sentences ("This shows that…").
5. **"Results summary"**: a small table of final scores and 3–5 bullet takeaways.
6. **"Questions you may be asked"**: 5 likely viva questions with short answers.

Each notebook's content must match its owner's section in the team briefs (see "Reference documents" above). At the end of each notebook block, tick the items in `docs/BRIEF_CHECKLIST.md` and fix anything not yet covered before moving on.

Other rules: load data only via `src.data.load_splits()` / `src.data.load_all_songs()`; log every final model's test scores via `src.evaluate.log_score(...)`; save final models to `models/<component>/`; save key figures to `results/figures/<component>/`; notebooks must run top to bottom without errors after a kernel restart.

## 6. The blocks

### Block 0 – Setup
- Move the proposal and team-briefs files into `docs/` (do not edit them). Read both fully.
- Write `docs/BRIEF_CHECKLIST.md`: the 7 components with owners and every "How it works" step, concept, "What you will show" item and question from each brief, as a checklist (ticked off as notebooks are built, re-checked in Block 14). Note in the build log anything in the documents that conflicts with this CLAUDE.md.
- Create the folder structure above, `requirements.txt` (pinned major versions), a `.gitignore` (ignore `data/raw/`, venv folders, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`), and a short `README.md` stub. Do NOT ignore `data/processed/`, `models/` or `results/`: they are small and the app and teammates need them. If any single file later exceeds 50 MB, warn me instead of committing it.
- Create a virtual environment and install requirements (include pytest and playwright; run `playwright install chromium` for app screenshots later).
- **GitHub setup – you handle all of it; I have NOT created a repo.**
  1. Check git is installed and that `user.name` / `user.email` are set; if not, ask me for them and set them.
  2. Check whether the GitHub CLI (`gh`) is installed. If not, install it for my OS (macOS: `brew install gh`; Windows: `winget install --id GitHub.cli`; Linux: the official apt/dnf instructions) and tell me if anything needs my approval.
  3. If `gh` is not logged in, run `gh auth login --web --git-protocol https`. It shows a one-time code and opens a browser: tell me clearly "copy this code, approve in the browser, then tell me done", and wait for me.
  4. Make the repo **private** (don't ask).
  5. Run `git init`, set the default branch to `main`, make the first commit (`Block 0: setup`), then create the repo and push in one step: `gh repo create soundsense --private --source=. --remote=origin --push`. If the name `soundsense` is taken on my account, use `soundsense-ml` and note it in the build log.
  6. Record the repo URL in the build log and show it to me at the end.
  - The `docs/` folder (proposal + team briefs) is included in the first push.
  - Only if `gh` truly cannot be installed or logged in: explain what failed, and as a last resort ask me to create an empty repo on github.com and paste its URL.
- Check: `gh repo view` shows the repo with the folder structure, CLAUDE.md, README stub and docs/.
- Check: imports of pandas, numpy, sklearn, matplotlib, seaborn, plotly, streamlit, joblib all work; print versions.

### Block 1 – Get the data
- Download the dataset CSV into `data/raw/spotify_tracks.csv` (`https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv`; if that path fails, check the dataset's file list for the CSV name, or use the `datasets`/`huggingface_hub` library. Only if every route fails, stop and ask me to download it manually, saying exactly where to put it).
- Print: row count, column names, number of unique genres, missing values per column, number of duplicate `track_id`s.
- Do not clean anything yet.

### Block 2 – Choose the 6 genres (decision with me)
- Show all genre names with counts. Propose 6 genres that sound clearly different (e.g. classical, a metal genre, hip-hop, edm, jazz, country – use the dataset's actual names).
- For your proposed 6: show counts, how many tracks appear in more than one of them, and mean energy/acousticness/danceability per genre to show they differ.
- Also show the hit rate (popularity ≥ 70) within those 6 genres.
- Choose the genres and hit threshold using the defaults in Section 2, rule 6, and record the choice and the reasons in the build log. Do not stop.

### Block 3 – Data pipeline `src/data.py` (then FROZEN)
- Functions: `build_dataset()` (clean → keep chosen genres → remove tracks that appear in more than one chosen genre → remove duplicate songs with the same track_name + artists → add `hit` → encode → split → fit scaler on train → save everything to `data/processed/`), `load_splits()` (returns scaled X_train, X_test, y dicts for genre/valence/hit/popularity, feature names), `load_all_songs()` (full kept table with metadata + scaled features), plus the fitted scaler and feature list saved with joblib/json.
- Store the chosen genres and hit threshold as constants at the top of the file.
- Add a script `python -m src.data` that builds everything and prints a summary.
- Checks (as pytest tests + logged numbers): total songs kept, songs per genre, train/test sizes, genre proportions in train vs test (should match within ~1%), hit rate in train vs test, no NaNs, train feature means ≈ 0 and stds ≈ 1, popularity not in the feature list.
- Once the tests pass, this file is FROZEN.

### Block 4 – Shared evaluation `src/evaluate.py`
- `classification_metrics(y_true, y_pred, y_proba=None)` → accuracy, macro F1, precision, recall, ROC-AUC (binary, if proba given).
- `regression_metrics(y_true, y_pred)` → RMSE, MAE, R².
- `clustering_metrics(X, labels, true_labels=None)` → silhouette, Adjusted Rand Index (if true labels given).
- `log_score(component, model_name, task, metrics_dict)` → appends one row to `results/scores.csv` (replaces an existing row with the same component+model+task, so reruns don't duplicate).
- `plot_confusion_matrix(...)` helper that also saves the figure.
- Test it with a DummyClassifier on the genre task and show the logged row.
- Checks: dummy accuracy ≈ 1/6 (~17%), scores.csv has exactly one row, rerunning does not add a duplicate.

### Block 5 – P1 notebook: The Data & Song Map
- Dataset summary (before/after cleaning counts, chosen genres, features), correlation heatmap, feature distributions per genre (energy, acousticness, danceability, valence at least).
- PCA: scree plot + cumulative explained variance; number of components for 90% variance; 2D PCA map coloured by genre; describe what PC1 and PC2 seem to represent using their loadings.
- DBSCAN on the scaled features (tune eps with a k-distance plot): count noise points, show 5 example "unusual" songs, mark them on the 2D map.
- Save: PCA model, 2D coordinates for all songs, DBSCAN outlier flags (for the app).

### Block 6 – P2 notebook: Happiness Meter
- Target valence (valence removed from inputs). Models: LinearRegression, polynomial features (degree 2) + LinearRegression, RidgeCV, LassoCV.
- Coefficient bar chart, predicted vs actual plot, list of features Lasso set to zero.
- Bonus: same models predicting popularity; compare R² and explain why it is lower.
- Log all test scores; save best valence model.

### Block 7 – P3 notebook: Genre Detector
- LogisticRegression (multinomial, C tuned by CV) vs GaussianNB.
- Accuracy, macro F1, both confusion matrices, LR coefficients for 2 genres, discussion of generative vs discriminative.
- Log scores; save both models.

### Block 8 – P4 notebook: Similar Songs
- Recommender: `NearestNeighbors` on all songs; function `similar_songs(track_index, k=5, metric)`; examples with Euclidean vs cosine; % of top-5 neighbours sharing the genre (averaged over many songs).
- kNN classifier for genre: test accuracy vs k (1–25, CV on train) plot, best k on test.
- SVM for genre: linear vs RBF, tune C (and gamma for RBF) with CV; report time taken; subsample the training set for tuning only if it takes too long (note it in the build log).
- Log scores; save recommender + best SVM + best kNN.

### Block 9 – P5 notebook: Hit Predictor
- Hit label task with class imbalance (`class_weight='balanced'` where supported).
- DecisionTreeClassifier (depth tuned by CV), RandomForestClassifier, AdaBoostClassifier (decision stumps).
- Report F1, precision, recall, ROC-AUC (not just accuracy); ROC curves of all three on one plot; plot the top 3 levels of the tree; feature importance chart from the forest; probability calibration (CalibratedClassifierCV) for the best model with a reliability plot.
- Add an honest note: sound only partly explains popularity.
- Log scores; save best calibrated model.

### Block 10 – P7 notebook: Mood Playlists
- Mood features only: energy, valence, tempo, acousticness, danceability (scaled).
- k-means: elbow + silhouette for k = 2–10; choose k (explain). Hierarchical (Ward) on a random sample of ~1,500 songs with a dendrogram. GaussianMixture with the same k: show one song's mood-membership probabilities.
- Name each mood from its cluster's average features (table). Show moods on P1's 2D PCA map. Silhouette for all three methods; Adjusted Rand Index of moods vs genres, with interpretation.
- Save: k-means and GMM models, mood names, mood label + GMM probabilities for every song.

### Block 11 – P6 notebook: Neural Network & Model Arena
- MLPClassifier for genre and hit, MLPRegressor for valence; hidden layer size and alpha tuned by CV; early stopping; loss curve plot.
- Model Arena: read `results/scores.csv`, show one leaderboard table per task (genre, hit, valence), highlight winners.
- Bootstrap (1,000 resamples of the test set) 95% confidence intervals for the top 3 models per task; say whether the winner is clearly better.
- Ablations: best genre model with vs without PCA (P1's components); with vs without the mood cluster (P7) as an extra feature.
- Log scores; save MLPs.

### Block 12 – From-scratch algorithms `src/scratch.py`
- NumPy implementations: k-nearest neighbours classifier, Gaussian Naive Bayes, linear regression (normal equation + gradient descent), k-means.
- Add a short "Built from scratch" section to the matching notebook (P4 kNN, P3 Naive Bayes, P2 linear regression, P7 k-means) comparing scratch vs scikit-learn results.
- Checks: each scratch version within ~1–2 percentage points (or near-identical R²/inertia) of scikit-learn.

### Block 13 – Streamlit app

**Goal: it should look polished and feel interactive, but stay simple.** It is a product demo, so **the app must NOT show teammate names, "P1–P7" labels, or anything that looks like a class assignment.** Every page should be understood in 10 seconds. Teammates find their part because each page is named exactly like its component in the team briefs (e.g. "Similar Songs", "Mood Playlists"); the README and demo script map pages to people. Build it in 3 steps (13a, 13b, 13c) and review the look yourself with screenshots.

**Look and feel (applies to all pages)**
- A custom theme in `.streamlit/config.toml`: dark background, one accent colour (e.g. Spotify-like green `#1DB954`), clean sans-serif font.
- A small shared helper `app/ui.py` for the repeated pieces (page header, "how it works" box, song card, metric cards, colours), so every page looks the same. A little custom CSS is fine (rounded cards, spacing); don't overdo it.
- Interactive plotly charts everywhere (hover, zoom), consistent genre colours across all pages, and gauges (plotly indicator) for hit chance and happiness.
- A "song card" component: song name, artist, genre chip, and a small radar chart of its audio features. Reuse it on every page that shows a song.
- Keep each page to 1–3 interactions (a search box, a few sliders or a dropdown). No walls of text, no clutter.
- No external images (no album art or photos); use charts, icons/emoji in headings and colour instead.

**Same layout on every feature page (clean for demos, easy to explain)**
1. **Header:** page title with an icon, e.g. "🎧 Similar Songs". No names, no P numbers.
2. **One-line purpose:** what this page answers, e.g. "Find songs that sound like the one you love."
3. **Try it:** the interactive part (search a song / move sliders / pick a mood). This is the main, biggest part of the page.
4. **How it works:** a collapsed expander (closed by default, so the page stays clean) with 3–5 numbered plain-English steps of what happens behind the result, and the ML concepts shown as small chips (e.g. `kNN` `Cosine distance` `SVM`). The presenter opens it while explaining.
5. **Behind the scenes:** a second collapsed expander with the model used, its test score (from `results/scores.csv`) and one key plot (from `results/figures/`).

**Pages**
- `Home.py`: hero title "SoundSense" + one-sentence pitch; **"Analyze a song"**: search any song → its song card plus genre, happiness gauge, hit gauge, mood and 5 similar songs, each with a small "→ see how" link to that component's page. Below it, an **"Explore SoundSense"** grid of 7 feature cards (icon, feature name, one line, link to the page), in the same order as the pages. No names or P numbers.
- `1_Song_Map.py` (P1): interactive 2D PCA scatter; toggle colour by genre / mood; outliers highlighted; hover shows song and artist; clicking or searching a song highlights it.
- `2_Happiness_Meter.py` (P2): pick a song → happiness gauge (predicted vs real); a bar chart of which features push happiness up or down.
- `3_Genre_Detector.py` (P3): pick a song → probability bar per genre from both models side by side (logistic regression vs Naive Bayes); show the confusion matrix.
- `4_Similar_Songs.py` (P4): pick a song → 5 song cards; toggle Euclidean / cosine and see the list change.
- `5_Hit_Predictor.py` (P5): pick a song → hit-chance gauge plus "why": top features from the tree/forest for this song, written as short reasons.
- `6_Song_Lab_and_Model_Arena.py` (P6): two tabs. **Song Lab:** sliders for the main features → live genre probabilities from the MLP. **Model Arena:** leaderboard per task (genre, hit, valence) with the winner highlighted; select a model → its scores and confusion matrix.
- `7_Mood_Playlists.py` (P7): pick a mood → a playlist of song cards; each song shows its GMM mood mix as a small stacked bar; mood clusters on the 2D map.

**Technical rules**
- The app only LOADS saved models and processed data (`st.cache_resource` / `st.cache_data`); no training at startup. It must open in a few seconds.
- Clear, friendly message if models are missing ("Run the notebooks first: see README").
- Song search: a searchable select box over "song – artist".

**Build order and reports**
- **13a:** theme, `ui.py`, Home page (with the "Explore SoundSense" grid) and ONE component page (Similar Songs) fully done. Then **review the look yourself**: launch the app headless, take screenshots of each page with Playwright (desktop width), look at them, and fix anything cluttered, misaligned, unreadable or ugly before building the rest. Save screenshots to `docs/screenshots/`.
- **13b:** the other six component pages using the same layout.
- **13c:** polish pass: screenshot every page again and review them as a demanding designer would; consistent colours, spacing, wording; every "How it works" box matches its owner's team brief; confirm no teammate names or P1–P7 labels appear anywhere in the app; test every page.

### Block 14 – Final verification + README
- Fresh run: rebuild data (`python -m src.data`), execute all notebooks in order P1, P2, P3, P4, P5, P7, P6 with `jupyter nbconvert --to notebook --execute --inplace`, then launch the app.
- **Brief coverage check:** go through each teammate's section in the team briefs and confirm every "How it works" step, concept, "What you will show" item and question is covered in their notebook. Fix anything missing, then update `docs/BRIEF_CHECKLIST.md` (component → item → notebook section → covered yes/no); every item must be yes.
- **Proposal check:** confirm every syllabus concept, methodology step and evaluation metric in the proposal is implemented somewhere; list where.
- Write the full README: what SoundSense is, how to set up, exact run order, how to launch the app, folder guide, which notebook belongs to which teammate (P1–P7), and a results summary table.
- Write `docs/DEMO_SCRIPT.md`: a live-demo plan in P1–P7 order (names belong here, not in the app). For each presenter: which page to open, exactly what to click or type (with a specific example song), what result to point at, and 2–3 sentences to say. Total demo about 1–2 minutes per person.
- Commit and push the final state, and tag it `v1.0`.
- **Independent review:** start a separate review subagent (one that did not build the project) and ask it to check, with fresh eyes: data leakage, the test set never used for tuning, every notebook runs and matches its owner's brief, the app screenshots look clean and professional with no teammate names, and the proposal's promises are all delivered. Fix everything it finds, then re-run all tests.
- Checks: every notebook executes OK, all pytest tests pass, final leaderboard winners, app launch command, and a fresh `git clone` of the repo into a temp folder that runs the app after only `pip install -r requirements.txt` (processed data and models come from the repo).

### Final summary (after Block 14, the only long message you show me)
Print one short summary: the repo URL; how to launch the app (`streamlit run app/Home.py`); final winners and key scores per task; any default decisions I should know about (especially the hit threshold if it changed from 70, since the proposal says 70); anything the independent review flagged; and the 3 things I should glance at myself (e.g. open the app, skim `docs/DEMO_SCRIPT.md`, check `docs/BUILD_LOG.md`).

### Block 15 (optional, only if I ask) – Deploy
- Prepare for Streamlit Community Cloud (requirements, small model files, data path handling) and give me step-by-step deployment instructions to get a public link.
