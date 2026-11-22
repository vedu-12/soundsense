# SoundSense – Team Briefs

Nov 22, 2026 · @Vedang

Each member owns one component of SoundSense: they understand it fully and present it. Each message below stands on its own and can be sent as is, with the proposal PDF as background.

## The Split

Every component pairs one app feature with 2–3 related syllabus topics. The harder topics are spread out: SVM sits with kNN, the neural network with the model comparison, and EM with k-means.

| Role and owner | Component | App feature | Syllabus topics | Presents |
| --- | --- | --- | --- | --- |
| P1 – Aryan Danny | The Data & Song Map | 2D map of all songs coloured by genre, unusual songs highlighted | Data types, preprocessing, scaling, train/test split, PCA, DBSCAN | The dataset, how it was prepared, what the map shows |
| P2 – Gunneet Singh | Happiness Meter | Predicted happiness (valence, 0–1) for any song | Linear regression, least squares, Ridge/Lasso, bias–variance | How well sound predicts happiness; what regularization changed |
| P3 – Shreyash Lakshmi Narayan | Genre Detector | The genre of any song | Logistic regression vs Naive Bayes (discriminative vs generative), confusion matrix, F1 | Which genres get confused and why the two models differ |
| P4 – Amithav Sachin | Similar Songs | Pick a song, get the 5 most similar | kNN, distance metrics, SVM (margins, kernels) | How "similar" is measured; how SVM separates genres |
| P5 – Vedang Chinhalli | Hit Predictor | Hit chance for a song, with the reasons | Decision tree, Random Forest, AdaBoost, class imbalance, ROC-AUC | What makes a hit; why ensembles beat one tree |
| P6 – Krish Jai Ganesh | Neural Network & Model Arena | Song Lab sliders; leaderboard of every model | Neural networks (MLP), backpropagation, cross-validation, model comparison | How the network works; which model won and why |
| P7 – Palakollu Sai Nikhil Reddy | Mood Playlists | Pick a mood, get a playlist with each song's mood mix | k-means, hierarchical clustering, GMM via EM, silhouette | How moods were found with no labels; whether they match genres |

The code is built in one place, with one notebook per component (`P1_data_song_map.ipynb` … `P7_mood_playlists.ipynb`), so each person walks through their own notebook.

## P1 – Aryan Danny: The Data & Song Map

Hey Aryan! For our ML project (SoundSense) you own **P1: The Data & Song Map**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P1_data_song_map.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You are the foundation. Every other part uses the data you prepare. In the app, your feature is the **Song Map**: every song plotted as a dot in 2D, coloured by genre, with "unusual" songs highlighted.

**How it works in our project**

1. **Cleaning:** the same song appears under several genres, so duplicate `track_id`s are removed. Text columns (song name, artist, album) are dropped from the inputs and kept only for display.
2. **Picking genres:** 125 genres is too many and too similar. We keep \~6 that sound clearly different.
3. **Encoding:** `key` (0–11) and `mode` (major/minor) are categories, not quantities, so they are one-hot encoded.
4. **Scaling:** features are on very different ranges (tempo \~60–200, danceability 0–1). Standard scaling gives each feature mean 0 and spread 1, so no feature dominates just because its numbers are bigger.
5. **Train/test split:** one stratified 80/20 split (same genre mix in both), frozen and shared by all parts. The scaler is fit on the training set only, so no test information leaks into training.
6. **PCA:** compresses the \~15 features into a few "principal components", the directions where songs differ most. The first two give the 2D Song Map.
7. **DBSCAN:** marks songs that sit far from any dense group as "noise". Those are the unusual songs highlighted on the map.

**Concepts you must understand**

- **Data types:** numeric vs categorical, and why categories need encoding.
- **Scaling:** why distance-based models (kNN, SVM, k-means, neural networks) and PCA need it.
- **Data leakage:** why we fit the scaler on training data only.
- **PCA:** covariance, principal components as directions of maximum variance, explained-variance ratio, and the scree plot (how much each component explains).
- **DBSCAN:** `eps` (neighbourhood radius) and `min_samples`; core, border and noise points; why it needs no number of clusters and can find odd-shaped groups.

**What you will show**

- Dataset summary: rows before and after cleaning, the chosen genres, the features.
- Correlation heatmap and a few feature distributions per genre (e.g. energy for metal vs classical).
- Scree plot: how many components keep \~90% of the information.
- The 2D Song Map, with outliers marked, and a few example unusual songs.

**Questions you should be ready for**

- Why scale before PCA? Why fit the scaler only on the training set?
- What does the first principal component represent in our data?
- How many components did we need, and what did we lose?
- Why DBSCAN for outliers instead of k-means?
- Why only 6 genres?

**How it connects:** everyone uses your cleaned, scaled split. P6 compares models with and without PCA, and P7's mood clusters are drawn on your map.

## P2 – Gunneet Singh: Happiness Meter

Hey Gunneet! For our ML project (SoundSense) you own **P2: Happiness Meter**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P2_happiness_meter.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You own the project's **regression** part. Spotify's `valence` column is a 0–1 score for how positive a song sounds (1 = happy, cheerful; 0 = sad, angry). Your model predicts valence from the other audio features, which is useful for auto-tagging new songs. In the app, this is the **Happiness Meter**: pick a song and see its predicted happiness next to the real value.

**How it works in our project**

1. **Inputs and target:** all audio features except valence go in; valence is what we predict.
2. **Linear regression:** the prediction is a weighted sum of the features plus a constant: happiness = w₁·danceability + w₂·energy + … + b. Each weight says how much that feature pushes happiness up or down.
3. **Least squares:** the weights are chosen to make the total squared error between predicted and real valence as small as possible. This can be solved directly (the normal equation) or step by step (gradient descent).
4. **Polynomial features:** we also try squares and pairwise products of features, so the model can bend a little, and check whether that helps or overfits.
5. **Ridge and Lasso:** add a penalty on large weights. Ridge (L2) shrinks all weights; Lasso (L1) can push some exactly to zero, effectively dropping useless features. The penalty strength (alpha) is picked by cross-validation.
6. **Bonus check:** the same models try to predict `popularity`. Expect much weaker results, since popularity depends on marketing and artist fame, not only sound. That is a good discussion point.

**Concepts you must understand**

- **Linear regression** and what each weight means.
- **Least squares** (parameter estimation): why we minimise squared error, and that it equals maximum likelihood when errors are Gaussian.
- **Overfitting and bias–variance:** why polynomial features can fit training data better but test data worse.
- **Regularization:** Ridge vs Lasso, what alpha does, why Lasso gives zeros.
- **Metrics:** RMSE (typical error size), MAE, and R² (share of variation explained; 1 is perfect, 0 is no better than guessing the average).

**What you will show**

- A bar chart of the weights: which features make songs sound happier or sadder.
- Predicted vs actual valence scatter plot.
- RMSE and R² for plain, polynomial, Ridge and Lasso models, and which features Lasso dropped.
- The popularity result and why it is low.

**Questions you should be ready for**

- What does R² = 0.4 mean in plain words?
- Difference between L1 and L2, and why Lasso sets weights to zero.
- What assumptions does linear regression make?
- Why is popularity so much harder to predict than valence?
- How did you pick alpha?

**How it connects:** P1 provides the scaled data. Your valence model feeds the app's Happiness Meter, and P6 compares it with the neural network's valence prediction in the Model Arena.

## P3 – Shreyash Lakshmi Narayan: Genre Detector

Hey Shreyash! For our ML project (SoundSense) you own **P3: Genre Detector**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P3_genre_detector.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You own the main **classification** task: given a song's audio features, say which of our \~6 genres it belongs to. This is the core use case, auto-tagging new uploads that have no genre. In the app, this is the **Genre Detector**: pick a song and see the predicted genre with a probability for each genre.

**How it works in our project**

1. **Two classic models on the same data:** logistic regression and Gaussian Naive Bayes, both trained on P1's training set and tested on the shared test set.
2. **Logistic regression (discriminative):** learns a weighted sum of features for each genre and turns the scores into probabilities with softmax. It directly learns the boundaries between genres. Weights are found by maximizing the likelihood of the correct genres (minimizing cross-entropy loss), with a regularization setting `C` tuned by cross-validation.
3. **Naive Bayes (generative):** learns what a "typical" song of each genre looks like: for each genre and each feature, a bell curve (mean and spread). For a new song it uses Bayes' rule to ask which genre most likely produced these features. "Naive" because it assumes features are independent within a genre.
4. **Compare:** accuracy, macro F1 and a confusion matrix for each model, then look at which genres get mixed up.

**Concepts you must understand**

- **Logistic regression:** sigmoid/softmax, linear decision boundaries, cross-entropy loss, regularization.
- **Naive Bayes:** Bayes' rule (posterior ∝ likelihood × prior), the independence assumption, Gaussian likelihoods.
- **Generative vs discriminative:** NB models how each genre generates data, P(features | genre); logistic regression models P(genre | features) directly. This is a key syllabus comparison.
- **Metrics:** accuracy, precision, recall, F1, and why we use macro F1 (treats every genre equally).
- **Confusion matrix:** rows = true genre, columns = predicted genre.

**What you will show**

- Accuracy and macro F1 for both models.
- Both confusion matrices, pointing out which genres are confused and why in musical terms (e.g. EDM vs hip-hop share high danceability).
- Logistic regression weights for one or two genres: which features say "classical" or "metal".
- Your verdict: which model won, and how that fits generative vs discriminative theory.

**Questions you should be ready for**

- Why is Naive Bayes called "naive"? Is the assumption true for our features (e.g. energy and loudness)?
- When can Naive Bayes beat logistic regression?
- What does softmax do?
- Why macro F1 instead of only accuracy?
- What does one off-diagonal cell in the confusion matrix mean?

**How it connects:** P1 provides the data. P4 and P6 run stronger models on the same genre task, and P6's Model Arena compares all of them against yours as the baseline.

## P4 – Amithav Sachin: Similar Songs

Hey Amithav! For our ML project (SoundSense) you own **P4: Similar Songs**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P4_similar_songs.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You own the **"geometry" models**: methods that treat every song as a point in feature space (one axis per feature) and reason about distances and boundaries. In the app, your feature is **Similar Songs**: pick any song and get the 5 songs that sound most like it, the classic "you might also like" recommender.

**How it works in our project**

1. **kNN recommender:** for a chosen song, find the 5 closest songs in the scaled feature space and list them with their genres.
2. **Distance metrics:** compare Euclidean distance (straight-line gap) with cosine similarity (angle between feature vectors: same "shape" of song regardless of overall intensity). Show how the recommendations change.
3. **kNN classifier:** predict a song's genre by majority vote of its k nearest training songs. Try k = 1, 3, 5, …, 25 and plot test accuracy vs k.
4. **SVM for genre:** finds the boundary that separates genres with the widest possible margin. Compare a linear kernel with an RBF kernel (curved boundaries) and tune `C` and `gamma` with cross-validation. Multiclass is handled by combining several two-genre SVMs.

**Concepts you must understand**

- **kNN:** a "lazy" learner (no training, all work at prediction time); small k overfits (noisy), large k underfits (too smooth).
- **Distance metrics:** Euclidean vs cosine, and why scaling is essential (unscaled tempo, around 120, would swamp danceability, between 0 and 1).
- **Curse of dimensionality:** distances become less meaningful as features grow.
- **SVM:** hyperplane, margin, support vectors (the few songs that define the boundary), soft margin and `C` (tolerance for mistakes).
- **Kernel trick:** RBF lets SVM draw curved boundaries without building new features by hand; `gamma` controls how wiggly they get.

**What you will show**

- Two or three live recommender examples ("for this metal song, we suggest…"), with Euclidean vs cosine side by side.
- Accuracy vs k plot, and the best k.
- Linear vs RBF SVM accuracy for genre, and the effect of `C`.
- What share of a song's 5 neighbours share its genre (a simple quality check for the recommender).

**Questions you should be ready for**

- Why is kNN called a lazy learner? What is slow about it?
- How does k affect bias and variance?
- What is a support vector? What happens if you remove a non-support-vector song?
- What does a large vs small `C` do?
- Explain the kernel trick in one sentence.

**How it connects:** you rely on P1's scaling (distances break without it). Your recommender is the app's Similar Songs page, and your kNN and SVM scores go into P6's Model Arena alongside P3 and P5.

## P5 – Vedang Chinhalli: Hit Predictor

Vedang, for our ML project (SoundSense) you own **P5: Hit Predictor**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P5_hit_predictor.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You answer the most interesting question in the project: **can we tell from sound alone whether a song will be a hit, and what makes a hit?** A song counts as a hit if its Spotify popularity is 70 or more. In the app, the **Hit Predictor** shows a hit chance for any song, plus the reasons behind it (e.g. "high danceability, low acousticness").

**How it works in our project**

1. **The label:** hit = popularity ≥ 70, otherwise not a hit. Hits are a small share of songs, so the classes are **imbalanced**.
2. **Decision tree:** asks a series of yes/no questions ("danceability > 0.7?", "acousticness < 0.2?") until it reaches a hit or not-hit leaf. Each split is chosen to make the groups as pure as possible (Gini impurity or entropy). Tree depth is tuned so it does not just memorise the training songs.
3. **Random Forest (bagging):** trains many trees, each on a random sample of songs and a random subset of features, then lets them vote. Averaging many different trees reduces variance.
4. **AdaBoost (boosting):** trains very small trees ("stumps") one after another; each new stump focuses more on the songs the previous ones got wrong. This reduces bias.
5. **Handling imbalance:** class weights make missing a hit cost more, and we judge models by F1 and ROC-AUC rather than plain accuracy.
6. **Explaining hits:** feature importance from the forest, and the top few levels of the tree drawn as readable rules.

**Concepts you must understand**

- **Decision trees:** Gini vs entropy, information gain, depth and overfitting, pruning.
- **Ensembles:** bagging vs boosting; why a forest is more stable than one tree; how AdaBoost reweights songs.
- **Class imbalance:** why 90% accuracy can mean nothing if 90% of songs are not hits; precision vs recall for hits.
- **ROC-AUC:** how well the model ranks hits above non-hits across all thresholds.
- **Feature importance:** which features the trees rely on most.

**What you will show**

- The drawn decision tree (top levels) as "rules for a hit".
- Feature-importance chart: what makes a hit.
- F1 and ROC-AUC for tree vs Random Forest vs AdaBoost, and a ROC curve.
- An honest limit: sound explains only part of a hit (artist fame and marketing matter), so expect moderate scores.

**Questions you should be ready for**

- Gini vs entropy: what is each measuring?
- Why does a random forest beat a single tree? Bagging vs boosting?
- Why not just use accuracy here?
- What does ROC-AUC = 0.5 mean? And 1.0?
- If danceability is most important, does that mean it causes hits?

**How it connects:** P1 creates the hit label and split. P3's models could also predict hits, and your ensembles are compared with everyone else's in P6's Model Arena; the best one powers the app's hit gauge.

## P6 – Krish Jai Ganesh: Neural Network & Model Arena

Hey Krish! For our ML project (SoundSense) you own **P6: Neural Network & Model Arena**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P6_neural_net_arena.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You own two things. First, the **neural network**, one flexible model that can do genre, happiness and hit prediction. Second, the **Model Arena**, the final scoreboard that compares every model in the project fairly and says which won and why. In the app you have two pages: **Song Lab** (move sliders for energy, tempo, etc. and watch the network's genre guess change live) and **Model Arena** (leaderboard of all models; click one to see its scores and confusion matrix).

**How it works in our project**

1. **The network (MLP):** inputs = the audio features; one or two hidden layers of neurons with ReLU activation; an output layer that depends on the task: softmax over genres, sigmoid for hit, a single linear output for valence.
2. **Training:** a forward pass makes a prediction; a loss measures how wrong it is (cross-entropy for classification, squared error for valence); **backpropagation** computes how each weight affected the loss; an optimizer (Adam) nudges the weights. This repeats over many epochs.
3. **Avoiding overfitting:** watch training vs validation loss, use early stopping and a small weight penalty, and tune the number of hidden neurons and learning rate with cross-validation.
4. **Model Arena:** collect every model's scores from the shared evaluation script (same test set, same metrics): P2's regressions, P3's logistic regression and Naive Bayes, P4's kNN and SVM, P5's tree and ensembles, and your network. Put them in one table per task.
5. **Is the gap real?:** bootstrap the test set (resample it many times) to get a range for each score, so we only call a winner when the difference is clear. Also compare scores with and without PCA (using P1's components).

**Concepts you must understand**

- **Neurons and layers:** weighted sum + activation; why without activations a deep network is just linear regression.
- **Activations:** ReLU, sigmoid, softmax and where each is used.
- **Backpropagation and gradient descent:** the chain rule passes the error backwards; learning rate and epochs.
- **Overfitting in networks:** training vs validation curves, early stopping, regularization.
- **Fair comparison:** same split, same metrics, cross-validation, and why a single score can be luck (bootstrap intervals).

**What you will show**

- A simple diagram of the network (inputs → hidden layer → output).
- Training vs validation loss curve.
- The Model Arena tables (genre, hit, valence), the winner of each, and why: e.g. nonlinear models beating linear ones where genres overlap.
- A quick Song Lab demo: raise energy and loudness and watch the guess move towards metal.

**Questions you should be ready for**

- Why do we need activation functions?
- Explain backpropagation in two sentences.
- How did you choose the number of hidden neurons?
- Why don't neural networks always win on small table-style data like ours?
- How do you know the winning model isn't just lucky?

**How it connects:** you depend on everyone. Every other part's results land in your Model Arena, and the winners power the app's main predictions.

## P7 – Palakollu Sai Nikhil Reddy: Mood Playlists

Hey Nikhil! For our ML project (SoundSense) you own **P7: Mood Playlists**. One of us builds all the code; your job is to understand your part fully and present it. Background is in the proposal PDF.

**The project in short (everyone knows this)**

SoundSense uses only a song's audio features to detect its genre, predict how happy it sounds, predict whether it will be a hit, group songs into mood playlists, and recommend similar songs. It all comes together in a Streamlit app.

- **Data:** the public Spotify Tracks Dataset (114,000 tracks, 125 genres). We remove duplicates and keep about 6 clearly different genres (e.g. classical, metal, hip-hop, EDM, jazz, country), around 6,000 songs.
- **Inputs:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, duration, explicit.
- **Targets:** genre (classification), valence/happiness 0–1 (regression, with valence removed from the inputs), hit = popularity ≥ 70 (yes/no). Mood has no label; it is found by clustering.
- **Pipeline:** clean → scale → one fixed train/test split used by everyone → models → same metrics → comparison → app.
- **Metrics:** accuracy and F1 for classification, RMSE and R² for regression, silhouette score for clustering.
- **The 7 parts:** P1 Data & Song Map (Aryan) · P2 Happiness Meter (Gunneet) · P3 Genre Detector (Shreyash) · P4 Similar Songs (Amithav) · P5 Hit Predictor (Vedang) · P6 Neural Network & Model Arena (Krish) · P7 Mood Playlists (Nikhil).
- **Your job:** understand your notebook (`P7_mood_playlists.ipynb`), present it in about 2–3 slides, and answer basic questions on the whole project.

**Your part: what it does**

You own the **unsupervised** part. Nobody has labelled songs with moods, so the model has to discover groups of similar-feeling songs by itself, then we give each group a name. In the app, **Mood Playlists** lets you pick a mood (e.g. Party, Chill, Sad, Intense) and get a playlist, and each song shows its mood mix (e.g. 70% Chill, 30% Sad).

**How it works in our project**

1. **Mood features:** only the features that capture feel: energy, valence, tempo, acousticness, danceability (scaled). Genre is not used.
2. **k-means:** picks k centre points, assigns every song to its nearest centre, moves each centre to the average of its songs, and repeats until stable. k is chosen with the elbow method (where adding clusters stops helping much) and the silhouette score (how well each song fits its own group vs the next).
3. **Hierarchical clustering:** starts with every song alone and keeps merging the two closest groups, producing a **dendrogram** (a tree of merges). Cutting the tree at a height gives the clusters. Run on a sample of \~1–2k songs so the tree stays readable.
4. **GMM via EM:** treats each mood as a bell-shaped cloud (a Gaussian). The **E-step** works out, for every song, the probability it belongs to each mood; the **M-step** updates each mood's centre and shape from those probabilities; repeat. This gives the soft mood mix shown in the app.
5. **Naming moods:** look at each cluster's average features, e.g. high energy + high valence = "Party", low energy + low valence = "Sad", high acousticness + low energy = "Chill".
6. **Checks:** compare the three methods' silhouette scores, and compare moods with real genres using the Adjusted Rand Index (do moods just copy genres, or cut across them?).

**Concepts you must understand**

- **Unsupervised vs supervised learning.**
- **k-means:** the assign/update loop, choosing k, its weaknesses (assumes round clusters, sensitive to the start).
- **Hierarchical clustering:** agglomerative merging, linkage (Ward, average, complete), reading a dendrogram.
- **GMM and EM:** hard vs soft clustering, E-step and M-step, why k-means is a special case of GMM, local optima.
- **Evaluation without labels:** elbow, silhouette, Adjusted Rand Index.

**What you will show**

- Elbow and silhouette plots, and the chosen number of moods.
- The mood clusters drawn on P1's 2D Song Map.
- A table of each mood's average features and its name, with a few example songs per mood.
- The dendrogram, one song's GMM mood mix, and the moods vs genres comparison.

**Questions you should be ready for**

- How did you choose the number of moods?
- What is the difference between hard and soft clustering?
- Explain the E-step and M-step in simple words.
- Why might two runs of k-means give different results?
- Do our moods match genres? Why or why not?

**How it connects:** P1 provides the scaled data and the 2D map you draw on. P6 tests whether adding your mood cluster as an extra feature helps the genre and hit models.
