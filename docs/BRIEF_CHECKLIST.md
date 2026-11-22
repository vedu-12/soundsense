# Brief Checklist

Every item from each teammate's section of `SoundSense – Team Briefs`. Items are ticked as each notebook is built (with the notebook section that covers them) and re-checked in Block 14.

## P1 – Aryan Danny: The Data & Song Map (`P1_data_song_map.ipynb`)

**How it works**
- [x] 1. Cleaning: duplicate track_ids removed; text columns kept only for display
- [x] 2. Picking ~6 clearly different genres
- [x] 3. Encoding: key and mode one-hot encoded
- [x] 4. Scaling: standard scaling (mean 0, spread 1)
- [x] 5. Stratified 80/20 split, frozen and shared; scaler fit on train only
- [x] 6. PCA: compress features; first two components give the 2D Song Map
- [x] 7. DBSCAN: noise points = unusual songs highlighted on the map

**Concepts**
- [x] Data types: numeric vs categorical, why categories need encoding
- [x] Scaling: why distance-based models and PCA need it
- [x] Data leakage: why scaler is fit on training data only
- [x] PCA: covariance, directions of max variance, explained-variance ratio, scree plot
- [x] DBSCAN: eps, min_samples; core/border/noise points; no k needed; odd-shaped groups

**What you will show**
- [x] Dataset summary: rows before/after cleaning, chosen genres, features
- [x] Correlation heatmap + feature distributions per genre
- [x] Scree plot: components for ~90% of the information
- [x] 2D Song Map with outliers marked + example unusual songs

**Questions**
- [x] Why scale before PCA? Why fit the scaler only on the training set?
- [x] What does PC1 represent in our data?
- [x] How many components did we need, and what did we lose?
- [x] Why DBSCAN for outliers instead of k-means?
- [x] Why only 6 genres?

## P2 – Gunneet Singh: Happiness Meter (`P2_happiness_meter.ipynb`)

**How it works**
- [x] 1. Inputs = all audio features except valence; target = valence
- [x] 2. Linear regression as weighted sum; weights' meaning
- [x] 3. Least squares: normal equation or gradient descent
- [x] 4. Polynomial features: helps or overfits?
- [x] 5. Ridge (L2) and Lasso (L1); alpha picked by CV
- [x] 6. Bonus: same models predict popularity; weaker results explained

**Concepts**
- [x] Linear regression and what each weight means
- [x] Least squares = maximum likelihood under Gaussian errors
- [x] Overfitting and bias–variance (polynomial features)
- [x] Regularization: Ridge vs Lasso, alpha, why Lasso gives zeros
- [x] Metrics: RMSE, MAE, R²

**What you will show**
- [x] Bar chart of weights (happier vs sadder features)
- [x] Predicted vs actual valence scatter plot
- [x] RMSE and R² for plain, polynomial, Ridge, Lasso + features Lasso dropped
- [x] Popularity result and why it is low

**Questions**
- [x] What does R² = 0.4 mean in plain words?
- [x] L1 vs L2, why Lasso sets weights to zero
- [x] Assumptions of linear regression
- [x] Why popularity is harder than valence
- [x] How alpha was picked

## P3 – Shreyash Lakshmi Narayan: Genre Detector (`P3_genre_detector.ipynb`)

**How it works**
- [ ] 1. Logistic regression and Gaussian NB on the same shared split
- [ ] 2. Logistic regression: weighted sums + softmax, cross-entropy, C tuned by CV
- [ ] 3. Naive Bayes: per-genre bell curves, Bayes' rule, independence assumption
- [ ] 4. Compare accuracy, macro F1, confusion matrices; which genres are mixed up

**Concepts**
- [ ] Logistic regression: sigmoid/softmax, linear boundaries, cross-entropy, regularization
- [ ] Naive Bayes: Bayes' rule, independence, Gaussian likelihoods
- [ ] Generative vs discriminative
- [ ] Metrics: accuracy, precision, recall, F1, macro F1
- [ ] Confusion matrix: rows = true, columns = predicted

**What you will show**
- [ ] Accuracy and macro F1 for both models
- [ ] Both confusion matrices with musical explanation of confusions
- [ ] LR weights for one or two genres
- [ ] Verdict: which model won and how it fits the theory

**Questions**
- [ ] Why "naive"? Is independence true (energy vs loudness)?
- [ ] When can NB beat LR?
- [ ] What does softmax do?
- [ ] Why macro F1 instead of only accuracy?
- [ ] What does one off-diagonal cell mean?

## P4 – Amithav Sachin: Similar Songs (`P4_similar_songs.ipynb`)

**How it works**
- [ ] 1. kNN recommender: 5 closest songs with their genres
- [ ] 2. Euclidean vs cosine; how recommendations change
- [ ] 3. kNN classifier: k = 1, 3, …, 25; accuracy vs k plot
- [ ] 4. SVM: linear vs RBF, tune C and gamma with CV; multiclass via several two-genre SVMs

**Concepts**
- [ ] kNN: lazy learner; small k overfits, large k underfits
- [ ] Distance metrics: Euclidean vs cosine; why scaling is essential
- [ ] Curse of dimensionality
- [ ] SVM: hyperplane, margin, support vectors, soft margin and C
- [ ] Kernel trick: RBF, gamma

**What you will show**
- [ ] 2–3 recommender examples, Euclidean vs cosine side by side
- [ ] Accuracy vs k plot and best k
- [ ] Linear vs RBF SVM accuracy, effect of C
- [ ] Share of 5 neighbours sharing the genre

**Questions**
- [ ] Why is kNN lazy? What is slow?
- [ ] How does k affect bias and variance?
- [ ] What is a support vector? Removing a non-support vector?
- [ ] Large vs small C
- [ ] Kernel trick in one sentence

## P5 – Vedang Chinhalli: Hit Predictor (`P5_hit_predictor.ipynb`)

**How it works**
- [ ] 1. Label hit = popularity ≥ threshold; imbalanced classes
- [ ] 2. Decision tree: yes/no splits, Gini/entropy, depth tuned
- [ ] 3. Random Forest (bagging): random samples + feature subsets, voting
- [ ] 4. AdaBoost (boosting): stumps focusing on previous mistakes
- [ ] 5. Imbalance: class weights; judge by F1 and ROC-AUC
- [ ] 6. Explaining hits: forest feature importance + top tree levels as rules

**Concepts**
- [ ] Decision trees: Gini vs entropy, information gain, depth/overfitting, pruning
- [ ] Ensembles: bagging vs boosting; forest stability; AdaBoost reweighting
- [ ] Class imbalance: accuracy paradox; precision vs recall
- [ ] ROC-AUC
- [ ] Feature importance

**What you will show**
- [ ] Drawn decision tree (top levels) as "rules for a hit"
- [ ] Feature-importance chart
- [ ] F1 and ROC-AUC for tree vs RF vs AdaBoost + ROC curve
- [ ] Honest limit: sound explains only part of a hit

**Questions**
- [ ] Gini vs entropy
- [ ] Why RF beats a tree? Bagging vs boosting?
- [ ] Why not accuracy?
- [ ] ROC-AUC = 0.5 vs 1.0
- [ ] Does importance mean causation?

## P6 – Krish Jai Ganesh: Neural Network & Model Arena (`P6_neural_net_arena.ipynb`)

**How it works**
- [ ] 1. MLP: inputs, hidden ReLU layers, task-specific output (softmax / sigmoid / linear)
- [ ] 2. Training: forward pass, loss, backpropagation, Adam, epochs
- [ ] 3. Avoid overfitting: train vs validation loss, early stopping, weight penalty, CV tuning of neurons and learning rate
- [ ] 4. Model Arena: every model's scores in one table per task
- [ ] 5. Bootstrap ranges; with vs without PCA

**Concepts**
- [ ] Neurons and layers; without activations it's linear
- [ ] Activations: ReLU, sigmoid, softmax
- [ ] Backpropagation and gradient descent; learning rate, epochs
- [ ] Overfitting in networks: curves, early stopping, regularization
- [ ] Fair comparison: same split/metrics, CV, bootstrap intervals

**What you will show**
- [ ] Network diagram (inputs → hidden → output)
- [ ] Training vs validation loss curve
- [ ] Model Arena tables (genre, hit, valence), winners and why
- [ ] Song Lab demo: raise energy and loudness → guess moves towards metal

**Questions**
- [ ] Why activation functions?
- [ ] Backpropagation in two sentences
- [ ] How hidden neurons were chosen
- [ ] Why NNs don't always win on small tabular data
- [ ] How do you know the winner isn't lucky?

## P7 – Palakollu Sai Nikhil Reddy: Mood Playlists (`P7_mood_playlists.ipynb`)

**How it works**
- [ ] 1. Mood features: energy, valence, tempo, acousticness, danceability (scaled); no genre
- [ ] 2. k-means loop; k via elbow and silhouette
- [ ] 3. Hierarchical clustering on ~1–2k sample; dendrogram; cutting the tree
- [ ] 4. GMM via EM: E-step, M-step; soft mood mix
- [ ] 5. Naming moods from cluster averages
- [ ] 6. Checks: silhouette of the three methods; ARI vs genres

**Concepts**
- [ ] Unsupervised vs supervised
- [ ] k-means: loop, choosing k, weaknesses
- [ ] Hierarchical: agglomerative, linkage (Ward/average/complete), dendrogram
- [ ] GMM and EM: hard vs soft, E/M steps, k-means as special case, local optima
- [ ] Evaluation without labels: elbow, silhouette, ARI

**What you will show**
- [ ] Elbow and silhouette plots, chosen number of moods
- [ ] Mood clusters on P1's 2D Song Map
- [ ] Table of mood averages + names + example songs per mood
- [ ] Dendrogram, one song's GMM mood mix, moods vs genres

**Questions**
- [ ] How was the number of moods chosen?
- [ ] Hard vs soft clustering
- [ ] E-step and M-step in simple words
- [ ] Why two k-means runs can differ
- [ ] Do moods match genres?
