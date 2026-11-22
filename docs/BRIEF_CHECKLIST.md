# Brief Checklist

Every item from each teammate's section of `SoundSense – Team Briefs`, mapped to the notebook section that covers it. Re-checked in Block 14 against the executed notebooks. "Concepts" = the notebook's *Concepts in plain English* section; "Q1–Q5" = *Questions you may be asked*.

## P1 – Aryan Danny: The Data & Song Map (`P1_data_song_map.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. Cleaning: duplicate track_ids removed; text columns display-only | §1 Raw dataset, §3 Cleaning, §4 Features | yes |
| How it works | 2. Picking ~6 clearly different genres | §2 Picking 6 genres | yes |
| How it works | 3. Encoding: key and mode one-hot | §4 Encoding categories | yes |
| How it works | 4. Standard scaling (mean 0, spread 1) | §6 Scaling | yes |
| How it works | 5. Stratified 80/20 split, scaler fit on train only | §5 Train/test split, §6 Scaling | yes |
| How it works | 6. PCA → first two components give the Song Map | §9 PCA, The 2D Song Map | yes |
| How it works | 7. DBSCAN noise = unusual songs on the map | §10 DBSCAN | yes |
| Concept | Data types: numeric vs categorical, why encode | Concepts, §4 | yes |
| Concept | Scaling for distance-based models and PCA | Concepts, §6 | yes |
| Concept | Data leakage: scaler on train only | Concepts, §6 | yes |
| Concept | PCA: covariance, max-variance directions, explained variance, scree plot | Concepts, §9 | yes |
| Concept | DBSCAN: eps, min_samples, core/border/noise, no k, odd shapes | Concepts, §10 | yes |
| Show | Dataset summary: rows before/after, genres, features | §1, §3, §4, Results summary | yes |
| Show | Correlation heatmap + distributions per genre | §7, §8 | yes |
| Show | Scree plot: components for ~90% | §9 (7 of 10 components) | yes |
| Show | 2D Song Map with outliers + example unusual songs | §10 (map + 5 examples) | yes |
| Question | Why scale before PCA? Why fit scaler on train only? | Q1 | yes |
| Question | What does PC1 represent? | §9 loadings, Q2 | yes |
| Question | How many components, what did we lose? | Q3 | yes |
| Question | Why DBSCAN instead of k-means? | Q4 | yes |
| Question | Why only 6 genres? | Q5 | yes |

## P2 – Gunneet Singh: Happiness Meter (`P2_happiness_meter.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. Inputs = all features except valence | Setup | yes |
| How it works | 2. Linear regression as weighted sum | Concepts, §1, §2 | yes |
| How it works | 3. Least squares: normal equation / gradient descent | Concepts, §7 Built from scratch | yes |
| How it works | 4. Polynomial features: help or overfit? | §1 (train vs test R² plot) | yes |
| How it works | 5. Ridge / Lasso, alpha by CV | §1, §3 | yes |
| How it works | 6. Bonus: popularity, weaker results | §5 | yes |
| Concept | Linear regression and weights | Concepts, §2 | yes |
| Concept | Least squares = maximum likelihood (Gaussian errors) | Concepts | yes |
| Concept | Overfitting and bias–variance | Concepts, §1 | yes |
| Concept | Ridge vs Lasso, alpha, zeros | Concepts, §3, Q2 | yes |
| Concept | RMSE, MAE, R² | Concepts, §1 | yes |
| Show | Bar chart of weights | §2 | yes |
| Show | Predicted vs actual scatter | §4 | yes |
| Show | RMSE and R² for all 4 models + Lasso's dropped features | §1, §3, Results summary | yes |
| Show | Popularity result and why it is low | §5, Q4 | yes |
| Question | What does R² = 0.4 mean? | Q1 | yes |
| Question | L1 vs L2, why Lasso zeros | Q2 | yes |
| Question | Assumptions of linear regression | Q3 | yes |
| Question | Why popularity is harder | Q4 | yes |
| Question | How alpha was picked | Q5 | yes |

## P3 – Shreyash Lakshmi Narayan: Genre Detector (`P3_genre_detector.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. LR and Gaussian NB on the shared split | Setup, §1, §2 | yes |
| How it works | 2. LR: softmax, cross-entropy, C by CV | Concepts, §1 | yes |
| How it works | 3. NB: per-genre bell curves, Bayes' rule, independence | Concepts, §2 | yes |
| How it works | 4. Accuracy, macro F1, confusion matrices, mix-ups | §3, §4 | yes |
| Concept | Sigmoid/softmax, linear boundaries, cross-entropy, regularization | Concepts | yes |
| Concept | Bayes' rule, independence, Gaussian likelihoods | Concepts, §2 | yes |
| Concept | Generative vs discriminative | Concepts, §7, Verdict | yes |
| Concept | Accuracy, precision, recall, F1, macro F1 | Concepts, §3 | yes |
| Concept | Confusion matrix rows/columns | Concepts, §4 | yes |
| Show | Accuracy and macro F1 for both | §3 | yes |
| Show | Both confusion matrices, musical explanation | §4 | yes |
| Show | LR weights for two genres | §5 (classical, heavy-metal) | yes |
| Show | Verdict vs theory | Verdict | yes |
| Question | Why "naive"? True for energy/loudness? | §6, Q1 | yes |
| Question | When can NB beat LR? | §7, Q2 | yes |
| Question | What does softmax do? | Q3 | yes |
| Question | Why macro F1? | Q4 | yes |
| Question | Off-diagonal cell meaning | Q5 | yes |

## P4 – Amithav Sachin: Similar Songs (`P4_similar_songs.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. kNN recommender: 5 closest songs with genres | §1 | yes |
| How it works | 2. Euclidean vs cosine, how lists change | §1 examples, §2 | yes |
| How it works | 3. kNN classifier, k = 1…25, accuracy vs k | §4 (cross-validated accuracy) | yes |
| How it works | 4. SVM linear vs RBF, C and gamma by CV, one-vs-one | Concepts, §5 | yes |
| Concept | Lazy learner; small k overfits, large k underfits | Concepts, §4, Q1–Q2 | yes |
| Concept | Euclidean vs cosine; scaling essential | Concepts, §3 | yes |
| Concept | Curse of dimensionality | Concepts, §2, §4 | yes |
| Concept | Hyperplane, margin, support vectors, soft margin, C | Concepts, §5 Support vectors | yes |
| Concept | Kernel trick, RBF, gamma | Concepts, §5 Effect of C | yes |
| Show | 2–3 recommender examples, both metrics side by side | §1 (metal, classical, hip-hop) | yes |
| Show | Accuracy vs k plot, best k | §4 (k = 13) | yes |
| Show | Linear vs RBF accuracy, effect of C | §5 | yes |
| Show | Share of 5 neighbours with the same genre | §2 | yes |
| Question | Why lazy? What is slow? | Q1 | yes |
| Question | k and bias–variance | Q2 | yes |
| Question | Support vectors; removing a non-support vector | Q3 | yes |
| Question | Large vs small C | Q4 | yes |
| Question | Kernel trick in one sentence | Q5 | yes |

## P5 – Vedang Chinhalli: Hit Predictor (`P5_hit_predictor.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. Hit = popularity ≥ 70; imbalanced | Setup, §1 | yes |
| How it works | 2. Decision tree, Gini/entropy, depth tuned | Concepts, §2 | yes |
| How it works | 3. Random Forest (bagging) | Concepts, §3 | yes |
| How it works | 4. AdaBoost with stumps (boosting) | Concepts, §4 | yes |
| How it works | 5. Class weights; F1 and ROC-AUC | §2–§5 | yes |
| How it works | 6. Feature importance + top tree levels as rules | §6, §7 | yes |
| Concept | Gini vs entropy, information gain, depth, pruning | Concepts, §2 | yes |
| Concept | Bagging vs boosting, forest stability, AdaBoost reweighting | Concepts, Q2 | yes |
| Concept | Accuracy paradox; precision vs recall | Concepts, §1 | yes |
| Concept | ROC-AUC | Concepts, §5, Q4 | yes |
| Concept | Feature importance | Concepts, §7 | yes |
| Show | Drawn tree (top levels) as rules | §6 | yes |
| Show | Feature-importance chart | §7 | yes |
| Show | F1 and ROC-AUC for all three + ROC curve | §5 | yes |
| Show | Honest limit | §9 | yes |
| Question | Gini vs entropy | Q1 | yes |
| Question | Why RF beats a tree; bagging vs boosting | Q2 | yes |
| Question | Why not accuracy | Q3 | yes |
| Question | ROC-AUC 0.5 vs 1.0 | Q4 | yes |
| Question | Importance ≠ causation | Q5 | yes |

## P6 – Krish Jai Ganesh: Neural Network & Model Arena (`P6_neural_net_arena.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. MLP: ReLU hidden layers; softmax / sigmoid / linear outputs | Concepts, §2, §4, §5 | yes |
| How it works | 2. Forward pass, loss, backpropagation, Adam, epochs | Concepts, §3 | yes |
| How it works | 3. Train vs val loss, early stopping, weight penalty, CV of neurons + learning rate | §2, §3 | yes |
| How it works | 4. Model Arena tables per task | §6 | yes |
| How it works | 5. Bootstrap ranges; with vs without PCA | §7, §8 | yes |
| Concept | Neurons, layers, why activations | Concepts, Q1 | yes |
| Concept | ReLU, sigmoid, softmax | Concepts | yes |
| Concept | Backpropagation, gradient descent, learning rate, epochs | Concepts, Q2 | yes |
| Concept | Overfitting: curves, early stopping, regularization | Concepts, §3 | yes |
| Concept | Fair comparison, CV, bootstrap | Concepts, §7, Q5 | yes |
| Show | Network diagram | §1 | yes |
| Show | Training vs validation loss curve | §3 | yes |
| Show | Arena tables, winners and why | §6, Results summary | yes |
| Show | Song Lab demo: energy + loudness → metal | §9 | yes |
| Question | Why activation functions? | Q1 | yes |
| Question | Backpropagation in two sentences | Q2 | yes |
| Question | How hidden neurons were chosen | Q3 | yes |
| Question | Why NNs don't always win on small tabular data | Q4 | yes |
| Question | Is the winner just lucky? | §7, Q5 | yes |

## P7 – Palakollu Sai Nikhil Reddy: Mood Playlists (`P7_mood_playlists.ipynb`)

| Type | Item | Notebook section | Covered |
|---|---|---|---|
| How it works | 1. Mood features only, genre unused | Setup | yes |
| How it works | 2. k-means; elbow + silhouette | Concepts, §1 | yes |
| How it works | 3. Hierarchical on ~1.5k sample; dendrogram; cut | §3 | yes |
| How it works | 4. GMM via EM; soft mood mix | Concepts, §4 | yes |
| How it works | 5. Naming moods from averages | §2 | yes |
| How it works | 6. Silhouette of 3 methods; ARI vs genres | §6 | yes |
| Concept | Unsupervised vs supervised | Concepts | yes |
| Concept | k-means loop, choosing k, weaknesses | Concepts, §8, Q4 | yes |
| Concept | Agglomerative, linkage (Ward/average/complete), dendrogram | Concepts, §3 | yes |
| Concept | Hard vs soft, E/M steps, k-means ⊂ GMM, local optima | Concepts, §4, §8 | yes |
| Concept | Elbow, silhouette, ARI | Concepts, §1, §6 | yes |
| Show | Elbow + silhouette, chosen k | §1 | yes |
| Show | Moods on P1's Song Map | §5 | yes |
| Show | Mood averages table + names + example songs | §2 | yes |
| Show | Dendrogram, one song's mood mix, moods vs genres | §3, §4, §6 | yes |
| Question | How the number of moods was chosen | Q1 | yes |
| Question | Hard vs soft clustering | Q2 | yes |
| Question | E-step and M-step | Q3 | yes |
| Question | Why two k-means runs can differ | §8, Q4 | yes |
| Question | Do moods match genres? | §6, Q5 | yes |

**Total: 136 items, all covered.**

---

## Proposal coverage (Block 14)

| Proposal promise | Where it is delivered |
|---|---|
| Objective 1: classify genre | P3 (LR, NB), P4 (kNN, SVM), P6 (MLP); app: Genre Detector, Home |
| Objective 2: predict valence and popularity | P2 (valence + popularity bonus), P6 (MLP valence); app: Happiness Meter |
| Objective 3: predict hits (popularity ≥ 70) and what drives them | P5 (tree, RF, AdaBoost, importance, rules); app: Hit Predictor |
| Objective 4: mood playlists without labels | P7 (k-means, Ward, GMM); app: Mood Playlists |
| Objective 5: recommend similar songs | P4 (kNN, Euclidean vs cosine); app: Similar Songs, Home |
| Objective 6: compare all algorithms, demo app | P6 Model Arena + bootstrap; Streamlit app (8 pages) |
| Dataset & preparation (duplicates removed, ~6 genres, hit label, class weights, 1–2k sample for dendrograms) | `src/data.py`, P1 §1–§6, P5, P7 §3 (4,267 songs, not ~6,000: re-released duplicates removed) |
| Preliminaries: data types, preprocessing, scaling, encoding, split | P1 §3–§6, `src/data.py` |
| Supervised: linear regression, Ridge/Lasso, logistic regression, NB, kNN, decision trees | P2, P3, P4, P5 |
| Generative vs discriminative | P3 Concepts, §7, Verdict |
| Unsupervised: k-means, hierarchical, DBSCAN, distance metrics | P7, P1 §10, P4 §1–§3 |
| PCA: reduction, 2D visualisation, with vs without PCA | P1 §9, P6 §8 |
| SVM (linear + RBF), Random Forest, AdaBoost, MLP | P4 §5, P5 §3–§4, P6 |
| Optional: least squares (parameter estimation), EM (GMM) | P2 Concepts + §7, P7 §4 |
| Methodology 1–10 (load/clean → targets → preprocess → explore → supervised → moods → recommender → shared evaluation → compare → app) | `src/data.py`, P1–P7, `src/evaluate.py`, `app/` |
| k-fold CV for hyperparameters | every supervised notebook (GridSearchCV / RidgeCV / LassoCV, 5-fold) |
| Depth: from-scratch NumPy algorithms checked vs scikit-learn | `src/scratch.py`; P2 §7, P3 §9, P4 §7, P7 §8 |
| Depth: calibrated hit probabilities | P5 §8 |
| Depth: bootstrap CIs | P6 §7 |
| Metrics: genre accuracy, macro F1, confusion matrix | P3, P4, P6, `src/evaluate.py` |
| Metrics: hit F1, ROC-AUC, precision/recall | P5, P6 |
| Metrics: valence/popularity RMSE, MAE, R² | P2, P6 |
| Metrics: clustering silhouette, ARI vs genre | P7 §6 |
| Metrics: recommender share of top-5 with same genre | P4 §2 |
| Expected: 10+ algorithms compared | 16 models on the leaderboards (+ baselines, clustering, recommender) |
