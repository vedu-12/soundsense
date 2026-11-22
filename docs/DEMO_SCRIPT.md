# SoundSense – Live Demo Script

About 1–2 minutes per presenter, in P1 → P7 order. One laptop runs the app; the presenter speaks while someone else clicks (or the presenter clicks).

**Before the demo**
- `streamlit run app/Home.py` → opens http://localhost:8501. Wait until the Home page shows the "HUMBLE. – Kendrick Lamar" card.
- Use a wide browser window (≥ 1280 px).
- The song search box remembers the last song across pages; type to search (song or artist).
- Every song used below is an **🆕 unseen test song**: the models never saw it while training, so the predictions are honest. The main song card shows this tag. (If you search other songs, a "📚 training song" tag means the models learned from it.)
- Each page has two closed boxes at the bottom: **How it works** (open it while explaining) and **Behind the scenes** (model, test scores, one plot).

**Opening (whoever starts, 20 s), on Home:** "SoundSense takes only how a song *sounds*, its Spotify audio features, and from that guesses its genre, how happy it sounds, whether it's a hit, its mood, and songs like it. Here's *HUMBLE.* by Kendrick Lamar, a song our models never saw: hip-hop, 84% sure, Party mood, and five look-alike songs, all hip-hop. The hit gauge says only 14%, even though it *is* a hit. Fame isn't in the audio, and we'll come back to that. Each of us built one of these parts."

---

## P1 – Aryan: The Data & Song Map

**Page:** 🗺️ Song Map

1. In **Find a song on the map**, type `One Night in Tokyo` and pick *One Night in Tokyo – Beast In Black*. The labelled white star jumps to the right side (loud, energetic).
2. Point at the colours: classical (blue) on the left, jazz in the middle, the four loud genres crowded on the right.
3. Switch **Colour by** → **Mood**, then back to **Genre**.
4. Click one of the red-ringed dots at the very bottom (e.g. the lowest one). The card on the right shows *Chanting of the Gayatri Mantra 108 Times*, an "unusual song".
5. Open **Behind the scenes**: 51% of variance in 2D, 7 components for 90%, 59 unusual songs.

**Say:** "We started with 114,000 rows, kept 6 clearly different genres and removed duplicates and re-releases, leaving 4,267 songs, split 80/20 once for everyone, with the scaler fit only on training songs so nothing leaks. PCA squeezes 10 audio features into 2 directions: left-to-right is intensity, quiet and acoustic versus loud and energetic. DBSCAN flags songs with too few close neighbours; the most unusual ones are very long devotional chants."

## P2 – Gunneet: Happiness Meter

**Page:** 😊 Happiness Meter

1. Search `OUT OUT` (Joel Corry, Jax Jones, Charli XCX, Saweetie). Gauge ≈ 63% predicted; white mark = real 80%.
2. Point at the bar chart: which features push happiness **up** (blue) or **down** (orange).
3. Search `In the Name of Love` (Martin Garrix, Bebe Rexha). Predicted ≈ 35%, real 17%: a gloomier song is predicted as less happy.
4. Open **Behind the scenes**: test R² ≈ 0.30, polynomial model only 0.10, popularity R² ≈ 0.11.

**Say:** "This is linear regression: happiness is a weighted sum of the features, with weights chosen by least squares. Danceability and energy make songs sound happier. Notice the predictions are pulled towards the middle: sound explains about 30% of happiness, and lyrics and harmony matter too. Polynomial features overfit; Ridge and Lasso barely changed things because the plain model wasn't overfitting, though Lasso dropped three useless features. Predicting popularity is much harder, because fame and marketing aren't in the audio."

## P3 – Shreyash: Genre Detector

**Page:** 🎸 Genre Detector

1. Search `HUMBLE.`. Both models say hip-hop (logistic regression 67%, Naive Bayes 50%), both ✅.
2. Search `Sexy Bitch` (David Guetta; Akon). It's **edm**: logistic regression says hip-hop ❌, Naive Bayes says edm ✅. The two models really do think differently.
3. In the confusion matrix, point at the **hip-hop row / edm column** (32 songs) and the **edm row / hip-hop column** (21).
4. Switch **Model** to *Gaussian Naive Bayes* and show its mistakes are more spread out.

**Say:** "Logistic regression is discriminative: it learns the boundaries between genres directly and uses softmax to give probabilities. Naive Bayes is generative: it learns a bell curve per genre and feature and applies Bayes' rule, assuming features are independent, which isn't true here because energy and loudness move together. Logistic regression wins, 71.5% vs 64.9% accuracy, though Naive Bayes is better when there are very few training songs. The biggest mix-up is hip-hop versus edm: both are danceable, energetic and electronic."

## P4 – Amithav: Similar Songs

**Page:** 🎧 Similar Songs

1. Search `HUMBLE.`. All 5 recommendations are hip-hop (*Clout*, *Dua Lipa*, *Oouuu*, *Old Town Road*, *Gangsta*). Point at the dotted outline (= HUMBLE.) on each mini radar.
2. Search `OUT OUT`, then click **Cosine**. The list changes: *Follow Me* and *That That* appear and the order shifts.
3. Search `One Night in Tokyo`. All 5 picks are heavy-metal with both measures.
4. Open **Behind the scenes**: 62% of top-5 picks share the genre (random: 18%); linear SVM genre accuracy 72%.

**Say:** "Each song is a point in a 10-dimensional space; we recommend its 5 nearest neighbours. Euclidean measures the straight-line gap; cosine measures the angle, the same shape of song even at a different intensity. Scaling is essential: without it the recommender just matches song length and only 27% of picks share the genre. The same geometry classifies genres: kNN votes among neighbours, and an SVM finds the widest-margin boundary, with the kernel trick allowing curved boundaries."

## P5 – Vedang: Hit Predictor

**Page:** 📈 Hit Predictor

1. Search `OUT OUT`. Hit chance ≈ 44%, about 5× the average song: "Sounds like a hit" (real popularity 81 ✅). Read the reasons, e.g. "High danceability… like most hits".
2. Search `One Night in Tokyo`. Hit chance ≈ 1%: low danceability and almost no acousticness, the heavy-metal profile (no heavy-metal song in our data is a hit).
3. Search `HUMBLE.`. Only ≈ 14%, yet its real popularity is 84: the honest limit.
4. Open **Behind the scenes**: test ROC-AUC 0.78, single tree 0.73, and the feature-importance chart.

**Say:** "A hit means popularity 70 or more, only 9% of songs, so accuracy is useless: saying 'never a hit' is 92% accurate. We use class weights and judge by F1 and ROC-AUC. One decision tree overfits; a Random Forest averages 300 trees and ranks hits best, at 0.78 AUC, ahead of AdaBoost and one tree. We calibrated the probabilities so 30% really means about 30%. Hits tend to be danceable, louder, less acoustic and shorter. But HUMBLE. shows the limit: Kendrick Lamar's fame and marketing aren't in the audio, and importance isn't causation."

## P6 – Krish: Neural Network & Model Arena

**Page:** 🧪 Song Lab & Model Arena

1. **Song Lab** tab: click **Reset to a typical song**. The network says country.
2. Drag **Energy** to 1.0, **Loudness** to about −2 dB and **Acousticness** to 0. The guess becomes **heavy-metal**.
3. Drag **Energy** to 0.1, **Loudness** to −25, **Acousticness** to 0.9, **Instrumental** to 0.8. The guess becomes **classical** (≈ 97%).
4. **Model Arena** tab: Task = **Genre**. The winner is the MLP at 0.728, the runner-up the linear SVM at 0.725, and **"Is the win real?" says Not clearly**.
5. Click **Hit** (Random Forest wins) and **Happiness** (MLP wins). Pick a model in **Inspect a model** to show its confusion matrix.

**Say:** "The network has one hidden layer of 64 ReLU neurons and a softmax output; backpropagation tells each weight how to change, and early stopping stops it memorising the training songs. The Model Arena scores all 16 models on the same 854 test songs. The network wins genre and happiness, the forest wins hits, but bootstrapping the test set 1,000 times shows the top models are statistically tied. On a small table of 26 features, a neural network doesn't automatically win."

## P7 – Nikhil: Mood Playlists

**Page:** 🌈 Mood Playlists

1. Mood **Chill** is selected. Point at the cards: country, jazz and even XXXTENTACION's hip-hop, with green mood-mix bars. Most are nearly 100% Chill; *Beautiful Crazy* (Luke Combs) shows slivers of other moods (90% Chill).
2. Click **Sad**: Johnny Cash, Einaudi, classical piano. Then **Party** and **Intense**.
3. Point at the map on the right: each mood is its own region of the Song Map.
4. Open **Behind the scenes**: silhouette 0.25, moods vs genres ARI 0.29, and the genre-by-mood heatmap (country appears in all four moods).

**Say:** "Nobody labelled moods; the computer found them. k-means grouped songs on five feel features (energy, happiness, tempo, acousticness and danceability) into 4 clusters, chosen with the elbow and silhouette scores, and we named them from their averages. Hierarchical clustering is consistent with four groups. A Gaussian Mixture Model trained with EM gives every song a probability per mood: that's the mix bar. Moods only partly match genres, ARI 0.29: heavy-metal is almost all Intense, but country spreads over every mood."

---

**Closing (10 s):** "Everything is reproducible: one shared split, one evaluation script, every random seed fixed, and four core algorithms re-written from scratch in NumPy to check scikit-learn. Thank you!"
