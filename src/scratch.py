"""Core algorithms written from scratch with NumPy only, to check we understand what scikit-learn does.

Each class has the same fit / predict style as scikit-learn. They are compared with scikit-learn in the
"Built from scratch" sections of P2 (linear regression), P3 (Naive Bayes), P4 (kNN) and P7 (k-means).
"""
import numpy as np


class KNNClassifierScratch:
    """k-nearest neighbours: store the training songs; predict by majority vote of the k closest."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, float)          # "lazy": training = remembering the data
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        # squared Euclidean distance from every query song to every training song: |a|² + |b|² - 2a·b
        d = (X ** 2).sum(1)[:, None] + (self.X ** 2).sum(1)[None, :] - 2 * X @ self.X.T
        nearest = np.argsort(d, axis=1, kind="stable")[:, :self.k]
        votes = self.y[nearest]
        # majority vote; ties go to the class whose nearest member is closest (first in the sorted list)
        out = []
        for row in votes:
            labels, counts = np.unique(row, return_counts=True)
            best = labels[counts == counts.max()]
            out.append(next(v for v in row if v in best))
        return np.array(out)


class GaussianNBScratch:
    """Gaussian Naive Bayes: one bell curve per (class, feature); predict with Bayes' rule in log space."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * X.var(axis=0).max()          # same smoothing rule as scikit-learn
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + eps
        return self

    def _log_posterior(self, X):
        X = np.asarray(X, float)
        # log P(x | class) = sum over features of log N(x; mean, var)   (independence assumption)
        ll = -0.5 * (np.log(2 * np.pi * self.var_)[None, :, :] +
                     (X[:, None, :] - self.mean_[None, :, :]) ** 2 / self.var_[None, :, :]).sum(axis=2)
        return ll + np.log(self.prior_)[None, :]                 # + log prior

    def predict_proba(self, X):
        lp = self._log_posterior(X)
        lp -= lp.max(axis=1, keepdims=True)                      # numerical stability
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]


class LinearRegressionScratch:
    """Least-squares linear regression, solved by the normal equation or by gradient descent."""

    def __init__(self, method="normal", lr=0.05, n_iter=5000):
        self.method, self.lr, self.n_iter = method, lr, n_iter

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb = np.c_[np.ones(len(X)), X]                           # add a column of 1s for the intercept b
        if self.method == "normal":
            # w = (XᵀX)⁻¹ Xᵀy, computed with solve() (more stable than an explicit inverse)
            w = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
            self.loss_history_ = None
        else:
            w = np.zeros(Xb.shape[1])
            self.loss_history_ = []
            n = len(y)
            for _ in range(self.n_iter):
                err = Xb @ w - y
                self.loss_history_.append((err ** 2).mean())
                grad = 2 / n * Xb.T @ err                        # gradient of the mean squared error
                w -= self.lr * grad                              # step downhill
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class KMeansScratch:
    """k-means: repeat (assign each point to the nearest centre, move centres to the mean) until stable.
    Uses k-means++ starting centres and keeps the best of n_init runs (lowest inertia)."""

    def __init__(self, k=4, n_init=10, max_iter=300, random_state=42):
        self.k, self.n_init, self.max_iter = k, n_init, max_iter
        self.rng = np.random.default_rng(random_state)

    def _init_centres(self, X):
        centres = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.k):                               # k-means++: far points are more likely
            d2 = ((X[:, None, :] - np.array(centres)[None]) ** 2).sum(2).min(1)
            centres.append(X[self.rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centres)

    def _run(self, X):
        C = self._init_centres(X)
        for _ in range(self.max_iter):
            labels = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)          # assign step
            newC = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(self.k)])  # update
            if np.allclose(newC, C):
                break
            C = newC
        labels = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        inertia = ((X - C[labels]) ** 2).sum()
        return C, labels, inertia

    def fit(self, X):
        X = np.asarray(X, float)
        best = min((self._run(X) for _ in range(self.n_init)), key=lambda r: r[2])
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return ((X[:, None, :] - self.cluster_centers_[None]) ** 2).sum(2).argmin(1)
