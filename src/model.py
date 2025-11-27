import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class StatisticalJumpModel(object):
    def __init__(
            self,
            n_states: int,
            jump_penalty: float = 50.0,
            max_iter: int = 5,
            random_state: int = 42,
            dist_method: str = "kmeans",   # "kmeans", "median", "huber", "mahalanobis", "gmm"
            huber_delta: float = 1.0,
        ) -> None:

        self.n_states = n_states
        self.jump_penalty = float(jump_penalty)
        self.max_iter = max_iter
        self.random_state = random_state

        self.dist_method = dist_method.lower()
        if self.dist_method not in {"kmeans", "median", "huber", "mahalanobis", "gmm"}:
            raise ValueError("dist_method must be one of "
                             "['kmeans', 'median', 'huber', 'mahalanobis', 'gmm']")

        self.huber_delta = huber_delta

        self.centroids_ = None
        self.train_states_ = None
        self.last_state_ = None

        self.cov_inv_ = None # global covariance inverse (mahalanobis 전용)
        self.gmm_ = None # GaussianMixture 객체 (gmm 전용)
        self.gmm_precisions_ = None # (K, D, D)

    def _check_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

    def _compute_distances(
            self, X_df: pd.DataFrame,
            centroids=None
        ) -> np.ndarray:
        """
        X_df: DataFrame (T, D)
        centroids: (K, D)
        returns: distances ndarray (T, K)
        """
        X = X_df.values.astype(float)
        T, D = X.shape
        if centroids is None:
            centroids = self.centroids_

        K = centroids.shape[0]
        distances = np.zeros((T, K))

        # Euclidean (kmeans / median / huber)
        if self.dist_method in {"kmeans", "median", "huber"}:
            for k in range(K):
                diff = X - centroids[k]
                distances[:, k] = 0.5 * np.sum(diff ** 2, axis=1)

        # Mahalanobis
        elif self.dist_method == "mahalanobis":
            if self.cov_inv_ is None:
                cov = np.cov(X.T)
                self.cov_inv_ = np.linalg.pinv(cov)
            cov_inv = self.cov_inv_
            for k in range(K):
                diff = X - centroids[k]
                distances[:, k] = 0.5 * np.einsum("ij,jk,ik->i", diff, cov_inv, diff)

        # GMM covariance 기반 Mahalanobis
        elif self.dist_method == "gmm":
            precisions = self.gmm_precisions_
            for k in range(K):
                diff = X - centroids[k]
                distances[:, k] = 0.5 * np.einsum("ij,jk,ik->i", diff, precisions[k], diff)

        return distances

    def _smooth_dynamic_programming(self, X_df: pd.DataFrame, centroids=None) -> np.ndarray:
        """
        DP + backward pass (offline smoothing)
        returns ndarray (T,)
        """
        X = X_df.values.astype(float)

        if centroids is None:
            centroids = self.centroids_
        if centroids is None:
            raise RuntimeError("centroids가 정의되지 않았습니다.")

        T, D = X.shape
        K = centroids.shape[0]

        distances = self._compute_distances(X_df, centroids=centroids)

        dp = np.full((T, K), np.inf)
        path = np.zeros((T, K), dtype=int)

        dp[0] = distances[0]

        for t in range(1, T):
            for k in range(K):
                transition_costs = dp[t - 1] + distances[t, k]
                transition_costs[np.arange(K) != k] += self.jump_penalty

                best_prev = np.argmin(transition_costs)
                dp[t, k] = transition_costs[best_prev]
                path[t, k] = best_prev

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmin(dp[-1])
        for t in range(T - 2, -1, -1):
            states[t] = path[t + 1, states[t + 1]]

        return states

    def _update_centroids(self, X_df: pd.DataFrame, states: np.ndarray) -> np.ndarray:
        X = X_df.values.astype(float)
        K = self.n_states
        D = X.shape[1]

        centroids = np.zeros((K, D))

        for k in range(K):
            mask = (states == k)
            if not np.any(mask):
                centroids[k] = self.centroids_[k] if (self.centroids_ is not None) else 0.0
                continue

            Xk = X[mask]

            if self.dist_method == "median":
                centroids[k] = np.median(Xk, axis=0)

            elif self.dist_method == "huber":
                mu = Xk.mean(axis=0)
                r = Xk - mu
                r_norm = np.linalg.norm(r, axis=1)
                delta = self.huber_delta

                w = np.ones_like(r_norm)
                big = r_norm > delta
                w[big] = delta / (r_norm[big] + 1e-12)
                w = w / (w.sum() + 1e-12)

                centroids[k] = (w[:, None] * Xk).sum(axis=0)

            else:
                # kmeans / mahalanobis / gmm → 기본 mean
                centroids[k] = Xk.mean(axis=0)

        return centroids

    def fit(self, X_train: pd.DataFrame) -> None:
        X_train = pd.DataFrame(X_train)
        X_np = X_train.values.astype(float)

        # 초기 클러스터링
        if self.dist_method == "gmm":
            gmm = GaussianMixture(
                n_components=self.n_states,
                covariance_type="full",
                random_state=self.random_state,
            )
            gmm.fit(X_np)
            initial_states = gmm.predict(X_np)
            self.gmm_ = gmm
            self.gmm_precisions_ = gmm.precisions_
            centroids = gmm.means_.copy()
        else:
            kmeans = KMeans(
                n_clusters=self.n_states,
                init="k-means++",
                random_state=self.random_state,
                n_init=10,
            )
            initial_states = kmeans.fit_predict(X_np)
            centroids = self._update_centroids(X_train, initial_states)

        # Mahalanobis용 global covariance
        if self.dist_method == "mahalanobis":
            cov = np.cov(X_np.T)
            self.cov_inv_ = np.linalg.pinv(cov)

        # Coordinate Descent
        for _ in range(self.max_iter):
            old = centroids.copy()
            states = self._smooth_dynamic_programming(X_train, centroids=centroids)
            centroids = self._update_centroids(X_train, states)
            if np.allclose(old, centroids, rtol=1e-4, atol=1e-6):
                break

        self.centroids_ = centroids
        self.train_states_ = pd.Series(states, index=X_train.index, name="state")
        self.last_state_ = None

    def smooth(self, X: pd.DataFrame) -> pd.Series:
        """
        Offline smoothing → 미래참조 있음
        """
        self._check_fitted()

        X = pd.DataFrame(X).copy()
        states = self._smooth_dynamic_programming(X, centroids=self.centroids_)
        return pd.Series(states, index=X.index, name="state")

    def filter_sequence(self, X: pd.DataFrame, init_state=None) -> pd.Series:
        """
        Online filtering → 미래참조 없음
        """
        self._check_fitted()

        X_df = pd.DataFrame(X).copy()
        T = len(X_df)
        K = self.n_states

        distances = self._compute_distances(X_df, centroids=self.centroids_)

        states = np.zeros(T, dtype=int)

        if init_state is None:
            states[0] = int(np.argmin(distances[0]))
        else:
            states[0] = int(init_state)

        for t in range(1, T):
            prev = states[t - 1]
            stay_cost = distances[t, prev]
            switch_costs = distances[t] + self.jump_penalty
            switch_costs[prev] = stay_cost
            states[t] = int(np.argmin(switch_costs))

        return pd.Series(states, index=X_df.index, name="state")

    def reset_online(self, init_state=None):
        if init_state is not None:
            self.last_state_ = int(init_state)
        else:
            self.last_state_ = None

    def filter_step(self, x_t):
        """
        x_t: pandas Series 또는 DataFrame(1-row)
        returns: 현재 상태값(int)
        """
        self._check_fitted()

        if isinstance(x_t, pd.DataFrame):
            x = x_t.values.astype(float).reshape(1, -1)
        elif isinstance(x_t, pd.Series):
            x = x_t.values.astype(float).reshape(1, -1)
        else:
            raise ValueError("x_t must be a pandas Series or single-row DataFrame")

        distances = np.empty((1, self.n_states), dtype=float)
        for k in range(self.n_states):
            diff = x - self.centroids_[k]
            distances[0, k] = 0.5 * np.sum(diff**2)

        d = distances[0]

        if self.last_state_ is None:
            state_t = int(np.argmin(d))
        else:
            prev = self.last_state_
            stay_cost = d[prev]
            switch_costs = d + self.jump_penalty
            switch_costs[prev] = stay_cost
            state_t = int(np.argmin(switch_costs))

        self.last_state_ = state_t
        return state_t

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        각 시점에 대해 softmax(-distance)를 이용한 상태 확률을 반환.
        columns = state_0, state_1, ..., state_(K-1)
        """
        self._check_fitted()

        X = pd.DataFrame(X)
        distances = self._compute_distances(X, centroids=self.centroids_)  # (T, K)

        # softmax(-distance)
        neg_d = -distances
        exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))  # overflow 방지
        prob = exp_d / exp_d.sum(axis=1, keepdims=True)

        cols = [f"state_{k}_prob" for k in range(self.n_states)]
        prob_df = pd.DataFrame(prob, index=X.index, columns=cols)
        return prob_df