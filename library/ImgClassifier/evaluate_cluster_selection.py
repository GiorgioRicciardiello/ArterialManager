import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from tabulate import tabulate
import matplotlib.pyplot as plt


class EvaluateClusterSelection:
    """
    Rigorous PCA + KMeans evaluation class for unsupervised clustering
    of vessel colocalization metrics.

    Steps:
        1. Select features
        2. Standardize
        3. PCA with explained variance threshold
        4. Evaluate K in a range: silhouette, DBI, CHI, stability (bootstrapped ARI)
        5. Compute normalized composite score
        6. Select optimal K
        7. Provide plotting functions + tabulate summary


    >Usage
        ecs = EvaluateClusterSelection(
        df=df_all_metrics,
        features=["Manders", "Dice", "Jaccard", "overlap", "weighted"],
        k_range=range(2, 10),
        )

        results = ecs.run()
        ecs.print_results()
        ecs.plot_all()

        best_k, table = ecs.get_best_k()


    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        k_range=range(2, 10),
        variance_threshold=0.9,
        n_boot=30,
        random_state=42,
    ):
        self.df = df.copy()
        self.features = features
        self.k_range = list(k_range)
        self.variance_threshold = variance_threshold
        self.n_boot = n_boot
        self.random_state = random_state

        # outputs
        self.feature_cols = None
        self.X = None
        self.X_scaled = None
        self.pca = None
        self.X_pca = None
        self.results_table = None
        self.best_k = None

    # ---------------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------------
    def _select_features(self):
        matched = []
        for f in self.features:
            matched.extend([c for c in self.df.columns if f in c])
        self.feature_cols = sorted(set(matched))

        if not self.feature_cols:
            raise ValueError("No feature columns matched the provided patterns.")

        X = self.df[self.feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        return X

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
    def _compute_pca(self, X):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.variance_threshold, svd_solver="full")
        self.X_pca = pca.fit_transform(self.X_scaled)
        self.pca = pca

    # ---------------------------------------------------------
    # SINGLE KMeans EVALUATION
    # ---------------------------------------------------------
    def _eval_single_k(self, k):
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
        labels = km.fit_predict(self.X_pca)

        sil = silhouette_score(self.X_pca, labels)
        db = davies_bouldin_score(self.X_pca, labels)
        ch = calinski_harabasz_score(self.X_pca, labels)

        # Stability via bootstrap ARI
        ari_list = []
        N = len(self.X_pca)

        for _ in range(self.n_boot):
            idx = np.random.choice(np.arange(N), size=int(0.8 * N), replace=True)
            km_boot = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
            labels_boot = km_boot.fit_predict(self.X_pca[idx])
            ari = adjusted_rand_score(labels[idx], labels_boot)
            ari_list.append(ari)

        return {
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "stability_ari": np.mean(ari_list),
        }

    # ---------------------------------------------------------
    # FULL EVALUATION
    # ---------------------------------------------------------
    def run(self):
        self.X = self._select_features()
        self._compute_pca(self.X)

        rows = []
        for k in self.k_range:
            rows.append(self._eval_single_k(k))

        df = pd.DataFrame(rows)

        # normalize metrics for composite score
        df["sil_norm"] = (df["silhouette"] - df["silhouette"].min()) / (
            df["silhouette"].max() - df["silhouette"].min()
        )
        df["db_norm"] = 1 - (
            (df["davies_bouldin"] - df["davies_bouldin"].min())
            / (df["davies_bouldin"].max() - df["davies_bouldin"].min())
        )
        df["ch_norm"] = (df["calinski_harabasz"] - df["calinski_harabasz"].min()) / (
            df["calinski_harabasz"].max() - df["calinski_harabasz"].min()
        )
        df["stab_norm"] = df["stability_ari"]

        # weighted composite score
        df["score"] = (
            0.35 * df["sil_norm"]
            + 0.20 * df["db_norm"]
            + 0.20 * df["ch_norm"]
            + 0.25 * df["stab_norm"]
        )

        self.results_table = df.sort_values("score", ascending=False)
        self.best_k = int(self.results_table.iloc[0]["k"])
        return self.results_table

    # ---------------------------------------------------------
    # GET BEST K
    # ---------------------------------------------------------
    def get_best_k(self):
        return self.best_k, self.results_table

    # ---------------------------------------------------------
    # PRETTY PRINT
    # ---------------------------------------------------------
    def print_results(self):
        if self.results_table is None:
            raise RuntimeError("Run .run() first")

        print(
            tabulate(
                self.results_table,
                headers="keys",
                tablefmt="github",
                floatfmt=".3f",
            )
        )
        print(f"\nRecommended number of clusters: **{self.best_k}**")

    # ---------------------------------------------------------
    # PLOTTING UTILITIES
    # ---------------------------------------------------------
    def plot_pca_scree(self):
        if self.pca is None:
            raise RuntimeError("Run .run() first")

        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker="o")
        plt.axhline(self.variance_threshold, color="red", linestyle="--")
        plt.xlabel("PCA Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Scree Plot")
        plt.grid(True)
        plt.show()

    def plot_pca_scatter(self):
        if self.X_pca is None:
            raise RuntimeError("Run .run() first")

        plt.figure(figsize=(6, 5))
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], s=20, alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection")
        plt.grid(True)
        plt.show()

    def plot_metrics_vs_k(self):
        if self.results_table is None:
            raise RuntimeError("Run .run() first")

        df = self.results_table.sort_values("k")

        plt.figure(figsize=(10, 6))
        plt.plot(df["k"], df["silhouette"], "-o", label="Silhouette")
        plt.plot(df["k"], df["davies_bouldin"], "-o", label="Davies–Bouldin")
        plt.plot(df["k"], df["calinski_harabasz"], "-o", label="Calinski–Harabasz")
        plt.plot(df["k"], df["stability_ari"], "-o", label="Stability (ARI)")
        plt.plot(df["k"], df["score"], "-o", label="Composite Score", linewidth=3)

        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Metric value")
        plt.title("Cluster Evaluation Metrics vs K")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self):
        self.plot_pca_scree()
        self.plot_pca_scatter()
        self.plot_metrics_vs_k()
