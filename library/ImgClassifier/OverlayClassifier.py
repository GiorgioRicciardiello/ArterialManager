from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from pathlib import Path
from config.config import CONFIG
import matplotlib.pyplot as plt
import seaborn as sns
from library.ImgClassifier.CreateDataset import make_ml_feature_table, summarize_ml_features
from library.ImgClassifier.evaluate_cluster_selection import EvaluateClusterSelection
from typing import List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from umap import UMAP
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from umap import UMAP
import hdbscan
from tabulate import tabulate
import base64
import cv2
import plotly.express as px

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import cv2

def unsupervised_overlap_classifier(
    df_all_metrics: pd.DataFrame,
    features: list[str] = None,
    n_clusters: int = 3
) -> tuple[pd.DataFrame, KMeans, PCA, pd.DataFrame]:
    """
    Unsupervised clustering of vessel colocalization metrics.

    Parameters
    ----------
    df_all_metrics : pd.DataFrame
        Rows = cells, columns = metrics.
    features : list of str, optional
        Feature names or string patterns to select metric columns.
        Example: ['Manders', 'Dice'] will match all columns containing those substrings.

    n_clusters : int, default 3
        Number of KMeans clusters.

    Returns
    -------
    (df_labels, kmeans, pca)
        df_labels : DataFrame with original data, cluster labels, and unsupervised class.
        kmeans    : fitted KMeans model.
        pca       : fitted PCA model.
    """
    df = df_all_metrics.copy()

    # --- feature selection ---
    if features is None:
        raise ValueError("No features specified.")

    # if features include patterns (e.g., "Manders"), select all matching
    matched = []
    for f in features:
        matched.extend([c for c in df.columns if f in c])
    feature_cols = sorted(set(matched))

    if not feature_cols:
        raise ValueError("No valid feature columns found for clustering.")

    # --- clean numeric data ---
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA for dimensionality reduction ---
    pca = PCA(n_components=0.9, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)

    # --- clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, clusters)

    # --- results ---
    df_labels = df.loc[X.index, feature_cols].copy()
    df_labels["cluster"] = clusters

    # order clusters by mean overlap strength (if available)
    overlap_cols = [c for c in df_labels.columns if "Manders_M2" in c or "Dice_skeletons" in c]
    if overlap_cols:
        strength = df_labels.groupby("cluster")[overlap_cols].mean().mean(axis=1)
        ordering = strength.sort_values().index.tolist()
        mapping = {old: new for new, old in enumerate(ordering)}
        df_labels["cluster_ranked"] = df_labels["cluster"].map(mapping)
    else:
        df_labels["cluster_ranked"] = df_labels["cluster"]

    class_names = {0: "Weak", 1: "Moderate", 2: "Strong"}
    df_labels["unsupervised_class"] = df_labels["cluster_ranked"].map(class_names)

    print(f"✅ Unsupervised clustering done — Silhouette score = {silhouette:.3f}")
    return df_labels, kmeans, pca, X_pca





def phenotype_vessels_umap_hdbscan(
    df: pd.DataFrame,
    features: list[str],
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    min_cluster_size: int = 20,
    min_samples: int = 10,
    use_gmm: bool = True,
    gmm_components: int = 3,
):
    """
    Absolute best workflow for unlabeled colocalization data:
        1. Feature selection
        2. Scaling
        3. UMAP manifold (non-linear embedding)
        4. HDBSCAN soft clustering (density-based)
        5. Continuous UMAP score (1D)
        6. Optional Gaussian Mixture soft labels on manifold
        7. Plots + tabulated summary

    Returns
    -------
    results : pd.DataFrame
        Contains UMAP coords, HDBSCAN cluster labels, probabilities,
        continuous UMAP score, and optional GMM soft probabilities.
    umap_model : fitted UMAP model
    hdbscan_model : fitted HDBSCAN clusterer
    gmm : fitted GMM model (optional, None if disabled)
    """

    # ------------------------------------------------------------
    # 1. Feature Selection
    # ------------------------------------------------------------
    matched = []
    for f in features:
        matched.extend([c for c in df.columns if f in c])
    feature_cols = sorted(set(matched))

    if len(feature_cols) == 0:
        raise ValueError("No matching feature columns found.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_index = X.index

    # ------------------------------------------------------------
    # 2. Scaling
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # 3. UMAP manifold
    # ------------------------------------------------------------
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )

    embedding = umap_model.fit_transform(X_scaled)

    # ------------------------------------------------------------
    # 4. HDBSCAN soft clustering
    # ------------------------------------------------------------
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        prediction_data=True
    )

    labels = hdb.fit_predict(embedding)
    probs = hdb.probabilities_

    # ------------------------------------------------------------
    # 5. Continuous UMAP score
    # ------------------------------------------------------------
    umap_score = (embedding[:, 0] - embedding[:, 0].min()) / (
        embedding[:, 0].max() - embedding[:, 0].min()
    )

    # ------------------------------------------------------------
    # 6. Optional GMM soft clustering on UMAP space
    # ------------------------------------------------------------
    gmm = None
    gmm_probs = None
    if use_gmm:
        gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type="full",
            random_state=42
        )
        gmm.fit(embedding)
        gmm_probs = gmm.predict_proba(embedding)

    # ------------------------------------------------------------
    # 7. Build results DataFrame
    # ------------------------------------------------------------
    results = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "HDBSCAN_label": labels,
        "HDBSCAN_prob": probs,
        "UMAP_score": umap_score,
    }, index=X_index)

    if use_gmm:
        for i in range(gmm_components):
            results[f"GMM_prob_{i}"] = gmm_probs[:, i]

    # ------------------------------------------------------------
    # 8. Plotting
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=probs, cmap="viridis", s=25, alpha=0.8)
    plt.title("UMAP + HDBSCAN (probabilities)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Cluster confidence")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=umap_score, cmap="plasma", s=25, alpha=0.8)
    plt.title("Continuous UMAP Score (Phenotype)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="UMAP Score")
    plt.show()

    # ------------------------------------------------------------
    # 9. Tabulated Summary
    # ------------------------------------------------------------
    summary = results.describe().T
    print("\nSummary Statistics:")
    print(tabulate(summary, headers="keys", tablefmt="github", floatfmt=".4f"))

    return results, umap_model, hdb, gmm




def vessel_umap_composite(
    df,
    features,
    img_path_col="img_path",
    n_neighbors=30,
    min_dist=0.1,
    min_cluster_size=20,
    min_samples=10,
    use_gmm=True,
    gmm_components=3,
    image_size=64
):
    """
    Full phenotyping pipeline:
      - feature scaling
      - composite overlap score calculation
      - UMAP manifold learning
      - HDBSCAN soft clustering
      - continuous phenotype score (UMAP1)
      - soft GMM clusters (optional)
      - image overlay on UMAP
    """

    # ----------------------------------------------------------------------
    # 0) FEATURE EXTRACTION
    # ----------------------------------------------------------------------
    matched = []
    for f in features:
        matched.extend([c for c in df.columns if f in c])
    feature_cols = sorted(set(matched))

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_index = X.index
    df = df.loc[X_index]

    # normalize all metrics for composite score
    scaler_norm = MinMaxScaler()
    X_norm = pd.DataFrame(scaler_norm.fit_transform(X), columns=X.columns)

    # ----------------------------------------------------------------------
    # 1) COMPOSITE OVERLAP SCORE (OPTION 2)
    # ----------------------------------------------------------------------
    # A = intensity colocalization
    A_cols = [
        'weighted_overlap_red_absolute', 'weighted_overlap_green_absolute',
        'weighted_overlap_mean_absolute', 'sum_overlap_strength_absolute',
        'sum_heatmap_intensity', 'frac_heatmap_intensity',
        'Manders_M2_intensity', 'Manders_M1_intensity'
    ]

    # B = structural overlap
    B_cols = [
        'Dice_skeletons_intensity', 'Jaccard_skeletons_intensity',
        'Tanimoto_intensity', 'assd_skeleton_absolute', 'hausdorff_skeleton_absolute'
    ]

    # C = channel balance
    C_cols = [
        'green_red_ratio_absolute', 'red_green_ratio_absolute'
    ]

    # D = information metrics
    D_cols = [
        'mutual_information_absolute', 'cosine_similarity_absolute',
        'icq_absolute', 'bhattacharyya_coeff_absolute', 'hellinger_distance_absolute'
    ]

    def safe_mean(cols):
        return X_norm[cols].mean(axis=1) if len(cols) > 0 else 0

    overlap_A = safe_mean([c for c in A_cols if c in X_norm])
    overlap_B = safe_mean([c for c in B_cols if c in X_norm])
    overlap_C = safe_mean([c for c in C_cols if c in X_norm])
    overlap_D = safe_mean([c for c in D_cols if c in X_norm])

    composite_score = (
        0.55 * overlap_A +
        0.30 * overlap_B +
        0.15 * overlap_C +
        0.20 * overlap_D
    )

    composite_score = MinMaxScaler().fit_transform(composite_score.values.reshape(-1,1)).flatten()

    # ----------------------------------------------------------------------
    # 2) UMAP EMBEDDING
    # ----------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=42
    )
    embedding = umap_model.fit_transform(X_scaled)

    # ----------------------------------------------------------------------
    # 3) HDBSCAN
    # ----------------------------------------------------------------------
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
        prediction_data=True
    )
    labels = hdb.fit_predict(embedding)
    probs = hdb.probabilities_

    # ----------------------------------------------------------------------
    # 4) CONTINUOUS SCORE (UMAP1)
    # ----------------------------------------------------------------------
    umap_score = (embedding[:, 0] - embedding[:, 0].min()) / (
        embedding[:, 0].max() - embedding[:, 0].min()
    )

    # ----------------------------------------------------------------------
    # 5) OPTIONAL GMM ON UMAP
    # ----------------------------------------------------------------------
    gmm = None
    gmm_probs = None
    if use_gmm:
        gmm = GaussianMixture(n_components=gmm_components, covariance_type="full", random_state=42)
        gmm.fit(embedding)
        gmm_probs = gmm.predict_proba(embedding)

    # ----------------------------------------------------------------------
    # 6) BUILD RESULTS DATAFRAME
    # ----------------------------------------------------------------------
    results = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "HDBSCAN_label": labels,
        "HDBSCAN_prob": probs,
        "CompositeScore": composite_score,
        "UMAP_score": umap_score,
        img_path_col: df[img_path_col]
    }, index=X_index)

    if use_gmm:
        for i in range(gmm_components):
            results[f"GMM_prob_{i}"] = gmm_probs[:, i]

    # ----------------------------------------------------------------------
    # 7) UMAP PLOT WITH IMAGE OVERLAY (FIXED COLORBAR)
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20,14))
    ax.set_title("UMAP Embedding with Image Overlay")

    # Base scatter for reference
    sc = ax.scatter(results["UMAP1"], results["UMAP2"],
                    c=composite_score, cmap="plasma", s=20, alpha=0.4)

    # Overlay images at reduced resolution
    for idx, r in results.iterrows():
        img = cv2.imread(r[img_path_col])
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))

        im = OffsetImage(img, zoom=0.35)
        ab = AnnotationBbox(im, (r["UMAP1"], r["UMAP2"]), frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # --- FIXED COLORBAR ---
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=composite_score.min(), vmax=composite_score.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])

    plt.colorbar(sm, ax=ax, label="Composite Score")
    plt.show()


    return results, umap_model, hdb, gmm


def plotly_umap_with_images(results, img_path_col="img_path", image_size=128):
    """
    Creates an interactive UMAP plot in a web browser where hovering over each point
    shows the corresponding vessel image.
    """

    # -------------------------------------------------------------------
    # Encode each image as base64 for Plotly hover display
    # -------------------------------------------------------------------
    encoded_images = []

    for path in results[img_path_col]:
        img = cv2.imread(path)
        if img is None:
            encoded_images.append("")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))

        # Convert to PNG bytes
        _, buffer = cv2.imencode(".png", img)
        encoded = base64.b64encode(buffer).decode("utf-8")
        encoded_images.append(f"<img src='data:image/png;base64,{encoded}' width='160'>")

    # Add encoded images to dataframe
    df_plot = results.copy()
    df_plot["img_b64"] = encoded_images

    # -------------------------------------------------------------------
    # Plotly scatter with image hover
    # -------------------------------------------------------------------
    fig = px.scatter(
        df_plot,
        x="UMAP1",
        y="UMAP2",
        color="CompositeScore",
        color_continuous_scale="Plasma",
        hover_data={"img_b64": True, "UMAP1": False, "UMAP2": False}
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "%{customdata[0]}",  # image
        ])
    )

    fig.update_layout(
        title="Interactive UMAP Viewer (Hover for Image)",
        width=900,
        height=700
    )

    fig.show()   # Opens in your default browser

if __name__ == "__main__":

    # %% format table to feature matrix
    df = make_ml_feature_table(
        path_in=CONFIG.get('paths')['data'].joinpath(r"features_imgs\Imgfeatures.xlsx"),
        path_out=CONFIG.get('paths')['data'].joinpath(r"features_imgs\ImgfeaturesWide.xlsx"),
        id_col="cell",
        path_imgs=CONFIG.get('paths')['local_images_output'].joinpath('processed_overlap'),
        overwrite=False
    )
    id_cols = ['cell', 'significant_overlap_absolute', 'img_path']
    # %% Define feature columns
    col_features = [col for col in df.columns if (not df[col].isna().any()) and (col not in id_cols)]


    # %% Evalaute cluster selection
    EVALUATE_CLUSTERS = False
    if EVALUATE_CLUSTERS:
        ecs = EvaluateClusterSelection(
            df=df,
            features=col_features,
            k_range=range(2, 14),
            n_boot=10,
        )

        results = ecs.run()
        ecs.print_results()
        ecs.plot_all()

        best_k, table = ecs.get_best_k()
    else:
        best_k = 10

    # %% Gets stats
    df_stats = summarize_ml_features(df, features=col_features)
    # drop nans
    df_features = df[col_features]

    DECOMPOSTION_METHOD = 'umapdb'
    if DECOMPOSTION_METHOD == 'pca':
        # %% PCA + Kmeans
        df_labels, kmeans, pca,X_pca = unsupervised_overlap_classifier(
            df_all_metrics=df_features,
            features=col_features,
            n_clusters=best_k
        )

        # Get valid indices (those kept after cleaning in unsupervised_overlap_classifier)
        valid_idx = df_labels.index

        # Safely attach identifier column(s)
        df_labels = pd.concat(
            [df.loc[valid_idx, id_cols].reset_index(drop=True),
             df_labels.reset_index(drop=True)],
            axis=1
        )
    elif DECOMPOSTION_METHOD == 'umapdb':
        # %% Phenotype UMAP + HDBSCAN
        results, umap_model, hdb, gmm = vessel_umap_composite(
            df=df,
            features=col_features,
            img_path_col="img_path"
        )

        interactive_umap_viewer(results)


    def interactive_umap_viewer(results, img_path_col="img_path", image_size=128):
        """
        Interactive UMAP visualization:
          - Left: UMAP scatter with Composite Score
          - Right: dynamically updated image on hover
        """
        umap1 = results["UMAP1"].values
        umap2 = results["UMAP2"].values
        scores = results["CompositeScore"].values
        img_paths = results[img_path_col].values
        # ------------------------------------------------------------------
        # Setup figure with 2 axes
        # ------------------------------------------------------------------
        fig, (ax_scatter, ax_image) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("Interactive UMAP Viewer (Hover a Point to See Image)", fontsize=14)
        # ------------------------------------------------------------------
        # Scatter plot with colorbar
        # ------------------------------------------------------------------
        sc = ax_scatter.scatter(umap1, umap2, c=scores, cmap="plasma", s=40, alpha=0.8)
        ax_scatter.set_title("UMAP Embedding")
        ax_scatter.set_xlabel("UMAP1")
        ax_scatter.set_ylabel("UMAP2")
        cbar = plt.colorbar(sc, ax=ax_scatter)
        cbar.set_label("Composite Score")
        # ------------------------------------------------------------------
        # Initial placeholder image
        # ------------------------------------------------------------------
        ax_image.set_title("Image Preview")
        ax_image.set_xticks([])
        ax_image.set_yticks([])
        placeholder = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        img_disp = ax_image.imshow(placeholder)
        # ------------------------------------------------------------------
        # Highlight circle
        # ------------------------------------------------------------------
        highlight, = ax_scatter.plot([], [], "o", ms=18, mec="yellow", mfc="none", mew=2)

        # ------------------------------------------------------------------
        # Hover event function
        # ------------------------------------------------------------------
        def on_move(event):
            if event.inaxes != ax_scatter:
                highlight.set_data([], [])
                return
            # Find nearest point
            x, y = event.xdata, event.ydata
            dists = np.sqrt((umap1 - x) ** 2 + (umap2 - y) ** 2)
            idx = np.argmin(dists)
            # Only update if the cursor is close enough
            if dists[idx] < 0.3:  # adjust sensitivity
                highlight.set_data([umap1[idx]], [umap2[idx]])
                img = cv2.imread(img_paths[idx])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (image_size, image_size))
                    img_disp.set_data(img)
                    ax_image.set_title(f"Composite Score: {scores[idx]:.3f}")
            else:
                highlight.set_data([], [])
            fig.canvas.draw_idle()

        # ------------------------------------------------------------------
        # Connect hover event
        # ------------------------------------------------------------------
        fig.canvas.mpl_connect("motion_notify_event", on_move)
        plt.show()



    # %%
    pca = PCA(n_components=0.9)
    X = df_features.replace([np.inf, -np.inf], np.nan).dropna()
    X_scaled = StandardScaler().fit_transform(X)

    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(n_components=3, covariance_type='full')
    labels = gmm.fit_predict(X_pca)
    probs = gmm.predict_proba(X_pca)

    embedding = UMAP(n_neighbors=30, min_dist=0.1).fit_transform(X_scaled)

    gmm = GaussianMixture(n_components=3)
    labels = gmm.fit_predict(embedding)



    # %% Plot the clusters with the PCAs and show the image
    def get_cluster_representatives(df_labels:pd.DataFrame,
                                    X_pca:pd.DataFrame,
                                    kmeans, n_samples=3):
        """
        Select n representative cells per cluster based on proximity to cluster centroids.
        """
        cluster_ids = df_labels["cluster"].values
        centroids = kmeans.cluster_centers_

        reps = []
        for c in np.unique(cluster_ids):
            idx = np.where(cluster_ids == c)[0]
            # compute distances in PCA space
            distances = np.linalg.norm(X_pca[idx] - centroids[c], axis=1)
            top_idx = idx[np.argsort(distances)[:n_samples]]
            reps.append(df_labels.iloc[top_idx])

        return pd.concat(reps)


    df_reps = get_cluster_representatives(df_labels=df_labels,
                                          X_pca=X_pca,
                                # features=col_features,
                                kmeans=kmeans,
                                n_samples=3)


    def plot_clusters_with_representatives(df_labels:pd.DataFrame,
                                           X_pca:pd.DataFrame,
                                           reps_df:pd.DataFrame=None, title="Cluster Visualization (PCA space)"):
        """
        Visualize unsupervised clusters in PCA space, optionally highlighting representative samples.

        Parameters
        ----------
        df_labels : pd.DataFrame
            DataFrame returned by unsupervised_overlap_classifier(), must contain 'cluster' column.
        X_pca : np.ndarray
            PCA-transformed feature matrix used in clustering (shape = n_samples × 2+).
        reps_df : pd.DataFrame, optional
            Representative samples (subset of df_labels).
        title : str, optional
            Plot title.
        """
        if X_pca.shape[1] > 2:
            # Reduce to 2D for visualization
            from sklearn.decomposition import PCA
            X_vis = PCA(n_components=2).fit_transform(X_pca)
        else:
            X_vis = X_pca

        clusters = df_labels["cluster"].values
        plt.figure(figsize=(7, 6))
        palette = sns.color_palette("Set2", len(np.unique(clusters)))

        # Plot all samples
        for c, color in zip(np.unique(clusters), palette):
            mask = clusters == c
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1],
                        s=40, alpha=0.7, label=f"Cluster {c}", color=color, edgecolor="none")

        # Highlight representative samples
        if reps_df is not None:
            rep_mask = df_labels.index.isin(reps_df.index)
            plt.scatter(X_vis[rep_mask, 0], X_vis[rep_mask, 1],
                        s=150, edgecolor="black", facecolor="none", linewidth=1.5, label="Representatives")

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()


    plot_clusters_with_representatives(df_labels, X_pca, df_reps)


    def visualize_clusters_biplot_heatmap(
            df_labels: pd.DataFrame,
            X_pca: np.ndarray,
            pca: PCA,
            cluster_col: str = "cluster",
            n_top_features: int = 20,
            title_prefix: str = "Cluster Visualization"
    ) -> pd.DataFrame:
        """
        Visualize PCA clusters (biplot + normalized heatmap) and return full normalized cluster table.

        Returns
        -------
        pd.DataFrame
            Z-scored mean of every feature across clusters (rows = features, columns = clusters).
        """
        # ---------- Select numeric features ----------
        clusters = df_labels[cluster_col].values
        feature_cols = [
            c for c in df_labels.select_dtypes(include=[np.number]).columns
            if c != cluster_col
        ]
        n_clusters = len(np.unique(clusters))
        palette = sns.color_palette("Set2", n_clusters)

        # ---------- Figure setup ----------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
        ax1, ax2 = axes

        # ---------- (A) PCA biplot ----------
        for c, color in zip(np.unique(clusters), palette):
            mask = clusters == c
            ax1.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                s=45,
                label=f"Cluster {c}",
                color=color,
                alpha=0.75,
                edgecolor="none",
            )

        loadings = pca.components_.T[:, :2]
        scaling = np.max(np.abs(X_pca[:, :2])) / np.max(np.abs(loadings))
        variances = np.var(df_labels[feature_cols], axis=0)
        top_idx = np.argsort(variances)[-n_top_features:]

        for i in top_idx:
            f = feature_cols[i]
            ax1.arrow(
                0,
                0,
                loadings[i, 0] * scaling * 0.8,
                loadings[i, 1] * scaling * 0.8,
                color="gray",
                alpha=0.4,
                head_width=0.03 * scaling,
            )
            ax1.text(
                loadings[i, 0] * scaling * 0.85,
                loadings[i, 1] * scaling * 0.85,
                f,
                fontsize=7,
                color="dimgray",
            )

        ax1.set_xlabel("PCA 1")
        ax1.set_ylabel("PCA 2")
        ax1.set_title(f"{title_prefix}: PCA Biplot")
        ax1.legend(frameon=False)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # ---------- (B) Cluster feature means + z-score normalization ----------
        cluster_means = df_labels.groupby(cluster_col)[feature_cols].mean().T
        cluster_means_norm = cluster_means.apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1
        )

        # select top features for visualization only
        top_feats = (
            cluster_means_norm.var(axis=1)
            .sort_values(ascending=False)
            .head(n_top_features)
            .index
        )
        cluster_means_top = cluster_means_norm.loc[top_feats]

        sns.heatmap(
            cluster_means_top,
            cmap="RdBu_r",
            center=0,
            linewidths=0.3,
            linecolor="gray",
            cbar_kws={"label": "Z-scored Mean per Feature"},
            ax=ax2,
        )
        ax2.set_title(f"{title_prefix}: Top Feature Means per Cluster")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Feature")

        plt.tight_layout()
        plt.show()

        # ---------- Return full normalized table ----------
        return cluster_means_norm



    cluster_means_norm = visualize_clusters_biplot_heatmap(
        df_labels=df_labels,
        X_pca=X_pca,
        pca=pca,
        cluster_col="cluster",
        n_top_features=20
    )


    def plot_cluster_feature_differences(
            df_labels: pd.DataFrame,
            cluster_col: str = "cluster",
            min_mean_diff: float = 0.05,
            top_n: int = 20,
            figsize: tuple = (14, 6),
            palette: str = "Set2"
    ):
        """
        Boxplot features whose mean values differ across clusters.

        Parameters
        ----------
        df_labels : pd.DataFrame
            DataFrame containing numeric features and cluster labels.
        cluster_col : str, default 'cluster'
            Column with cluster IDs.
        min_mean_diff : float, default 0.05
            Minimum absolute difference between cluster means to include a feature.
            Features with smaller inter-cluster mean differences are excluded.
        top_n : int, default 20
            Show only top N most variable features for readability.
        figsize : tuple, default (14, 6)
            Size of the boxplot figure.
        palette : str, default 'Set2'
            Seaborn color palette.
        """
        # ---- select numeric features ----

        # ---- compute mean per cluster ----
        cluster_means = df_labels.groupby(cluster_col)[numeric_cols].mean()
        mean_diff = cluster_means.max() - cluster_means.min()

        # ---- filter features with notable difference ----
        diff_features = mean_diff[mean_diff > min_mean_diff].sort_values(ascending=False)
        selected_features = diff_features.head(top_n).index.tolist()

        if not selected_features:
            print("⚠️ No features exceed the mean-difference threshold.")
            return

        # ---- reshape for plotting ----
        df_melt = df_labels.melt(
            id_vars=[cluster_col],
            value_vars=selected_features,
            var_name="feature",
            value_name="value"
        )

        # ---- plotting ----
        plt.figure(figsize=figsize)
        sns.boxplot(
            data=df_melt,
            x="feature",
            y="value",
            hue=cluster_col,
            palette=palette,
            showfliers=False
        )
        plt.xticks(rotation=90)
        plt.title(f"Top {len(selected_features)} Features with Mean Difference > {min_mean_diff}")
        plt.tight_layout()
        plt.show()

        # ---- return summary table ----
        summary = pd.DataFrame({
            "mean_diff": mean_diff[selected_features],
            "var_across_clusters": cluster_means[selected_features].var()
        }).sort_values("mean_diff", ascending=False)
        return summary


    summary_table = plot_cluster_feature_differences(
        df_labels=df_labels,
        cluster_col="cluster",
        min_mean_diff=0.1,  # threshold for inclusion
        top_n=30,  # plot top 30 varying features
        figsize=(16, 6)
    )










