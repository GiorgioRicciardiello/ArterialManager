# ImgClassifier Library

This library provides an **unsupervised machine learning pipeline** for phenotyping and classifying vascular networks based on the quantitative features extracted by the `ImageOverlap` library. It bridges raw metrics with high-level biological interpretation using dimensionality reduction and clustering.

## 🚀 Key Features

*   **Automated Dataset Creation:** Converts long-format metric reports into ready-to-use ML feature tables.
*   **Unsupervised Phenotyping:**
    *   **Dimensionality Reduction:** PCA and UMAP to visualize high-dimensional metric spaces.
    *   **Clustering:** KMeans and HDBSCAN to identify distinct vessel phenotypes without prior labeling.
    *   **Composite Scoring:** Calculates a multi-parametric "Composite Score" combining structural, intensity, and informational metrics.
*   **Interactive Visualization:**
    *   **Dash App:** A web-based viewer to explore UMAP embeddings interactively—hover over data points to instantly see the corresponding original vessel image.
    *   **Static Plots:** PCA biplots, cluster heatmaps, and representative sample visualizations.

---

## 📂 File Structure

*   `OverlayClassifier.py`: **Core Analysis**. Runs the unsupervised learning pipelines (PCA+KMeans or UMAP+HDBSCAN). Generates static plots and analysis tables.
*   `CreateDataset.py`: **Data Prep**. helper functions (`make_ml_feature_table`) to aggregate Excel reports, find corresponding image paths recursively, and format data for ML.
*   `app_umap_image_viewer.py`: **Interactive Viewer**. A Dash application for exploring the UMAP projections with dynamic image previews.
*   `evaluate_cluster_selection.py`: Tools for evaluating the optimal number of clusters (Silhouette scores, Elbow method).
*   `stat_tests_features_outcome.py`: Script for running statistical tests to compare features across known biological conditions (outcomes).

---

## 🛠️ Workflow & Usage

### 1. Prepare the Dataset
First, aggregate the individual metrics from `ImageOverlap` into a single feature table.

```python
from library.ImgClassifier.CreateDataset import make_ml_feature_table
from config.config import CONFIG

df = make_ml_feature_table(
    path_in=CONFIG.get('paths')['data'] / "features_imgs/Imgfeatures.xlsx",
    path_out=CONFIG.get('paths')['data'] / "features_imgs/ImgfeaturesWide.xlsx",
    path_imgs=CONFIG.get('paths')['local_images_output'] / 'processed_overlap',
    overwrite=True
)
```

### 2. Run Unsupervised Classification
Use `OverlayClassifier.py` to run the analysis. You can choose between PCA+KMeans or UMAP+HDBSCAN.

```python
from library.ImgClassifier.OverlayClassifier import vessel_umap_composite

# Define features to use
features = ['Manders', 'Dice', 'structural_overlap'] 

# Run UMAP pipeline
results, umap_model, hdb, gmm = vessel_umap_composite(
    df=df,
    features=features,
    img_path_col="img_path"
)
```

### 3. Interactive Visualization
To launch the interactive viewer:

1.  Navigate to the directory:
    ```bash
    cd library/ImgClassifier
    ```
2.  Run the app:
    ```bash
    python app_umap_image_viewer.py
    ```
3.  Open the provided URL (usually `http://127.0.0.1:8050/`) in your browser.


### 4. Statistical test and visualization of the UMAP projections
We use the script `stat_tests_features_outcome.py` to compare the features across known biological conditions (outcomes).
The statistical tests focuses on a subset of different features 

---

## 📊 Outputs

*   **`results.csv`**: A CSV file containing the original metadata plus calculated UMAP coordinates, cluster labels, and composite scores.
*   **Static Plots**:
    *   PCA Biplots showing feature contributions.
    *   Heatmaps of feature distributions across clusters.
    *   Cluster visualization with representative sample highlights.
*   **Interactive Dashboard**: A browser-based tool to explore the data manifold visually.
