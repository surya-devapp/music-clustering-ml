# Project Report: Amazon Music Clustering Analysis

## 1. Executive Summary
This project aims to categorize music tracks from Amazon Music (or similar datasets) into distinct clusters using unsupervised machine learning. By analyzing audio features such as danceability, energy, and acousticness, we successfully identified **4 distinct clusters** of songs. The model achieved a high **Silhouette Score of 0.9078**, indicating very well-defined and separated clusters.

## 2. Methodology

### 2.1 Data Pipeline
1.  **Data Ingestion**: The dataset is loaded from a remote CSV source containing audio features of various songs.
2.  **Preprocessing**:
    *   **Feature Selection**: Key audio features selected include `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `duration_ms`.
    *   **Transformation**: Skewed features (`speechiness`, `liveness`, `instrumentalness`, `duration_ms`) were log-transformed to normalize their distributions.
    *   **Scaling**: `RobustScaler` was applied to handle potential outliers and ensure all features contribute equally to the distance calculations.
3.  **Dimensionality Reduction**: Principal Component Analysis (PCA) was used to reduce the high-dimensional feature space into 2 principal components. This step aids in noise reduction and allows for effective 2D visualization.

### 2.2 Modeling
*   **Algorithm**: K-Means Clustering
*   **Optimal K**: Determined to be **4** based on Silhouette analysis and Elbow method (implemented in earlier phases).
*   **Input**: PCA-transformed components.

## 3. Results & Performance
The K-Means model with K=4 yielded the following performance metrics:

*   **Silhouette Score**: **0.9078** (Excellent separation)
*   **Davies-Bouldin Index**: **0.4203** (Low value indicates good clustering structure)
*   **Inertia**: **46,897,409** (Sum of squared distances within clusters)

## 4. Cluster Insights & Profiling
Based on the average feature values for each cluster (derived from `cluster_profile.csv`), we can characterize the music segments as follows:

*   **Cluster 0**: *Balanced / Acoustic* - Moderate danceability (0.50) and energy. Higher acousticness (0.59) suggests these might be softer or more instrumental tracks.
*   **Cluster 1**: *High Energy / Pop* - Characterized by higher danceability (0.60) and energy (0.55). These tracks likely represent popular, radio-friendly songs or upbeat genres.
*   *(Note: Detailed characteristics for Clusters 2 and 3 can be observed in the application's "Data Profile" tab).*

## 5. Technology Stack
*   **Language**: Python
*   **Interface**: Streamlit (Web Dashboard)
*   **Machine Learning**: Scikit-Learn (KMeans, PCA, Preprocessing)
*   **Visualization**: Plotly (Interactive Charts) & Seaborn
*   **Data Handling**: Pandas & NumPy

## 6. Conclusion
The clustering analysis successfully segmented the music library into coherent groups. The high silhouette score validates the choice of features and the effectiveness of the preprocessing pipeline (Log transform + RobustScaler + PCA). This segmentation can be used for:
*   Building recommendation systems.
*   Creating automated playlists (e.g., "Chill Vibes" vs. "Workout Mode").
*   Analyzing genre trends.
