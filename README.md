# Amazon Music Clustering Analysis Dashboard

An interactive unsupervised machine learning application to categorize music tracks based on audio features. This project leverages K-Means clustering to distinguish between various music segments such as "Dance", "Focus", "Rock", and "Folk".

## 🚀 Key Features

### 1. Intelligent Preprocessing
- **Log Transformation**: Corrects skewed features (`speechiness`, `liveness`, `instrumentalness`, `duration_ms`) for better model performance.
- **Robust Scaling**: Handles outliers using Interquartile Range (IQR) to ensure stable distance calculations.
- **PCA Dimensionality Reduction**: Reduces 10 audio features into 2 principal components for noise reduction and visualization.

### 2. Model Optimization
- **Automated K-Selection**: Interactive **Elbow Method** and **Silhouette Analysis** help determine the mathematically optimal number of clusters.
- **K-Means Clustering**: Core algorithm used to assign songs to distinct groups.

### 3. Interactive Insights Dashboard
- **2D Cluster Visualization**: Explorable PCA scatter plot.
- **Feature Analysis**: 
  - **Mean Bar Charts**: Compare characteristics across clusters.
  - **Correlation Heatmaps**: Understand relationships between audio features.
- **Data Profiles**: Detailed statistical summary for each cluster.
- **Cluster Insights Tab**:
  - **Top Tracks**: Identify the 5 most representative songs (closest to centroid) for each cluster.
  - **Summary Report**: Auto-generated text descriptions of each cluster's unique traits.

### 4. Data & Model Export
- **Export Clustered Data**: Download the full dataset with assigned labels as a CSV.
- **Export Model**: Save the trained KMeans model as a `.pkl` file for future inference.
- **Export Summary**: Download the textual cluster characteristics as a `.txt` report.

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone [repository-url]
    cd music-clustering-ml
    ```

2.  **Install Dependencies**:
    You can install all necessary packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    # For the main Clustering Dashboard:
    python -m streamlit run clustering_model.py
    ```

## 📊 Datasets
The application dynamically loads the artist audio features from a cloud-hosted dataset containing features like danceability, energy, tempo, and more.

## 📝 Troubleshooting
- **Port Issues**: If the local URL does not load, verify that port `8501` is open or check the terminal output for the correct address.
- **Permissions**: Ensure write access to the project directory for saving results and intermediate files.
