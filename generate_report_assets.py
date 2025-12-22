
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# Set aesthetic style
sns.set(style="whitegrid")

def load_and_preprocess():
    url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
    print(f"Loading data from {url}...")
    df = pd.read_csv(url)
    
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Missing cols check skipped for brevity, assuming standard dataset
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Log Transform
    skewed_features = ['speechiness', 'liveness', 'instrumentalness', 'duration_ms']
    for col in skewed_features:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    X = df_clean[feature_cols].copy()
    scaler = RobustScaler(unit_variance=True)
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled, feature_cols

def generate_assets():
    df, X_scaled, feature_cols = load_and_preprocess()
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans K=4
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = cluster_labels
    cluster_names = {
        0: "Dance",
        1: "Focus",
        2: "Rock",
        3: "Folk"
    }
    df['cluster_label'] = df['cluster'].map(cluster_names)
    
    # 1. PCA Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster_label'], palette='viridis', s=50, alpha=0.7)
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.savefig('cluster_visualization_pca.png')
    plt.close()
    print("Saved cluster_visualization_pca.png")
    
    # 2. Heatmap
    plt.figure(figsize=(12, 6))
    means = df.groupby('cluster_label')[feature_cols].mean()
    sns.heatmap(means.T, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Feature Heatmap across Clusters')
    plt.tight_layout()
    plt.savefig('cluster_feature_heatmap.png')
    plt.close()
    print("Saved cluster_feature_heatmap.png")
    
    # 3. Bar Chart (Means)
    means_melted = means.reset_index().melt(id_vars='cluster_label', var_name='Feature', value_name='Mean Value')
    plt.figure(figsize=(14, 7))
    sns.barplot(data=means_melted, x='Feature', y='Mean Value', hue='cluster_label', palette='viridis')
    plt.title("Average Feature Values by Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cluster_feature_barchart.png')
    plt.close()
    print("Saved cluster_feature_barchart.png")

    # 4. Save Profile CSV
    means.to_csv('cluster_profile.csv')
    print("Saved cluster_profile.csv")

if __name__ == "__main__":
    generate_assets()
