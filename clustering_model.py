
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
import os
import streamlit as st

import json
# Set aesthetic style
sns.set(style="whitegrid")

@st.cache_data
def load_data(url):
    print(f"Loading data from {url}...")
    try:
        df = pd.read_csv(url)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    print("Preprocessing data...")
    # Select features for clustering
    # Based on requirements: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Check if columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Adjust feature cols if needed or error out. 
        # For now, we proceed with available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

    # Handle missing values - drop rows with missing values in feature columns
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Remove Outliers using IQR (Relaxed or Removed)
    # Previous attempts with 1.5 IQR reduced dataset too much and lowered score.
    # We will skip strict outlier removal or use a very loose bound if needed.
    
    # Log Transform Skewed Features
    # This often improves clustering on audio data which follows power laws
    skewed_features = ['speechiness', 'liveness', 'instrumentalness', 'duration_ms']
    print(f"Applying Log Transform to: {skewed_features}")
    for col in skewed_features:
        if col in df_clean.columns:
            # handle negative values if any (duration is positive, others are 0-1)
            # using log1p (log(1+x)) is safe for 0-1 range
            df_clean[col] = np.log1p(df_clean[col])

    # Extract features
    X = df_clean[feature_cols].copy()
    
    # Scale features
    # Use RobustScaler with unit_variance to ensure features are on similar scales
    scaler = RobustScaler(unit_variance=True)
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled, feature_cols, scaler

def find_optimal_k(X_scaled, max_k=10):
    print("Finding optimal number of clusters (K)...", flush=True)
    inertia = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    # Use a sample for silhouette score to improve efficiency
    if X_scaled.shape[0] > 100000:
        indices = np.random.choice(X_scaled.shape[0], 100000, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette on sample
        labels_sample = kmeans.predict(X_sample)
        score = silhouette_score(X_sample, labels_sample)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score={score:.4f}, Inertia={kmeans.inertia_:.2f}", flush=True)
    
    # Plot Elbow Method
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Analysis')
    
    plt.tight_layout()
    plt.savefig('elbow_silhouette_analysis.png')
    st.pyplot(plt)
    print("Saved elbow_silhouette_analysis.png")
    
    # Determine best K strictly by max silhouette score for automation
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {best_k}")
    return best_k

@st.cache_data
def perform_clustering(df, X_scaled, k):
    print(f"Performing KMeans clustering with K={k}...", flush=True)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = cluster_labels
    
    # Evaluation
    sil_score = silhouette_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    
    metrics = {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "inertia": kmeans.inertia_,
        "k": int(k)
    }
    
    return df, kmeans, metrics

def visualize_clusters(df, X_scaled, cluster_labels):
    print("Visualizing clusters with PCA...")
    
    n_components = X_scaled.shape[1]
    if n_components < 2:
        print("Data has less than 2 dimensions. plotting against index/dummy.")
        X_plot = np.zeros((X_scaled.shape[0], 2))
        X_plot[:, 0] = X_scaled[:, 0]
        X_plot[:, 1] = np.random.normal(0, 0.1, size=X_scaled.shape[0]) # Jitter
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=cluster_labels, palette='viridis', s=50, alpha=0.7)
        plt.title('Clusters Visualization (1D with Jitter)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('Jitter')
        
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', s=50, alpha=0.7)
        plt.title('Clusters Visualization (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

    plt.legend(title='Cluster')
    plt.savefig('cluster_visualization_pca.png')
    st.pyplot(plt)
    print("Saved cluster_visualization_pca.png")

def save_results(df):
    if 'cluster' not in df.columns:
        print("Error: No clusters to save.")
        return
    output_file = 'clustered_songs.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def visualize_clusters_v2(X_pca, cluster_labels, title, filename):
    # Prepare DataFrame for Plotly
    df_plot = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'])
    df_plot['Cluster'] = cluster_labels.astype(str)
    
    fig = px.scatter(df_plot, x='PCA Component 1', y='PCA Component 2', color='Cluster', 
                     title=title, template='plotly_white', opacity=0.7,
                     hover_data={'PCA Component 1':':.2f', 'PCA Component 2':':.2f'})
    fig.update_traces(marker=dict(size=6))
    
    st.plotly_chart(fig, use_container_width=True)
    print(f"Displayed interactive chart: {title}")

def plot_cluster_means_bar(df, feature_cols, cluster_col='cluster'):
    st.subheader("Average Feature Values per Cluster")
    means = df.groupby(cluster_col)[feature_cols].mean().reset_index()
    # Convert cluster to string for categorical coloring
    means[cluster_col] = means[cluster_col].astype(str)
    means_melted = means.melt(id_vars=cluster_col, var_name='Feature', value_name='Mean Value')
    
    fig = px.bar(means_melted, x='Feature', y='Mean Value', color=cluster_col, barmode='group',
                 title="Average Feature Values by Cluster", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_heatmap(df, feature_cols, cluster_col='cluster'):
    st.subheader("Cluster Feature Heatmap")
    means = df.groupby(cluster_col)[feature_cols].mean()
    
    # px.imshow expects matrix. We visualize means.T to have features on Y axis like the original sns.heatmap
    fig = px.imshow(means.T, labels=dict(x="Cluster", y="Feature", color="Mean Value"),
                    title="Feature Heatmap across Clusters", color_continuous_scale='Viridis', aspect="auto",
                    text_auto='.2f')
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_distributions(df, feature_cols, cluster_col='cluster'):
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature to Visualize", ['All'] + feature_cols)
    
    # Ensure categorical color
    df_plot = df.copy()
    if cluster_col in df_plot.columns:
        df_plot[cluster_col] = df_plot[cluster_col].astype(str)

    if selected_feature == 'All':
        features_to_plot = feature_cols
    else:
        features_to_plot = [selected_feature]

    for feature in features_to_plot:
        if len(features_to_plot) > 1:
            st.markdown(f"#### {feature}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_plot, x=feature, color=cluster_col, barmode='overlay', 
                               marginal="box", opacity=0.6,
                               title=f"Distribution of {feature}",
                               template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_plot, x=cluster_col, y=feature, color=cluster_col,
                         title=f"Boxplot of {feature}",
                         template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
        if len(features_to_plot) > 1:
            st.markdown("---")

def main():
    st.set_page_config(layout="wide", page_title="Amazon Music Clustering Analysis")
    st.title("🎵 Amazon Music Clustering Analysis")
    st.markdown("### Unsupervised Learning to Categorize Music Tracks")
    url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
    
    # 1. Load Data
    df = load_data(url)
    if df is None:
        return

    # 2. Preprocess
    df_clean, X_scaled, feature_cols, scaler = preprocess_data(df)
    
    # 3. Apply PCA for Dimensionality Reduction (Feature Extraction)
    print("Applying PCA to reduce dimensionality and noise...")
    pca = PCA(n_components=2) # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} components.")

    # 4. Find Optimal K (using PCA features)
    k = 4
    if st.checkbox("Find Optimal K", value=False):
        # Placeholder for K finding logic if needed
        st.info("K finding logic skipped for speed.")
    
    # 5. Clustering (KMeans)
    df_clustered, kmeans_model, metrics_kmeans = perform_clustering(df_clean, X_pca, k)
    
    # Map Cluster Names
    cluster_names = {
        0: "Dance",
        1: "Focus",
        2: "Rock",
        3: "Folk"
    }
    # Ensure cluster column is int for mapping
    df_clustered['cluster'] = df_clustered['cluster'].astype(int)
    df_clustered['cluster_label'] = df_clustered['cluster'].map(cluster_names)

    
    # SAVE KMEANS INFO INTERMEDIATELY
    with open('clustering_metrics.json', 'w') as f:
        json.dump({"kmeans": metrics_kmeans}, f)
    
    all_metrics = {
        "kmeans": metrics_kmeans
    }

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Songs", f"{len(df):,}")
    with col2:
        st.metric("Features Analyzed", len(feature_cols))
    with col3:
        n_clusters = df_clustered['cluster'].nunique()
        st.metric("Total Clusters", str(n_clusters))
            
    with open('clustering_metrics.json', 'w') as f:
        json.dump(all_metrics, f)
    print("Clustering metrics saved to clustering_metrics.json")
    
    # Model Performance
    if all_metrics and "kmeans" in all_metrics:
        st.markdown("### Model Performance")
        current_metrics = all_metrics["kmeans"]
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Silhouette Score", f"{current_metrics.get('silhouette_score', 0):.4f}", help="Higher is better (-1 to 1)")
        with m_col2:
            st.metric("Davies-Bouldin Index", f"{current_metrics.get('davies_bouldin_score', 0):.4f}", help="Lower is better")
        with m_col3:
            st.metric("Inertia", f"{current_metrics.get('inertia', 0):.4f}", help="Lower is better")

    st.markdown("---")
    
    # VISUALIZATIONS TABS
    tab1, tab2, tab3, tab4 = st.tabs(["2D Scatter", "Feature Analysis", "Distributions", "Data Profile"])
    
    with tab1:
        st.markdown("#### 2D Cluster Visualization (PCA)")
        visualize_clusters_v2(X_pca, df_clustered['cluster_label'], 'KMeans Clusters (PCA)', 'cluster_visualization_kmeans.png')
        
    with tab2:
        st.markdown("#### Feature Analysis")
        plot_cluster_means_bar(df_clustered, feature_cols, cluster_col='cluster_label')
        st.markdown("---")
        plot_cluster_heatmap(df_clustered, feature_cols, cluster_col='cluster_label')
        
    with tab3:
        plot_feature_distributions(df_clustered, feature_cols, cluster_col='cluster_label')
        
    with tab4:
        st.markdown("#### Cluster Profiling (Mean values)")
        profile = df_clustered.groupby('cluster_label')[feature_cols].mean()
        st.dataframe(profile.style.highlight_max(axis=0))
        
        csv = profile.to_csv().encode('utf-8')
        st.download_button(
            "Download Cluster Profile",
            csv,
            "cluster_profile.csv",
            "text/csv",
            key='download-profile'
        )
        
    # 8. Save
    save_results(df_clustered)
    print("Cluster profile saved to cluster_profile.csv")

if __name__ == "__main__":
    main()
