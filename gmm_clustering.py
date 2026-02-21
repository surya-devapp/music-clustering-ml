import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io

# Set page config
st.set_page_config(layout="wide", page_title="Music GMM Clustering (No PCA)")

@st.cache_data
def load_and_preprocess(url):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

    # Feature Selection
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Handle missing cols
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Log Transform Skewed Features
    skewed_features = ['speechiness', 'liveness', 'instrumentalness', 'duration_ms']
    for col in skewed_features:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])
            
    # Scale Features (RobustScaler)
    X = df_clean[feature_cols].copy()
    scaler = RobustScaler(unit_variance=True)
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled, feature_cols

def fit_gmm_bagging(X, n_components, n_init=10):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        n_init=n_init, # Restarting/Bagging equivalent for stability
        random_state=42
    )
    gmm.fit(X)
    return gmm

def calculate_metrics(gmm, X):
    labels = gmm.predict(X)
    
    # AIC/BIC (Lower is better)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    
    # Silhouette (Higher is better) - define sample for speed if large
    if X.shape[0] > 20000:
        indices = np.random.choice(X.shape[0], 20000, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels

    sil_score = silhouette_score(X_sample, labels_sample)
    db_score = davies_bouldin_score(X_sample, labels_sample)
    
    return {
        "k": int(gmm.n_components),
        "AIC": aic,
        "BIC": bic,
        "Silhouette": sil_score,
        "Davies-Bouldin": db_score
    }

def main():
    st.title("🌌 Music GMM Clustering Analysis (No PCA)")
    st.markdown("""
    This application performs **Gaussian Mixture Model (GMM)** clustering on music audio features. 
    Unlike K-Means, GMM assumes data points are generated from a mixture of several Gaussian distributions, 
    making it more flexible for complex data shapes.
    """)
    
    url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
    
    with st.sidebar:
        st.header("Settings")
        n_init = st.slider("n_init (Bagging Restarts)", 5, 50, 20, help="Number of initializations to run for stability.")
        max_k = st.slider("Max K to test", 2, 10, 6)
        run_analysis = st.button("Run Full GMM Analysis")

    df, X, feature_cols = load_and_preprocess(url)
    
    if df is None:
        return

    if run_analysis:
        results = []
        progress_bar = st.progress(0)
        
        K_range = range(2, max_k + 1)
        for i, k in enumerate(K_range):
            st.write(f"Evaluating K={k}...")
            gmm = fit_gmm_bagging(X, k, n_init=n_init)
            metrics = calculate_metrics(gmm, X)
            results.append(metrics)
            progress_bar.progress((i + 1) / len(K_range))
        
        results_df = pd.DataFrame(results)
        
        # Display Results Table
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(results_df.style.highlight_min(subset=['AIC', 'BIC', 'Davies-Bouldin'], color='#d4edda')
                                   .highlight_max(subset=['Silhouette'], color='#d4edda'))
        
        # Visualization: AIC/BIC
        st.subheader("📈 Information Criteria (AIC/BIC)")
        st.markdown("*Lower is better. BIC is typically more conservative.*")
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(x=results_df['k'], y=results_df['AIC'], mode='lines+markers', name='AIC'))
        fig_ic.add_trace(go.Scatter(x=results_df['k'], y=results_df['BIC'], mode='lines+markers', name='BIC'))
        fig_ic.update_layout(title="AIC & BIC vs Number of Clusters", xaxis_title="K", yaxis_title="Score", template="plotly_white")
        st.plotly_chart(fig_ic, use_container_width=True)
        
        # Visualization: Silhouette/DB
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Silhouette Score")
            fig_sil = px.line(results_df, x='k', y='Silhouette', markers=True, title="Higher is Better")
            fig_sil.update_layout(template="plotly_white")
            st.plotly_chart(fig_sil, use_container_width=True)
        with col2:
            st.subheader("Davies-Bouldin Index")
            fig_db = px.line(results_df, x='k', y='Davies-Bouldin', markers=True, title="Lower is Better")
            fig_db.update_layout(template="plotly_white")
            st.plotly_chart(fig_db, use_container_width=True)
            
        # Best Model
        best_k = results_df.loc[results_df['BIC'].idxmin()]['k']
        st.success(f"**Recommendation Based on BIC:** Optimal cluster count is **{int(best_k)}**.")
        
        # Final Run with Best K to show labels
        st.divider()
        st.header(f"✨ Best GMM Model (K={int(best_k)})")
        best_gmm = fit_gmm_bagging(X, int(best_k), n_init=n_init)
        df['gmm_cluster'] = best_gmm.predict(X)
        
        # Profile Plot
        means = df.groupby('gmm_cluster')[feature_cols].mean().reset_index()
        means_melted = means.melt(id_vars='gmm_cluster', var_name='Feature', value_name='Mean Value')
        fig_profile = px.bar(means_melted, x='Feature', y='Mean Value', color=means_melted['gmm_cluster'].astype(str), 
                             barmode='group', title="Cluster Feature Profiles (Centroids)")
        st.plotly_chart(fig_profile, use_container_width=True)
        
        # Downloads
        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered Data (CSV)", csv, "gmm_clustered_songs.csv", "text/csv")
        with col_dl2:
            metrics_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Metrics Summary (CSV)", metrics_csv, "gmm_metrics.csv", "text/csv")

if __name__ == "__main__":
    main()
