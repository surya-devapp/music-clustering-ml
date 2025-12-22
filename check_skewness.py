import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"

try:
    df = pd.read_csv(url)
    print("Data loaded successfully.")
    
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Filter for existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Calculate Skewness
    print("\nSkewness of features:")
    skewness = df[feature_cols].skew().sort_values(ascending=False)
    print(skewness)
    
    # Visualize Distributions
    print("\nGenerating histograms...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 4, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} (Skew: {df[col].skew():.2f})")
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    print("Saved feature_distributions.png")
    
except Exception as e:
    print(f"Error: {e}")
