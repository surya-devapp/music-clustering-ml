import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def analyze_skewness(url):
    print(f"Loading data for skewness analysis...")
    df = pd.read_csv(url)
    
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Filter for features that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    data = df[feature_cols].dropna()
    
    # Calculate Skewness
    skew_values = data.apply(lambda x: skew(x))
    
    print("\nBaseline Skewness (Higher absolute value = more skewed):")
    for col, val in skew_values.items():
        note = "<- LOG TRANSFORMED" if col in ['speechiness', 'liveness', 'instrumentalness', 'duration_ms'] else ""
        print(f"{col:<20}: {val:>8.4f} {note}")

    # Plotting histograms to visualize skew
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 4, i+1)
        sns.histplot(data[col], kde=True)
        plt.title(f"{col}\nSkew: {skew_values[col]:.2f}")
    
    plt.tight_layout()
    plt.savefig('raw_feature_distributions.png')
    print("\nSaved distribution plots to 'raw_feature_distributions.png'")

    # Proof of Improvement for Transformed Features
    transformed_cols = ['speechiness', 'liveness', 'instrumentalness', 'duration_ms']
    print("\nProof of Improvement (Skew reduction):")
    print(f"{'Feature':<20} | {'Raw Skew':<10} | {'Log Skew':<10} | {'Improvement'}")
    print("-" * 60)
    
    for col in transformed_cols:
        if col in data.columns:
            raw_s = skew(data[col])
            log_s = skew(np.log1p(data[col]))
            improvement = (abs(raw_s) - abs(log_s)) / abs(raw_s)
            print(f"{col:<20} | {raw_s:>10.4f} | {log_s:>10.4f} | {improvement:>10.2%}")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
    analyze_skewness(url)
