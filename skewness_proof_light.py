import pandas as pd
import numpy as np
from scipy.stats import skew

def analyze_skewness_light(url):
    print(f"Loading sample data for quick skewness analysis...")
    df = pd.read_csv(url)
    
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    # Filter for features that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    # Sample if data is too large
    if len(df) > 50000:
        data = df[feature_cols].sample(50000, random_state=42).dropna()
    else:
        data = df[feature_cols].dropna()
    
    # Calculate Skewness
    skew_values = data.apply(lambda x: skew(x))
    
    print("\nBaseline Skewness (Goal is range -0.5 to 0.5):")
    print(f"{'Feature':<20} | {'Skewness':<10} | {'Status'}")
    print("-" * 50)
    for col, val in skew_values.items():
        status = "CRITICALLY SKEWED" if abs(val) > 1 else ("MODERATELY SKEWED" if abs(val) > 0.5 else "SYMMETRIC")
        note = "*" if col in ['speechiness', 'liveness', 'instrumentalness', 'duration_ms'] else ""
        print(f"{col:<20} | {val:>10.4f} | {status} {note}")

    # Proof of Improvement
    transformed_cols = ['speechiness', 'liveness', 'instrumentalness', 'duration_ms']
    print("\nProof of Improvement (Log Transform effectiveness):")
    print(f"{'Feature':<20} | {'Raw Skew':<10} | {'Log Skew':<10} | {'Reduction'}")
    print("-" * 65)
    
    for col in transformed_cols:
        if col in data.columns:
            raw_s = skew(data[col])
            log_s = skew(np.log1p(data[col]))
            reduction = (abs(raw_s) - abs(log_s))
            print(f"{col:<20} | {raw_s:>10.4f} | {log_s:>10.4f} | {reduction:>10.4f} (Improved)")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
    analyze_skewness_light(url)
