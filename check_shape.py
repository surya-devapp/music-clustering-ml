
import pandas as pd
url = "https://raw.githubusercontent.com/surya-devapp/datasets/main/projectdatasets/single_genre_artists.csv"
try:
    df = pd.read_csv(url)
    print(f"Shape: {df.shape}")
except Exception as e:
    print(e)
