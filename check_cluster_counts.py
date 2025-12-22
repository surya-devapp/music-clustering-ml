
import pandas as pd

def check_counts():
    try:
        df = pd.read_csv("clustered_songs.csv")
        print("Dataset Shape:", df.shape)
        
        if 'cluster' in df.columns:
            print("\nK-Means Cluster Counts:")
            print(df['cluster'].value_counts().sort_index())
            

            
    except FileNotFoundError:
        print("clustered_songs.csv not found.")

if __name__ == "__main__":
    check_counts()
