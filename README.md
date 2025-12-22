# Amazon Music Clustering App

This project implements an unsupervised learning model to categorize music tracks using K-Means clustering. It features an interactive Streamlit dashboard for visualizing clusters, analyzing feature distributions, and profiling music segments.

## Prerequisites

- Python 3.8 or higher

## Installation

1.  **Clone or Download** this repository to your local machine.
2.  **Install the required Python packages**. You can install them directly using pip:

    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit
    ```

## How to Run

To start the dashboard, verify you are in the project directory (e.g., `d:\Clustering`) and run:

```bash
python -m streamlit run clustering_model.py
```

*Alternatively, you can use `streamlit run clustering_model.py`.*

## Application Features

The `clustering_model.py` script performs the full pipeline:
1.  **Data Loading**: Fetches dataset from the source.
2.  **Preprocessing**: Cleans data, handles missing values, and scales features.
3.  **Clustering**: Applies K-Means clustering to categorize songs.
4.  **Visualization**:
    *   **2D Scatter Plot**: PCA-reduced visualization of song clusters.
    *   **Feature Analysis**: Bar charts and heatmaps showing average feature values per cluster.
    *   **Distributions**: Interactive histograms and boxplots for all audio features.
    *   **Data Profile**: Detailed statistics and "Download" options for cluster profiles.

## Troubleshooting

-   If the app does not open automatically, copy the "Local URL" (e.g., `http://localhost:8501`) shown in the terminal and paste it into your browser.
-   **CSV Permissions**: Ensure the script has write permissions in the folder to save `clustered_songs.csv` and intermediate files.
