# README.txt

## Instructions for Running the Code

This repository contains Python code for performing clustering analysis on two datasets (`dataset1.csv` and `dataset2.csv`) using hierarchical clustering and k-means clustering algorithms. Below are the instructions to run the code and reproduce the results.

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.x installed.
2. **Required Libraries**: Install the necessary Python libraries using the following command:
   ```bash
   pip install numpy pandas matplotlib scikit-learn plotly scipy

# Dataset Preparation

**Dataset Files**: Place `dataset1.csv` and `dataset2.csv` in the appropriate directory as specified in the code. Update the `path_dataset1` and `path_dataset2` variables in the code to point to the correct file paths.

# Running the Code

## Hierarchical Clustering:
- The code performs hierarchical clustering on both datasets using single and average linkage methods.
- Dendrograms are plotted to help determine the optimal number of clusters.
- The clustering results are visualized using 2D and 3D scatter plots.

## K-Means Clustering:
- The code implements k-means clustering with two initialization methods: uniform random initialization and k-means++.
- The Sum of Squared Errors (SSE) is calculated for each clustering result.
- The optimal number of clusters is determined using the elbow method.
- Clustering results are visualized using 2D and 3D scatter plots.

## Interactive Plots:
- The code uses Plotly to create interactive 3D scatter plots for better visualization of clustering results.

# Code Attribution

- **Hierarchical Clustering**: The hierarchical clustering implementation uses `scipy.cluster.hierarchy` and `sklearn.cluster.AgglomerativeClustering`. The dendrogram plotting function is adapted from the official documentation of `scipy.cluster.hierarchy.dendrogram`.
- **K-Means Clustering**: The k-means clustering implementation is based on the standard Lloyd's algorithm. The k-means++ initialization method is implemented as described in the original k-means++ paper by Arthur and Vassilvitskii (2007).
- **Plotly**: Interactive 3D scatter plots are created using `plotly.express`.

# Modified Files

- **Hierarchical Clustering**: The `plot_dendrogram_and_decide_k` function was modified to include automatic determination of the optimal number of clusters using the elbow method.
- **K-Means Clustering**: The `kmeans` function was implemented from scratch, including the `uniform_rand_init` and `kmeans_plusplus_init` methods. The `calculate_sse` function was added to compute the Sum of Squared Errors.

# Running the Code in Jupyter Notebook
1. Open the Jupyter Notebook containing the code.
2. Run each cell sequentially to load the datasets, perform clustering, and visualize the results.
3. Adjust the k values and other parameters as needed to explore different clustering configurations.

# Output

- **2D Scatter Plots**: Visualizations of clustering results for `dataset1`.
- **3D Scatter Plots**: Interactive visualizations of clustering results for `dataset2`.
- **Dendrograms**: Plots to help determine the optimal number of clusters for hierarchical clustering.
- **SSE and Silhouette Scores**: Printed output of SSE and silhouette scores for different clustering configurations.

# Notes

- Ensure that the dataset files are correctly formatted as CSV files without headers.
- The code assumes that `dataset1` is a 2D dataset and `dataset2` is a 3D dataset. Adjust the code if your datasets have different dimensions.

For any questions or issues, please contact the repository owner.
