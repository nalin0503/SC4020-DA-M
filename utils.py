"""
This script currently contains functions to load a common dataset - `load_dataset`, 
visualise clustering results - `visualize_clustering_results` and 
evaluate the clustering results - `evaluate_clustering`
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, silhouette_score
from matplotlib import cm

# Function to load dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    X = df.drop('Label', axis=1).values
    y_true = df['Label'].values  # For evaluation purposes
    return X, y_true

def visualize_clustering_results(X, labels, y_true=None, title='', reduce_dim=False):
    """
    Common function to visualize clustering results.
    
    Parameters:
    - X: np.ndarray
        The original feature matrix.
    - labels: np.ndarray
        The predicted cluster labels from the clustering algorithm.
    - y_true: np.ndarray, optional
        The ground truth labels, if available. Default is None.
    - title: str
        The title for the plot.
    - reduce_dim: bool
        Whether to apply PCA to reduce dimensionality for visualization. Default is False.
    """
    
    # Standardize features for high-dimensional data visualization
    if reduce_dim and X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        print("Data will be reduced to 2D using PCA for visualization.")
    else:
        X_vis = X
    
    plt.figure(figsize=(12, 5))
    
    # Plot predicted clusters
    plt.subplot(1, 2, 1)
    unique_labels = sorted(set(labels))
    colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise in DBSCAN
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X_vis[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6, label=f'Cluster {k}')
    
    plt.title(f'Clustering Results: {title}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    # Plot ground truth, if provided
    if y_true is not None:
        plt.subplot(1, 2, 2)
        unique_labels_true = set(y_true)
        colors_true = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_true))]
        
        for k, col in zip(unique_labels_true, colors_true):
            class_member_mask = (y_true == k)
            xy = X_vis[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6, label=f'Class {k}')
        
        plt.title(f'Ground Truth: {title}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def evaluate_clustering(X_scaled, labels, y_true, n_clusters, title=''):
    """
    Evaluate clustering performance with AMI, V-measure, and Silhouette Score.
    
    Parameters:
    - X_scaled: np.ndarray
        Scaled features (after normalization or standardization).
    - labels: np.ndarray
        Predicted cluster labels from the clustering algorithm.
    - y_true: np.ndarray
        Ground truth labels.
    - n_clusters: int
        Number of predicted clusters.
    - title: str
        Title for the evaluation printout.
    
    Returns:
    - ami: float
        Adjusted Mutual Information.
    - v_measure: float
        V-measure.
    - silhouette_avg: float or None
        Silhouette score, or None if it cannot be computed.
    """
    # Adjusted Mutual Information and V-measure
    ami = adjusted_mutual_info_score(y_true, labels)
    v_measure = v_measure_score(y_true, labels)
    
    print(f"{title}")
    print(f"Adjusted Mutual Information: {ami:.4f}")
    print(f"V-measure: {v_measure:.4f}")
    
    # Silhouette Score (exclude noise points in comparison of pred vs ground truth clusters)
    if n_clusters > 1:
        labels_filtered = labels[labels != -1]
        X_filtered = X_scaled[labels != -1]
        if len(set(labels_filtered)) > 1:
            silhouette_avg = silhouette_score(X_filtered, labels_filtered)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        else:
            silhouette_avg = None
            print("Silhouette Score: Cannot be calculated with less than 2 clusters after removing noise.")
    else:
        silhouette_avg = None
        print("Silhouette Score: Cannot be calculated with less than 2 clusters.")

    print("=====================================================================")
    
    return ami, v_measure, silhouette_avg

