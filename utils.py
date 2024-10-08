import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, silhouette_score
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(filename:str):
    """
    Function to load the CSV dataset.
    """
    df = pd.read_csv(filename)
    X = df.drop('Label', axis=1).values
    y_true = df['Label'].values  # Ground truth labels for evaluation
    return X, y_true

def visualize_clustering_2D(X:np.ndarray, 
                            labels:np.ndarray, 
                            y_true:np.ndarray = None, 
                            title:str = ''):
    """
    Function to visualize clustering results in 2D.
    
    Parameters:
    - X: np.ndarray
        The input feature matrix (2D).
    - labels: np.ndarray
        Predicted cluster labels from clustering algorithm
    - y_true: np.ndarray, optional
        Ground truth labels, if available. Default is None
    - title: str
        Title for the plots.
    """
    plt.figure(figsize=(12, 5))

    # Plot predicted clusters
    plt.subplot(1, 2, 1)
    unique_labels = np.unique(labels)
    colors = [cm.nipy_spectral(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50, label=f'Cluster {k}')

    plt.title(f'Clustering Results: {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Plot ground truth if available
    if y_true is not None:
        plt.subplot(1, 2, 2)
        unique_labels_true = np.unique(y_true)
        colors_true = [cm.nipy_spectral(float(i) / len(unique_labels_true)) for i in range(len(unique_labels_true))]

        for k, col in zip(unique_labels_true, colors_true):
            class_member_mask = (y_true == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50, label=f'Class {k}')

        plt.title(f'Ground Truth: {title}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_clustering_3D(X: np.ndarray, 
                            labels: np.ndarray, 
                            y_true: np.ndarray = None, 
                            title: str = ''):
    """
    Function to visualize clustering results in 3D.
    
    Parameters:
    - X: np.ndarray
        The input feature matrix (will be reduced to 3D if necessary).
    - labels: np.ndarray
        Predicted cluster labels from clustering algorithm
    - y_true: np.ndarray, optional
        Ground truth labels, if available. Default is None
    - title: str
        Title for the plots.
    """
    # Reduce data to 3D using PCA if necessary
    if X.shape[1] > 3:
        pca = PCA(n_components=3, random_state=42)
        X_vis = pca.fit_transform(X)
        print("Data has been reduced to 3D using PCA for visualization.")
    else:
        X_vis = X

    fig = plt.figure(figsize=(16, 8))

    # Adjust marker size here, increase as needed
    marker_size = 100  # ive fixed this at 100 TODO: fix visibility of 3D plot

    # Plot predicted clusters
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    unique_labels = np.unique(labels)
    colors = [cm.nipy_spectral(float(i) / len(unique_labels)) for i in range(len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xyz = X_vis[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], edgecolor='k', s=marker_size, alpha=0.7, label=f'Cluster {k}')  # Adjust alpha if needed

    ax.set_title(f'Clustering Results: {title}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    ax.grid(True)

    # Plot ground truth if available
    if y_true is not None:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        unique_labels_true = np.unique(y_true)
        colors_true = [cm.nipy_spectral(float(i) / len(unique_labels_true)) for i in range(len(unique_labels_true))]

        for k, col in zip(unique_labels_true, colors_true):
            class_member_mask = (y_true == k)
            xyz = X_vis[class_member_mask]
            ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], edgecolor='k', s=marker_size, alpha=0.7, label=f'Class {k}')  # Adjust alpha if needed

        ax2.set_title(f'Ground Truth: {title}')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.set_zlabel('Component 3')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_clustering(X_scaled: np.ndarray, 
                        labels: np.ndarray, 
                        y_true: np.ndarray, 
                        n_clusters: int, 
                        title: str = ''):
    """
    Evaluate clustering performance with AMI, V-measure, and Silhouette Score.
    
    Parameters:
    - X_scaled: np.ndarray
        Scaled feature matrix.
    - labels: np.ndarray
        Predicted cluster labels.
    - y_true: np.ndarray
        Ground truth labels.
    - n_clusters: int
        Number of clusters.
    - title: str
        Title for the evaluation results.
    """
    # Adjusted Mutual Information and V-measure
    ami = adjusted_mutual_info_score(y_true, labels)
    v_measure = v_measure_score(y_true, labels)

    print(f"{title}")
    print(f"Adjusted Mutual Information: {ami:.4f}")
    print(f"V-measure: {v_measure:.4f}")

    # Silhouette Score (only if number of clusters > 1)
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        silhouette_avg = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    else:
        silhouette_avg = None
        print("Silhouette Score: Cannot be calculated with less than 2 clusters.")

    print("=====================================================================")

    return ami, v_measure, silhouette_avg