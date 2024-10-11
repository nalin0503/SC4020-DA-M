import numpy as np
import matplotlib.pyplot as plt

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange']

def calculate_wcss(X, clusters, centroids):
    wcss = 0
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            wcss += np.sum((X[idx] - centroids[i]) ** 2)
    return wcss

# Plot the data itself
def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Raw data w/o clusters")
    plt.show()

