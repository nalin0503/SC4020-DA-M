import numpy as np
import random
import math
from sklearn.cluster import KMeans

# Accurate Adder Sum
def accurate_adder(num1, num2, tot_num_bits, inaccurate_bits):
    num1 = num1 % (2 ** tot_num_bits)
    num2 = num2 % (2 ** tot_num_bits)
    total = num1 + num2
    total = total % (2 ** (tot_num_bits + 1))
    return total

def kmeans_with_adder(X, k, max_iters=100, random_state=26, adder=accurate_adder, bits=(32, 4)):
    # set the seed for reproducibility
    np.random.seed(random_state)
    # Initialize centroids randomly
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    centroid_track = [centroids]  # Track centroid values over iterations for visualisation

    # Number of epochs
    for iter in range(max_iters):

        # Assign samples to closest centroids (create clusters)
        clusters = [[] for _ in range(k)] # clusters is a list of k empty lists. We will put our clustered data points in these lists
        
        for idx, point in enumerate(X):
            # For every point in the dataset, calculate the distance between the point and each centroid
            distances = [0 for _ in range(k)] # This will end up being a list of k distances for each point
            
            for i, centroid in enumerate(centroids):
                distance = 0
                for j in range(n_features):
                    distance = adder(distance, subtract_using_adder(point[j], centroid[j], adder, bits)**2, *bits)
                    # distance = adder(distance, (point[j] - centroid[j])**2, *bits)  # Calculate the Euclidean distance between the point and the centroid
                    # distance += (point[j] - centroid[j])**2  # Calculate the Euclidean distance between the point and the centroid
                    # distance = math.sqrt(distance)
                distances[i] = adder(distance, distances[i], *bits) # Add the distance to the list of distances
                # distances[i] += distance

            centroid_idx = np.argmin(distances) # Choose index of the the centroid with the smallest distance for that point
            clusters[centroid_idx].append(idx) # Add the point to the cluster with the smallest distance
        
        # Now we have assigned all points to clusters. We will now update the centroids
        new_centroids = []

        for cluster in clusters:
            if cluster:
                # The cluster is not empty, we will update the centroid to the mean of the points in the cluster
                new_centroids.append(X[cluster].mean(axis=0))
            else:
                # The cluster is empty, we will reinitialize it to a random point
                new_centroids.append(X[np.random.randint(0, n_samples)])
        new_centroids = np.array(new_centroids)
        centroid_track.append(new_centroids)
        
        if np.allclose(centroids, new_centroids, 0.1):
            break
        centroids = new_centroids

    return clusters, centroids, np.array(centroid_track)

def kmeansplus_with_adder(X, k, max_iters=100,  random_state=26, adder=accurate_adder, bits=(32, 4)):
    # this will be the exact same as the kmeans_with_adder function, except we will use the kmeans++ initialization
    # set the seed for reproducibility
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=random_state)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    centroid_track = [centroids]  # Track centroid values over iterations for visualisation

    # Number of epochs
    for iter in range(max_iters):

        # Assign samples to closest centroids (create clusters)
        clusters = [[] for _ in range(k)] # clusters is a list of k empty lists. We will put our clustered data points in these lists
        
        for idx, point in enumerate(X):
            # For every point in the dataset, calculate the distance between the point and each centroid
            distances = [0 for _ in range(k)] # This will end up being a list of k distances for each point
            
            for i, centroid in enumerate(centroids):
                distance = 0
                for j in range(n_features):
                    distance = adder(distance, subtract_using_adder(point[j], centroid[j], adder, bits)**2, *bits)
                    # distance = adder(distance, (point[j] - centroid[j])**2, *bits)  # Calculate the Euclidean distance between the point and the centroid
                    # distance += (point[j] - centroid[j])**2  # Calculate the Euclidean distance between the point and the centroid
                    # distance = math.sqrt(distance)
                distances[i] = adder(distance, distances[i], *bits) # Add the distance to the list of distances
                # distances[i] += distance

            centroid_idx = np.argmin(distances) # Choose index of the the centroid with the smallest distance for that point
            clusters[centroid_idx].append(idx) # Add the point to the cluster with the smallest distance
        
        # Now we have assigned all points to clusters. We will now update the centroids
        new_centroids = []

        for cluster in clusters:
            if cluster:
                # The cluster is not empty, we will update the centroid to the mean of the points in the cluster
                new_centroids.append(X[cluster].mean(axis=0))
            else:
                # The cluster is empty, we will reinitialize it to a random point
                new_centroids.append(X[np.random.randint(0, n_samples)])
        new_centroids = np.array(new_centroids)
        centroid_track.append(new_centroids)
        
        if np.allclose(centroids, new_centroids, 0.1):
            break
        centroids = new_centroids

    return clusters, centroids, np.array(centroid_track)

def kmeans(X, k, max_iters=100, random_state=42) -> tuple[np.array, np.array, np.array]:
    """
    @param X: np.array of shape (n_samples, n_features)
    @param k: int, number of clusters
    @param max_iters: int, maximum number of iterations
    @param random_state: int, seed for reproducibility
    @return clusters: list of lists, each list contains the indices of the data points in that cluster
    @return centroids: np.array of shape (k, n_features), final centroids
    @return centroid_track: np.array of shape (n_iters, k, n_features), track of centroid values over iterations
    """
    
    # set the seed for reproducibility
    np.random.seed(random_state)
    # Initialize centroids randomly
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    centroid_track = [centroids]  # Track centroid values over iterations for visualisation

    # Number of epochs
    for iter in range(max_iters):

        # Assign samples to closest centroids (create clusters)
        clusters = [[] for _ in range(k)] # clusters is a list of k empty lists. We will put our clustered data points in these lists
        
        for idx, point in enumerate(X):
            # For every point in the dataset, calculate the distance between the point and each centroid
            distances = [0 for _ in range(k)] # This will end up being a list of k distances for each point
            
            for i, centroid in enumerate(centroids):
                distance = 0
                for j in range(n_features):
                    distance += (point[j] - centroid[j])**2
                    distance = math.sqrt(distance)  # Calculate the Euclidean distance between the point and the centroid
                distances[i] += distance

            centroid_idx = np.argmin(distances) # Choose index of the the centroid with the smallest distance for that point
            clusters[centroid_idx].append(idx) # Add the point to the cluster with the smallest distance
        
        # Now we have assigned all points to clusters. We will now update the centroids
        new_centroids = []

        for cluster in clusters:
            if cluster:
                # The cluster is not empty, we will update the centroid to the mean of the points in the cluster
                new_centroids.append(X[cluster].mean(axis=0))
            else:
                # The cluster is empty, we will reinitialize it to a random point
                new_centroids.append(X[np.random.randint(0, n_samples)])
        new_centroids = np.array(new_centroids)
        centroid_track.append(new_centroids)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids


    return np.matrix(clusters), centroids, np.array(centroid_track)

def kmeans_plus(X, k, max_iters=100, random_state=42):
    # this will be the exact same as the kmeans function, except we will use the kmeans++ initialization
    # set the seed for reproducibility
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=random_state)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    centroid_track = [centroids]  # Track centroid values over iterations for visualisation
    
    # Number of epochs
    for iter in range(max_iters):

        # Assign samples to closest centroids (create clusters)
        clusters = [[] for _ in range(k)] # clusters is a list of k empty lists. We will put our clustered data points in these lists
        
        for idx, point in enumerate(X):
            # For every point in the dataset, calculate the distance between the point and each centroid
            distances = [0 for _ in range(k)] # This will end up being a list of k distances for each point
            
            for i, centroid in enumerate(centroids):
                distance = 0
                for j in range(n_features):
                    distance += (point[j] - centroid[j])**2  # Calculate the Euclidean distance between the point and the centroid
                    distance = math.sqrt(distance)
                distances[i] += distance # Add the distance to the list of distances
                # distances[i] += distance

            centroid_idx = np.argmin(distances) # Choose index of the the centroid with the smallest distance for that point
            clusters[centroid_idx].append(idx) # Add the point to the cluster with the smallest distance
        
        # Now we have assigned all points to clusters. We will now update the centroids
        new_centroids = []

        for cluster in clusters:
            if cluster:
                # The cluster is not empty, we will update the centroid to the mean of the points in the cluster
                new_centroids.append(X[cluster].mean(axis=0))
            else:
                # The cluster is empty, we will reinitialize it to a random point
                new_centroids.append(X[np.random.randint(0, n_samples)])
        new_centroids = np.array(new_centroids)
        centroid_track.append(new_centroids)
        
        if np.allclose(centroids, new_centroids, 0.1):
            break
        centroids = new_centroids

    return clusters, centroids, np.array(centroid_track)
    

def calculate_wcss(X, clusters, centroids):
    wcss = 0
    for i, cluster in enumerate(clusters):
        points = X[cluster]
        centroid = centroids[i]
        squared_distances = np.sum((points - centroid) ** 2, axis=1)
        wcss += np.sum(squared_distances)
    return wcss

def subtract_using_adder(a, b, adder, adder_args):
    # We are trying to compute a - b
    # Notice that that = C - (b-a) - C
    # Where C is a chosen arbitrary constant
    # So that = C - b + a - C
    # So = adder(C - b, a, ...) - C
    if a < 0 or b < 0: 
        # print(f"Both numbers must be positive, got {a} and {b}")
        return abs(a - b)

    # a must be the bigger number so that the result is positive
    if b > a: a, b = b, a
    
    C = 10 ** (math.floor(math.log10(a) + 1)) # This ensures C - b is never negative
    return adder(C - b, a, *adder_args) - C
