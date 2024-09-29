"""
This is the data generation script. 
It creates the artificial clusters for our theoretical analysis of the algorithms stated.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os

def generate_isotropic_blobs(output_dir:str)->None:
    """
    Generates isotropic Gaussian blobs with varying cluster sizes.
    """
    print("Generating Isotropic Gaussian Blobs...")
    centers = [(-5, -5), (0, 0), (5, 5)]
    cluster_std = [1.0, 1.5, 0.5]  # Varying cluster sizes
    X, y = make_blobs(n_samples=[300, 500, 200], centers=centers, 
                      cluster_std=cluster_std,random_state=42)
                      
    X, y = shuffle(X, y, random_state=42) # so that the algorithms do not learn any order-related patterns in the dataset
    
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Label'] = y
    df.to_csv(os.path.join(output_dir, 'isotropic_blobs.csv'), index=False) # Save to CSV
    
    print("Isotropic Gaussian Blobs generated and saved to isotropic_blobs.csv\n")

def generate_anisotropic_blobs(output_dir: str) -> None:
    """
    Generates absolutely elongated Gaussian blobs by applying a strong linear transformation.
    """
    print("Generating Anisotropic Gaussian Blobs...")
    
    # Generate the blobs dataset with higher cluster_std for better separation
    X, y = make_blobs(n_samples=1000, 
                      centers=3, 
                      cluster_std=1.8,  # Keep some spread, but not too dense
                      random_state=42)
    
    # Apply a very strong linear transformation to create highly elongated clusters
    transformation = np.array([[9.0, 0.0],    # Strong elongation in the x-direction
                               [0.0, 0.1]])   # Very little spread in the y-direction
    
    X_aniso = np.dot(X, transformation)
    
    # Shuffle the dataset
    X_aniso, y = shuffle(X_aniso, y, random_state=42)
    
    # Save to CSV
    df = pd.DataFrame(X_aniso, columns=['Feature1', 'Feature2'])
    df['Label'] = y
    df.to_csv(os.path.join(output_dir, 'anisotropic_blobs.csv'), index=False)
    print("Anisotropic Gaussian Blobs generated and saved to anisotropic_blobs.csv\n") 


def generate_moons_and_circles(output_dir: str) -> None:
    """
    Generates a combined dataset of moons and circles with specifically no overlap.
    """ 
    print("Generating Moons and Circles...")

    # Generate the moons dataset
    X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=600)

    # Generate the circles dataset
    X_circles, y_circles = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=600)
    
    # Adjust the labels to be unique
    y_circles += 2  # Labels: 0 and 1 for moons, 2 and 3 for circles
    
    # Translate the circles dataset to avoid overlap
    X_circles += np.array([2.0, 2.0])  # Adding an offset of (2, 2) to all points in the circles dataset
    
    # Combine the datasets
    X = np.vstack((X_moons, X_circles))
    y = np.hstack((y_moons, y_circles))
    
    # Shuffle the combined dataset
    X, y = shuffle(X, y, random_state=42)
    
    # Save to CSV
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Label'] = y
    df.to_csv(os.path.join(output_dir, 'moons_circles_no_overlap.csv'), index=False)
    print("Moons and Circles generated saved to moons_circles_no_overlap.csv\n")

def generate_overlapping_clusters(output_dir:str)->None:
    """
    Generates clusters that overlap to test algorithms' ability to handle ambiguous clusters 
    (favourable for soft clustering methods).
    """
    print("Generating Overlapping Clusters...")
    centers = [(0, 0), (4, 4), (2, 2)]
    cluster_std = [1, 1, 1]  # Large std to create overlap
    X, y = make_blobs(n_samples=1000,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=42)
    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)
    # Save to CSV
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Label'] = y
    df.to_csv(os.path.join(output_dir, 'overlapping_clusters.csv'), index=False)
    print("Overlapping Clusters generated and saved to overlapping_clusters.csv\n")

def generate_high_dimensional_data(output_dir:str)->None:
    """
    Generates high-dimensional data with noise 
    (challenge dataset, simulation of complex, 
    real-life data with multiple features, extracted feature vectors for images etc.)
    """
    print("Generating High-Dimensional Data with Noise...")
    n_samples = 1000
    n_features = 50  # High dimensionality
    n_informative = 10
    n_clusters = 5

    # Generate informative features using make_blobs
    X_informative, y = make_blobs(n_samples=n_samples,
                                  centers=n_clusters,
                                  n_features=n_informative,
                                  cluster_std=2.5,
                                  random_state=42)

    # Generate noise features
    X_noise = np.random.uniform(-10, 10, size=(n_samples, n_features - n_informative))

    # Combine informative and noise features
    X = np.hstack((X_informative, X_noise))

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Standardize features for better performance
    X = StandardScaler().fit_transform(X)

    # Save to CSV
    columns = [f'Feature{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['Label'] = y
    df.to_csv(os.path.join(output_dir, 'high_dimensional_data.csv'), index=False)
    print("High-Dimensional Data generated and saved to high_dimensional_data.csv\n")

def main():
    output_dir = 'datasets'
    os.makedirs(output_dir, exist_ok=True)
    # generate_isotropic_blobs(output_dir)
    # generate_anisotropic_blobs(output_dir)
    # generate_moons_and_circles(output_dir)
    # generate_overlapping_clusters(output_dir)
    # generate_high_dimensional_data(output_dir)
    print("All datasets have been generated and saved.")

if __name__ == '__main__':
    main()
