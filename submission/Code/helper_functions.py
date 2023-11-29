from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD


def find_lowest_index_greater_than(arr, target_value):
    left, right = 0, len(arr) - 1
    lowest_index = None

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] > target_value:
            lowest_index = mid  # Update the lowest index found so far
            right = mid - 1
        else:
            left = mid + 1

    return lowest_index


def find_explained_variance(image_data):
    image_data = np.array(image_data)

    # Use the smaller of the two dimensions
    n_components = min(image_data.shape[0], image_data.shape[1])
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(image_data)
    singular_values = svd.singular_values_

    explained_variance = np.cumsum(singular_values) / np.sum(singular_values)

    return n_components, explained_variance, singular_values


def mds(data, n_components=2, max_iterations=1500, n_init=1, verbose=True):
    # Number of data points
    n = data.shape[0]

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    best_stress = float('inf')
    best_Y = None

    for init in range(n_init):
        # Random initialization of configuration
        X = np.random.rand(n, n_components)

        # Double centering
        B = -0.5 * H @ data @ H

        for iteration in range(max_iterations):
            # Compute pairwise Euclidean distances in the current configuration
            dist_matrix = squareform(pdist(X, 'euclidean'))

            # Compute stress
            stress = np.sum((data - dist_matrix) ** 2)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{max_iterations}, Stress: {stress}")

            # Check convergence
            if iteration > 0 and np.abs(stress - prev_stress) < 1e-8:
                break

            prev_stress = stress

            Bx = B @ X
            eigenvalues, eigenvectors = np.linalg.eigh(Bx)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
            Y = eigenvectors @ np.diag(sqrt_eigenvalues)

            # Normalize rows of Y
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)

            X = Y

        # Check if this initialization gives a lower stress
        if stress < best_stress:
            best_stress = stress
            best_Y = Y

    return best_Y


def top_k_min_indices(arr, k=None):
    if k is not None:
        indices = np.argsort(arr)[:k]
    else:
        indices = np.argsort(arr)[:len(arr)]
    return indices


def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


def range_query(data, point, epsilon):
    neighbors = []
    for i, other_point in enumerate(data):
        if euclidean_distance(point, other_point) <= epsilon:
            neighbors.append(i)
    return neighbors
