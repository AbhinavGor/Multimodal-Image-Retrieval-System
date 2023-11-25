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


def dbscan(data, eps, min_pts):
    clusters = []
    visited = set()

    for point in data:
        if point in visited:
            continue
        visited.add(point)
        neighbor_points = get_neighbors(data, point, eps)

        if len(neighbor_points) < min_pts:
            continue

        cluster, visited = expand_cluster(
            data, point, neighbor_points, eps, min_pts, visited)
        clusters.append(cluster)

    return clusters


def expand_cluster(data, point, neighbor_points, eps, min_pts, visited):
    cluster = [point]

    for neighbor in neighbor_points:
        if neighbor not in visited:
            visited.add(neighbor)
            new_neighbors = get_neighbors(data, neighbor, eps)

            if len(new_neighbors) >= min_pts:
                neighbor_points.extend(new_neighbors)

        if neighbor not in [p for c in clusters for p in c]:
            cluster.append(neighbor)

    return cluster, visited


def get_neighbors(data, point, eps):
    neighbors = []
    for p in data:
        if np.linalg.norm(point - p) <= eps:
            neighbors.append(p)
    return neighbors

def find_explained_variance(image_data):
    image_data = np.array(image_data)

    n_components = min(image_data.shape[0], image_data.shape[1])  # Use the smaller of the two dimensions
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(image_data)
    singular_values = svd.singular_values_

    explained_variance = np.cumsum(singular_values) / np.sum(singular_values)

    return n_components, explained_variance, singular_values


def classical_mds(data, n_components=2, max_iterations=1500, n_init=1, verbose=True):
    """
    Perform classical MDS on the input distance matrix.

    Parameters:
        - data: Input distance matrix.
        - n_components: Number of dimensions for the output.
        - max_iterations: Maximum number of iterations.
        - n_init: Number of random initializations.
        - verbose: If True, print stress at each iteration.

    Returns:
        - Y: Embedding of data in reduced-dimensional space.
    """

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

            # Update configuration
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

# # Example usage
# np.random.seed(42)
# # Generate a random distance matrix
# distance_matrix = np.random.rand(10, 10)
# distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)  # Make it symmetric

# # Specify the number of dimensions
# num_dimensions = 2

# # Specify the number of iterations and random initializations
# num_iterations = 100
# num_init = 5

# # Run classical MDS
# embedding = classical_mds(distance_matrix, num_dimensions, max_iterations=num_iterations, n_init=num_init)
# print("Final Embedding:")
# print(embedding)
def top_k_min_indices(arr, k):
    indices = np.argpartition(arr, k)[:k]
    return indices