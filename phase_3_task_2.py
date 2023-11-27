from database_connection import connect_to_mongo
import numpy as np


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.visited = set()

    def fit(self, X):
        self.labels = np.full(X.shape[0], -1, dtype=int)
        cluster_id = 0

        for i in range(X.shape[0]):
            print(i)
            if i not in self.visited:
                self.visited.add(i)
                neighbors = self.region_query(X, i)

                if len(neighbors) < self.min_samples:
                    self.labels[i] = 0  # Mark as noise
                else:
                    cluster_id += 1
                    self.expand_cluster(X, i, neighbors, cluster_id)

        return self.labels

    def region_query(self, X, center_idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[center_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, X, center_idx, neighbors, cluster_id):
        self.labels[center_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if neighbor_idx not in self.visited:
                self.visited.add(neighbor_idx)
                new_neighbors = self.region_query(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id

            i += 1


mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features

features = collection.find()

image_features = []

for i in features:
    print(i["image_id"])
    image_features.append(np.array(i["layer3"]).flatten())
# Example usage:
if __name__ == "__main__":
    # Generate some random data for testing
    np.random.seed(42)
    X = np.random.rand(100, 1000)

    # Instantiate and fit the DBSCAN model
    # # clusters = 5
    # eps = 2.4
    # min_samples = 10
    #
    # # clusters = 9
    # eps = 2.4
    # min_samples = 6
    #
    # # clusters = 6
    # eps = 2.6
    # min_samples = 6
    #
    # # clusters = 10
    # eps = 2.37
    # min_samples = 6
    eps = 2.37
    min_samples = 6
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit(np.array(image_features))

    # Print the resulting labels
    print("DBSCAN Labels:", list(labels))
    print(len(np.array(np.unique(labels)).tolist()))
