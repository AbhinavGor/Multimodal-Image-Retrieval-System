from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.manifold import MDS
from sklearn.metrics import f1_score, precision_score, recall_score
from database_connection import connect_to_mongo
import numpy as np

from output_plotter import output_plotter, output_plotter_task_2


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
image_ids = []
image_labels = []
for i in features:
    print(i["image_id"])
    image_features.append(np.array(i["layer3"]).flatten())
    image_ids.append(i['image_id'])
    image_labels.append(i["target"])

c = int(input("Enter c value for number of clusters: "))
eps = 2.4
min_samples = 10

if c == 10:
    eps = 2.37
    min_samples = 6

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit(np.array(image_features))

# Print the resulting labels
print("DBSCAN Labels:", list(labels))
print(len(np.array(np.unique(labels)).tolist()))

label_encoder = LabelEncoder()
class_labels_encoded = label_encoder.fit_transform(labels)
print("Encoded labels ", class_labels_encoded)

cluster_dict = {}

# Iterate over the image IDs and cluster labels to populate the dictionary
for image_id, cluster_label in zip(image_ids, labels):
    if cluster_label not in cluster_dict:
        cluster_dict[cluster_label] = [image_id]
    else:
        cluster_dict[cluster_label].append(image_id)

print("Printing images")
for i in list(cluster_dict.keys()):
    output_plotter_task_2(cluster_dict[i], f"Cluster {i} images")

try:
    precision = precision_score(image_labels, labels, average=None)
    recall = recall_score(image_labels, labels, average=None)
    f1 = f1_score(image_labels, labels, average=None)

    print("| Label | Precision | Recall | F Score |")
    for i in range(len(precision)):
        print(
            f"| {i}     | {precision[i]}         | {recall[i]}      | {f1[i]}       |")
    # Print the results
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
except:
    print("Error encountered when calculating label-wise scores")

mds = MDS(verbose=2, n_components=2, n_init=6, max_iter=400,
          dissimilarity="euclidean", n_jobs=8)
print(mds)
feature_vectors_2d = mds.fit_transform(image_features)

# Scatter plot the points with different colors for each class
for label in np.unique(class_labels_encoded):
    indices = class_labels_encoded == label
    plt.scatter(feature_vectors_2d[indices, 0],
                feature_vectors_2d[indices, 1], label=label_encoder.classes_[label])

plt.title('MDS Visualization of Feature Vectors')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.show()
