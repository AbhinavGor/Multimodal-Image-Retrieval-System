from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from database_connection import connect_to_mongo
from helper_functions import classical_mds

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

image_data = []
even_indices = []
label_indices = []
for image in collection.find():
    print(image["image_id"])
    even_indices.append(int(image['image_id']))
    label_indices.append(int(image['target']))
    image_data.append(np.array(image["layer3"]).flatten())
# Assuming your data is loaded into a variable 'X'
# You may need to apply dimensionality reduction and preprocessing if necessary

# Normalize or standardize the data
# scaler = StandardScaler()
X_normalized = image_data
# Create and fit the DBScan model
# # For color moment
# eps = 2.4
# min_samples = 2
eps = 2.4
min_samples = 2
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X_normalized)

# Get cluster labels (-1 represents noise points)
labels = dbscan.labels_

cluster_indices = defaultdict(list)

# Loop through the cluster labels and store image indices in the corresponding cluster
for i, label in enumerate(labels):
    cluster_indices[label].append(i)

# print(cluster_indices)

# Print the number of clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

X = np.array(X_normalized)
# Apply Multi-Dimensional Scaling (MDS) for dimensionality reduction
# Reduce the data to 2 dimensions for visualization
X_mds = MDS(n_components=2, max_iter=500, n_init=10, verbose=2, n_jobs=10)
distance_matrix = squareform(pdist(X, 'euclidean'))
distance_matrix = 0.5 * \
    (distance_matrix + distance_matrix.T)  # Make it symmetric

# Specify the number of dimensions
num_dimensions = 2

# Specify the number of iterations and random initializations
num_iterations = 100
num_init = 5


# X_mds = classical_mds(X)

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(10, 6))

# Plot points for each cluster with different colors
print(labels)
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        # Plot noise points in black
        plt.scatter(X_mds[labels == label, 0],
                    X_mds[labels == label, 1], c='k', label='Noise')
    else:
        # Plot points in each cluster with different colors
        plt.scatter(X_mds[labels == label, 0],
                    X_mds[labels == label, 1], label=f'Cluster {label}')

plt.title('DBScan Clustering Results with MDS Visualization')
plt.legend()
plt.grid(True)
plt.show()
