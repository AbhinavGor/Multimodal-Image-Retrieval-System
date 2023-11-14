import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys

from database_connection import connect_to_mongo

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

image_data = []

for image in collection.find():
    print(image["image_id"])
    image_data.append(np.array(image[str(sys.argv[1])]).flatten())

X = image_data
# Calculate distances to k-nearest neighbors for a range of k values
k_values = range(30, 51, 5)  # You can adjust the range as needed
distances = []

for k in k_values:
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(X)
    distances_k, _ = nbrs.kneighbors(X)
    distances.append(distances_k[:, -1])  # Select the distance to the k-th nearest neighbor

# Plot the k-distance graph
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values):
    plt.plot(np.sort(distances[i]), label=f'k={k}')

plt.xlabel('Data Points Sorted by Distance')
plt.ylabel('Distance to k-th Nearest Neighbor')
plt.title('K-Distance Graph')
plt.legend()
plt.grid(True)
plt.show()
