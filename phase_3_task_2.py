import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
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
clustering = DBSCAN(eps=0.5, min_samples=5).fit(image_data)

labels = clustering.labels_
print("MDS")
mds = MDS(n_components=2, dissimilarity="precomputed")
distances = 1 / (1 + np.abs(np.corrcoef(image_data)))  # Calculate distances
mds_coordinates = mds.fit_transform(distances)

# Plot clusters
plt.scatter(mds_coordinates[:, 0], mds_coordinates[:, 1], c=labels)
plt.title("Clusters in 2D MDS Space")
plt.show()

unique_labels = np.unique(labels)

for label in unique_labels:
    print(label)
    cluster_indices = np.where(labels == label)[0]
    cluster_images = [image_data[i] for i in cluster_indices]

    # Create a grid of thumbnails
    num_rows = int(np.ceil(len(cluster_images) / 5))
    fig, ax = plt.subplots(num_rows, 5, figsize=(10, 2 * num_rows))

    for i, image in enumerate(cluster_images):
        row, col = divmod(i, 5)
        ax[row, col].imshow(image, cmap='gray')
        ax[row, col].axis('off')

    plt.suptitle(f'Cluster {label}')
    plt.show()