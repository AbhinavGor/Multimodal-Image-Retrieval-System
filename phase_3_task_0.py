import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2, os
import torchvision, torch, cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models
import torchvision.transforms as transforms
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm
import sys

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

transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

uri = "mongodb://localhost:27017/cse515_project_phase1"

# Create a new client and connect to the server
mongo_client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

dbnme = mongo_client.cse515_project_phase1
collection = dbnme.phase2_features

image_data = []

# Simulate iterating through a database query result
image_features = collection.find()
document_count = collection.count_documents({})

print("Database query complete")
for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
    # print(image["image_id"])
    image_data.append(np.array(image[str(sys.argv[1])]).flatten())

# Convert the list of flattened images to a numpy array
image_data = np.array(image_data)

n_components = len(image_data[0])
# Perform PCA
pca = PCA(n_components=n_components)
pca.fit(image_data)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio to decide the intrinsic dimensionality
cum_sum = np.cumsum(explained_variance_ratio)
plt.plot(cum_sum)
intrinsic_dim = find_lowest_index_greater_than(cum_sum, 0.95)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.axhline(y=0.95, color='red', linestyle='dotted', label='y = 0.95')
plt.axvline(x=intrinsic_dim, color='green', linestyle='dotted', label='intrinsic_dim')
plt.grid()
plt.show()
