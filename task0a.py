import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from database_connection import connect_to_mongo

# Connect to the mongo database
mongo_client = connect_to_mongo()
db = mongo_client.cse515_project_phase1
collection = db.phase2_features

layer3 = []
for i in collection.find():
    layer3.append(i["layer3"])

# Standardize the data
scaler = StandardScaler()
images_std = scaler.fit_transform(layer3)

# Compute PCA
n_components = int(len(layer3[0]))  # You can adjust this number
pca = PCA(n_components=n_components)
pca.fit(images_std)

# Compute the explained variance
explained_variance = pca.explained_variance_
print(explained_variance)
inherent_dimensionality = np.sum(explained_variance[:int(n_components/1.32)])/np.sum(explained_variance)
print("n_components", int(n_components/1.3), " /", int(n_components))
print(f"Inherent Dimensionality of even-numbered Caltech101 images: {inherent_dimensionality:.4f}")
