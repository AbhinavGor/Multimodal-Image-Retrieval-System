import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2

# Set the path to the directory containing the Caltech 101 dataset
dataset_path = 'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases'

# List all the image files in the dataset
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# Filter even-numbered images
even_numbered_images = [img for img in image_files if int(img.split("_")[1]) % 2 == 0]

# Initialize an empty list to store the flattened image data
image_data = []

# Load and flatten even-numbered images
for img_file in even_numbered_images:
    img_path = os.path.join(dataset_path, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = img.flatten()  # Flatten the image to a 1D array
    image_data.append(img)

# Convert the list of flattened images to a numpy array
image_data = np.array(image_data)

# Standardize the data
scaler = StandardScaler()
image_data_std = scaler.fit_transform(image_data)

# Perform PCA to compute the inherent dimensionality
pca = PCA()
pca.fit(image_data_std)

# Print the explained variance for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained variance for each principal component:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i + 1}: {var:.4f}")

# Compute the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance)

# Find the number of principal components needed to capture a significant amount of variance (e.g., 95%)
threshold_variance = 0.95
inherent_dimensionality = np.argmax(cumulative_explained_variance >= threshold_variance) + 1

print(f"Inherent Dimensionality: {inherent_dimensionality}")
