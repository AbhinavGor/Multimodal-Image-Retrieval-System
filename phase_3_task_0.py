import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
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

from helper_functions import find_explained_variance, find_lowest_index_greater_than
from database_connection import connect_to_mongo

transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

image_data = []

task_selection = int(input("What is your desired output?\n1. Intrinsic dimensionality of even numbered Caltech101 images.\n2. Intrinsic dimensionality of labels of even numbered images in Caltech101\n"))

match task_selection:
    case 1:
        image_features = collection.find()
        document_count = collection.count_documents({})

        print("Database query complete")
        for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
            image_data.append(np.array(image[str(sys.argv[1])]).flatten())
    case 2:
        image_features = rep_image_collection.find({"feature": str(sys.argv[1])})
        document_count = rep_image_collection.count_documents({"feature": str(sys.argv[1])})

        print("Database query complete")
        for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
            try:
                if "feature_value" in image.keys():
                    image_data.append(np.array(image["feature_value"]).flatten())
            except:
                print(image)


n_components, explained_variance, singular_values = find_explained_variance(image_data)
plt.figure()
plt.plot(range(1, n_components + 1), explained_variance, marker='.')
intrinsic_dim = find_lowest_index_greater_than(explained_variance, 0.95)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.axhline(y=0.95, color='red', linestyle='dotted', label='y = 0.95')
plt.axvline(x=intrinsic_dim, color='green', linestyle='dotted', label='intrinsic_dim')
differences = np.diff(singular_values)
elbow_index = np.argmin(differences)  # Find the index of the minimum difference
elbow_x = elbow_index + 1  # Adjust for 0-based indexing
plt.text(elbow_x, singular_values[elbow_index], f'({elbow_x}, {singular_values[elbow_index]:.2f})', fontsize=12, verticalalignment='bottom')
plt.grid()
plt.show()