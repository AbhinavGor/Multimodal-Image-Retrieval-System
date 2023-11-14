import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_openml

from database_connection import connect_to_mongo
import sys
# Function to load Caltech101 dataset
def load_caltech101_dataset():
    caltech101 = fetch_openml(data_home='./', name="caltech101")
    images = caltech101.images
    labels = caltech101.target.astype(int)
    return images, labels

# Function to load and preprocess images
def load_and_preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))  # Resize images to a common size (adjust as needed)
        images.append(img)
    return images

# Function to compute DBScan clusters
def compute_dbscan_clusters(images, eps, min_samples):
    X = np.array(images).reshape(len(images), -1)
    X = StandardScaler().fit_transform(X)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    return labels

# Function to perform MDS and visualize clusters
def visualize_clusters(images, labels, title):
    mds = MDS(n_components=2, dissimilarity="precomputed")
    distances = pairwise_distances(images, metric="euclidean")
    X_2d = mds.fit_transform(distances)

    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(12, 8))
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {label}", s=50)
    
    plt.title(title)
    plt.legend()
    plt.show()

# Function to evaluate image classification
def evaluate_image_classification(images, labels, cluster_labels):
    unique_labels = np.unique(labels)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    
    for label in unique_labels:
        mask = (labels == label)
        cluster_labels_masked = cluster_labels[mask]
        most_common_label = np.bincount(cluster_labels_masked).argmax()
        
        precision = precision_score(cluster_labels_masked, [most_common_label] * len(cluster_labels_masked))
        recall = recall_score(cluster_labels_masked, [most_common_label] * len(cluster_labels_masked))
        f1 = f1_score(cluster_labels_masked, [most_common_label] * len(cluster_labels_masked))
        accuracy = accuracy_score(cluster_labels_masked, [most_common_label] * len(cluster_labels_masked))
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
    
    return precision_scores, recall_scores, f1_scores, accuracy_scores

# Main function
def main():
    mongo_client = connect_to_mongo()

    dbname = mongo_client.cse515_project_phase1
    collection = dbname.phase2_features
    rep_image_collection = dbname.phase2_representative_images

    image_data = []

    for image in collection.find():
        print(image["image_id"])
        image_data.append(np.array(image[str(sys.argv[1])]).flatten())

    even_labels = compute_dbscan_clusters(image_data, eps=0.5, min_samples=5)
    for i in even_labels: print(i)

    visualize_clusters(image_data, even_labels, "Even-Numbered Images Clusters")

    # precision, recall, f1, accuracy = evaluate_image_classification(odd_images, labels[1::2], even_labels)
    
    # print(f"Per-Label Precision: {precision}")
    # print(f"Per-Label Recall: {recall}")
    # print(f"Per-Label F1-Score: {f1}")
    # print(f"Overall Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
