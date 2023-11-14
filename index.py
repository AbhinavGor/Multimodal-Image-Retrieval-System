import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

from database_connection import connect_to_mongo

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
client = connect_to_mongo()
dbname = client.cse515_project_phase1
collection = dbname.phase2_features

image_data = []
image_features = collection.find()
document_count = collection.count_documents({})
for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
    image_data.append({"feature": np.array(image["hog"]), "target": image["target"]})

data = np.array(image_data)

X = [image["feature"] for image in data]
y = [image["target"] for image in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(k=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

actual_vs_predicted = list(zip(y_test, y_pred))

with open('knn_actual_vs_predicted.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Actual', 'Predicted'])
    for actual, predicted in actual_vs_predicted:
        writer.writerow([actual, predicted])