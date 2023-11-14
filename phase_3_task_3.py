import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from database_connection import connect_to_mongo

def euclidean_distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))

def get_nearest_neighbors(test_sample, train_data, k):
    distances = [(i, euclidean_distance(test_sample, train_sample)) for i, train_sample in enumerate(train_data)]
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def predict_class(neighbors, train_labels):
    neighbor_labels = [train_labels[i] for i, _ in neighbors]
    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def predict(test_data, train_data, train_labels, k):
    predictions = []
    for i, test_sample in enumerate(test_data):
        nearest_neighbors = get_nearest_neighbors(test_sample, train_data, k)
        predicted_class = predict_class(nearest_neighbors, train_labels)
        predictions.append(predicted_class)
        # Add a progress print statement
        if i % 100 == 0:
            print(f'Processed {i} test samples out of {len(test_data)}')
    return predictions

client = connect_to_mongo()
dbname = client.cse515_project_phase1
collection = dbname.phase2_features

image_data = []
image_features = collection.find()
document_count = collection.count_documents({})
for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
    image_data.append({"feature": image["hog"], "target": image["target"]})
# print(image_data.size)
data = np.array(image_data)
split_ratio = 0.8

np.random.shuffle(data)
split_index = int(len(data)*split_ratio)

X_data = [d["feature"] for d in data]
y_data = [d["target"] for d in data]
X_train, X_test = X_data[:split_index], X_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

y_pred = predict(X_test, X_train, y_train, 20)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print actual vs. predicted values
print("Actual vs. Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")
