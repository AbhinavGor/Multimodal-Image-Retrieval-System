import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
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

# Train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

y_pred = predict(X_test, X_train, y_train, 20)

# Predict using the Decision Tree Classifier
y_pred_dt = dt_classifier.predict(X_test)

# Calculate metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro')
recall_dt = recall_score(y_test, y_pred_dt, average='macro')
f1_dt = f1_score(y_test, y_pred_dt, average='macro')

# Print the results
print(f'Decision Tree Accuracy: {accuracy_dt * 100:.2f}%')
print(f'Decision Tree Precision: {precision_dt:.2f}')
print(f'Decision Tree Recall: {recall_dt:.2f}')
print(f'Decision Tree F1-Score: {f1_dt:.2f}')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print actual vs. predicted values
print("Actual vs. Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")
