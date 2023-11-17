import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from database_connection import connect_to_mongo
import networkx as nx 

# NOT FINISHED
def personalized_page_rank_predict(G, odd_data, even_data, k):
    predictions = {}
    for test_entry in odd_data:
        # Create personalization vector
        personalization = {entry['image_id']: 1 if entry['image_id'] == test_entry['image_id'] else 0 for entry in even_data}
        ranks = nx.pagerank(G, personalization=personalization)

        # Get top k nodes and their labels
        top_k_nodes = sorted(ranks, key=ranks.get, reverse=True)[:k]
        top_k_labels = [even_data[node]['label'] for node in top_k_nodes if node in even_data]

        # Predict label
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        predictions[test_entry['image_id']] = predicted_label

    return predictions

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

classifier_choice = input("Select classifier (mnn, dt, ppr): ")

if classifier_choice == 'mnn':
    k = int(input("Enter k for m-NN: "))
    y_pred = predict(X_test, X_train, y_train, k)
elif classifier_choice == 'dt':
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
elif classifier_choice == 'ppr':
    k = int(input("Enter k for PPR: "))
    # Create graph G and use personalized_page_rank_predict() function
else:
    raise ValueError("Invalid classifier choice")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print results
print(f"Classifier: {classifier_choice}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Print actual vs. predicted values
print("Actual vs. Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")
