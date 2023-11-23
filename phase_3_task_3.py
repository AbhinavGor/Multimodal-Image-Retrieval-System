import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from database_connection import connect_to_mongo
import networkx as nx
import torchvision.transforms as transforms
from torchvision import datasets
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

transforms = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.Caltech101(
    'your_path', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

def load_odd_image_data(feature):
    odd_number_images = []
    for image_ID in range(8677):
        if image_ID % 2 != 0:
            img, label = dataset[image_ID]

            resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
            resized_resnet_img = [cv2.resize(i, (224, 224)) for i in img.numpy()]

            # checking if the image has 3 channels
            if len(resized_img) == 3:
                if feature == "color_moment":
                    descriptor = extract_color_moment(resized_img)
                elif feature == "hog":
                    descriptor = extract_hog(resized_img)
                elif feature == "avgpool":
                    descriptor_temp = extract_from_resnet(resized_resnet_img)
                    descriptor = descriptor_temp["avgpool"]
                elif feature == "layer3":
                    descriptor_temp = extract_from_resnet(resized_resnet_img)
                    descriptor = descriptor_temp["layer3"]
                elif feature == "fc":
                    descriptor_temp = extract_from_resnet(resized_resnet_img)
                    descriptor = descriptor_temp["fc"]
                entry = {
                    "image_id": str(image_ID),
                    "label": label,
                    "feature": descriptor
                }
                odd_number_images.append(entry)
    return odd_number_images

def fetch_data_from_db(descriptor):
    data = []
    for image_data in collection.find():
        if descriptor in image_data:
            entry = {
                'image_id': image_data['image_id'],
                'feature': np.array(image_data[descriptor]),
                'label': image_data['target']
            }
            data.append(entry)
    return data

def create_similarity_graph(data, n):
    G = nx.Graph()
    for i, entry in enumerate(data):
        G.add_node(entry['image_id'], label=entry['label'])
        
        # Calculate similarities with all other nodes
        similarities = [(other_entry['image_id'], 1 / (1 + euclidean_distance(entry['feature'], other_entry['feature']))) for other_entry in data if other_entry['image_id'] != entry['image_id']]
        
        # Sort by similarity and take top n
        top_n_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

        for other_image_id, similarity in top_n_similarities:
            G.add_edge(entry['image_id'], other_image_id, weight=similarity)

    return G

def personalized_page_rank_predict(graph, test_data, alpha):
    predictions = {}
    for test_entry in test_data:
        # Create personalization vector
        personalization = {node: 1 if graph.nodes[node]['label'] == test_entry['label'] else 0 for node in graph.nodes}
        
        ranks = nx.pagerank(graph, personalization=personalization, alpha=alpha)
        
        predicted_label = max(set(ranks), key=ranks.get)
        predictions[test_entry['image_id']] = graph.nodes[predicted_label]['label']
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
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG\n3. Layer 3\n4. AvgPool\n5. FC\n6. Label-Label Similarity Matrix\n7. Image-Image Similarity Matrix")
feature = int(input("Select one feature from above: "))

if classifier_choice == 'mnn':
    m = int(input("Enter m for m-NN: "))
    y_pred = predict(X_test, X_train, y_train, m)
elif classifier_choice == 'dt':
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
elif classifier_choice == 'ppr':
    jump_prob = int(input("Enter PPR random jump probability: "))
    data = fetch_data_from_db(feature)
    graph = create_similarity_graph(data, 10)
    odd_number_images = load_odd_image_data()
    predictions = personalized_page_rank_predict(graph, odd_number_images, jump_prob)
    # y_test = [entry['label'] for entry in odd_number_images]
    y_pred = list(predictions.values())
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
