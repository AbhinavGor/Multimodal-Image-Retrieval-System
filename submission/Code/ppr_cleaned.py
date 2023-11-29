import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from tqdm import tqdm
from database_connection import connect_to_mongo
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_similarity_matrix2(X):
    similarity_matrix = cosine_similarity(X)

    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix


def personalized_page_rank(similarity_matrix, alpha, max_iter=100, tol=1e-17):
    alpha = 0.85

    transition_matrix = normalize(similarity_matrix, norm="l1", axis=1)

    n = similarity_matrix.shape[0]
    ranks = np.ones(n) / n
    i = 0
    j = 0
    for _ in range(max_iter):
        new_ranks = alpha * np.dot(transition_matrix, ranks) + (1 - alpha) / n

        i = i + 1

        conv_condition = np.linalg.norm(new_ranks - ranks, 1)

        print(f"Iteration {i}, Convergence condition: {conv_condition<tol}")
        if conv_condition < tol:
            j = j + 1
            break
        ranks = new_ranks

    print(f"no of times in if loop:{j}")

    return ranks


client = connect_to_mongo()
dbname = client.cse515_project_phase1
collection = dbname.phase2_features
collection_odd = dbname.phase3_odd_features

odd_image_data = []
odd_image_features = collection_odd.find()
odd_image_count = collection_odd.count_documents({})
image_data = []
image_features = collection.find()
document_count = collection.count_documents({})

jump_prob = float(input("Enter PPR random jump probability: "))

feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print(
    "1. Color Moment\n2. HoG\n3. Layer 3\n4. AvgPool\n5. FC\n6. Label-Label Similarity Matrix\n7. Image-Image Similarity Matrix"
)
feature = int(input("Select one feature from above: "))
feature = feature_names[feature - 1]
my_num = int(input("enter number of images: "))
i = 0
r = my_num
for image in tqdm(
    image_features, desc="Loading Progress", unit="images", total=document_count
):
    image_data.append({"feature": image[feature], "target": image["target"]})
    i = i + 1
    if i == r:
        break
i = 0
for image in tqdm(
    odd_image_features, desc="Loading Progress", unit="images", total=odd_image_count
):
    odd_image_data.append(
        {
            "feature": image[feature],
            "label": image["target"],
            "image_id": image["image_id"],
        }
    )
    i = i + 1
    if i == r:
        break

random.shuffle(image_data)

X_train = [data["feature"] for data in image_data]
print(len(X_train))
y_train = [data["target"] for data in image_data]

random.shuffle(odd_image_data)

X_test = [data["feature"] for data in odd_image_data]
y_test = [data["label"] for data in odd_image_data]


X_train = [np.ravel(feature) for feature in X_train]
y_train = [np.ravel(feature) for feature in y_train]
X_test = [np.ravel(feature) for feature in X_test]
y_test = [np.ravel(feature) for feature in y_test]


similarity_matrix = calculate_similarity_matrix2(X_train)

predicted_labels = []
score_set = set()
for test_sample in X_test:
    personalized_vector = np.array(
        [1 / (euclidean(test_sample, train_sample) + 1e-5)
         for train_sample in X_train]
    )

    ppr_scores = personalized_page_rank(
        similarity_matrix * personalized_vector, jump_prob, max_iter=100
    )

    predicted_label = y_train[np.argmax(ppr_scores)]
    predicted_labels.append(predicted_label)
    score_set.add(tuple(predicted_label))


for actual, predicted in zip(y_test, predicted_labels):
    print(f"Actual: {actual}, Predicted: {predicted}")
print("Length of set :", len(score_set))

accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average="macro")
recall = recall_score(y_test, predicted_labels, average="macro")
f1 = f1_score(y_test, predicted_labels, average="macro")


print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
