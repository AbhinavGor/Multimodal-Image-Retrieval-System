import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from database_connection import connect_to_mongo


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=0, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _best_split(self, X, y):
        best_idx, best_thr = None, None
        best_gini = 1.0
        num_samples = len(y)
        num_classes = len(set(y))

        for idx in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, idx])
            thresholds = X[sorted_indices, idx]
            classes = y[sorted_indices]

            class_indices = {label: idx for idx,
                             label in enumerate(sorted(set(y)))}
            num_left = [0] * num_classes
            num_right = [np.sum(y == c) for c in range(num_classes)]

            for i in range(1, num_samples):
                c = class_indices[classes[i - 1]]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - \
                    sum((num_left[x] / i) ** 2 for x in range(num_classes))
                gini_right = 1.0 - sum(
                    (num_right[x] / (num_samples - i)) ** 2 for x in range(num_classes)
                )
                gini = (i * gini_left + (num_samples - i)
                        * gini_right) / num_samples

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == c) for c in range(len(set(y)))]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=1.0 - sum((np.sum(y == c) / len(y)) **
                           2 for c in range(len(set(y)))),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        print(
            f"Depth: {depth}, Num Samples: {len(y)}, Gini: {node.gini}, Predicted Class: {node.predicted_class}")

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                print(f"Depth: {depth}, Threshold: {thr}")
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                # Calculate the predicted class based on the majority class in the leaf node
                predicted_class_left = np.argmax(
                    [np.sum(y_left == c) for c in range(len(set(y_left)))])
                predicted_class_right = np.argmax(
                    [np.sum(y_right == c) for c in range(len(set(y_right)))])

                node.feature_index = idx
                node.threshold = thr

                # Recursively grow the tree for left and right nodes
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

                # Set the predicted class based on the majority class in the leaf node
                node.left.predicted_class = predicted_class_left
                node.right.predicted_class = predicted_class_right

        return node

    def _predict(self, inputs, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if inputs[node.feature_index] < node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

    def fit(self, X, y):
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs, self.tree_) for inputs in X]


def print_tree(node, depth=0):
    if node is not None:
        print("  " * depth, end="")
        if node.left is None and node.right is None:
            print(
                f"Leaf - Predicted Class: {node.predicted_class}, Class Distribution: {node.num_samples_per_class}")
        else:
            print(f"Depth: {depth}, Threshold: {node.threshold}")
            print_tree(node.left, depth + 1)
            print_tree(node.right, depth + 1)


# Load data from MongoDB
client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_features
collection_odd = db.phase3_odd_features

odd_image_data = []
odd_image_features = collection_odd.find()
odd_image_count = collection_odd.count_documents({})
extracted_items = []
image_features = collection.find()

feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

field_to_extract = feature_names[feature-1]

for document in collection.find({}, {field_to_extract: 1, "target": 1}):
    if field_to_extract in document:
        extracted_items.append(
            {"feature": document[field_to_extract], "target": document["target"]})

# Convert data to NumPy array
data = np.array(extracted_items)

for image in tqdm(
    odd_image_features, desc="Loading Progress", unit="images", total=odd_image_count
):
    odd_image_data.append(
        {
            "feature": image[field_to_extract],
            "label": image["target"],
            "image_id": image["image_id"],
        }
    )


odd_image_data_array = np.array(odd_image_data)

X_train = np.array([data["feature"] for data in data])
y_train = np.array([data["target"] for data in data])

print("Training data: ", X_train)
print("Y train: ", set(y_train), len(set(y_train)))

X_test = np.array([data["feature"] for data in odd_image_data_array])
y_test = np.array([data["label"] for data in odd_image_data_array])

print("Test data: ", X_test)
print("Y test: ", set(y_test), len(set(y_test)))

np.savetxt("data.txt", y_train)

# Initialize and fit the custom Decision Tree model
tree = DecisionTreeClassifierCustom(min_samples_split=2, max_depth=5)
tree.fit(X_train, y_train)

# Add this line after fitting the tree
print_tree(tree.tree_)

# Predict using the custom Decision Tree model
y_pred = tree.predict(X_test)


# Print actual vs. predicted values
print("Actual vs. Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
