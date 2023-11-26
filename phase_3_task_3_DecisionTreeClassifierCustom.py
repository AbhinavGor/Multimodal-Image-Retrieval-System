import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from database_connection import connect_to_mongo

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

class DecisionTreeClassifierCustom:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        print(f"Current depth: {curr_depth}")
        print(f"Number of samples: {num_samples}")
        print(f"Number of features: {num_features}")

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                print(f"Splitting at feature {best_split['feature_index']} with threshold {best_split['threshold']}")
                print(f"Left dataset size: {len(best_split['dataset_left'])}, Right dataset size: {len(best_split['dataset_right'])}")
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree,
                            best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(y)
        print(f"Reached leaf node with value: {leaf_value}")
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        print("Best Split:")
        print(f"Feature Index: {best_split['feature_index']}")
        print(f"Threshold: {best_split['threshold']}")
        print(f"Info Gain: {best_split['info_gain']}")
        print(f"Left Dataset Size: {len(best_split['dataset_left'])}")
        print(f"Right Dataset Size: {len(best_split['dataset_right'])}")

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = dataset[dataset[:, feature_index] <= threshold]
        dataset_right = dataset[dataset[:, feature_index] > threshold]
        
        #print("Split:")
        print(f"Feature Index: {feature_index}")
        print(f"Threshold: {threshold}")
        print(f"Left Dataset Size: {len(dataset_left)}")
        print(f"Right Dataset Size: {len(dataset_right)}")

        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)

    def fit(self, X, y):
        # Ensure X is a 2D array
        if X.ndim > 2:
            # If X has more than 2 dimensions, flatten it
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 1:
            # If X is 1D, convert it to a 2D array
            X = np.atleast_2d(X).T

        # Ensure y is a 1D array
        y = np.squeeze(y)

        # Reshape y to be a column vector
        y = y.reshape(-1, 1)

        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Load data from MongoDB
client = connect_to_mongo()
db = client.CSE515ProjectDB
collection = db.Phase2

extracted_items = []
image_features = collection.find()

feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

field_to_extract = feature_names[feature-1]

for document in collection.find({}, {field_to_extract: 1, "target": 1}):
    if field_to_extract in document:
        extracted_items.append({"feature": document[field_to_extract], "target": document["target"]})

# Convert data to NumPy array
data = np.array(extracted_items)

# Shuffle and split the data
split_ratio = 0.8
np.random.shuffle(data)
split_index = int(len(data) * split_ratio)

X_data = np.array([d["feature"] for d in data])
y_data = np.array([d["target"] for d in data])
X_train, X_test = X_data[:split_index], X_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# Initialize and fit the custom Decision Tree model
tree = DecisionTreeClassifierCustom(min_samples_split=2, max_depth=2)
print("Tree:", tree)
tree.fit(X_train, y_train)

# Predict using the custom Decision Tree model
y_pred = tree.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print actual vs. predicted values
print("Actual vs. Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")

txtfilename = "Tree_result"
np.savetxt(txtfilename, tree, fmt='%.6f', delimiter='\t')
