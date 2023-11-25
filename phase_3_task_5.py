import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.metrics import accuracy_score
import ast

# Create a SVM classifier
class MultiClassSVM:
    def __init__(self, C=1):
        self.C = C
        self.models = []

    def fit(self, X, y):
        classes = np.unique(y)
        num_classes = len(classes)
        print("Classes", num_classes)
        num_samples, num_features = X.shape

        for i in range(num_classes):
            # Create a binary classification problem for each class
            binary_labels = np.where(y == classes[i], 1, -1)

            # Define the variables
            w = cp.Variable(num_features)
            b = cp.Variable()

            # Define the objective function
            obj = cp.Minimize(0.5 * cp.norm(w, 2) + self.C * cp.sum(cp.pos(1 - cp.multiply(binary_labels, (X @ w + b)))))

            # Define the problem
            prob = cp.Problem(obj)

            # Solve the problem
            prob.solve()

            # Save the model parameters
            self.models.append((w.value, b.value))

    def predict(self, X):
        scores = []
        for w, b in self.models:
            scores.append(X @ w + b)
            # print(w, b)
        scores = np.array(scores).T
        return [np.argmax(scores, axis=1)]

# Load the data from the CSV file
df = pd.read_csv('task_4_output.csv')
df["FeatureVector"] = df["FeatureVector"].apply(ast.literal_eval)
# Extract feature vectors and labels
X = np.array(df['FeatureVector'].tolist())
y = np.array(df['relevance'])

# Map labels to integers for training the SVM
label_mapping = {'R+': 0, 'R-': 1, 'I-': 2, 'I+': 3}
y_numeric = np.array([label_mapping[label] if label in label_mapping else -1 for label in y])

# Filter rows with labels for training
train_mask = y_numeric != -1
X_train = X[train_mask]
y_train = y_numeric[train_mask]

# Filter rows without labels for testing
test_mask = ~train_mask
X_test = X[test_mask]
y_test = y_numeric[test_mask]

print(X_test.shape)
# Create a SVM classifier
clf = MultiClassSVM(C=4)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
print(y_pred)