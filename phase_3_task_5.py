import pandas as pd
import numpy as np
import ast

from MultiClassSVM import MultiClassSVM
from output_plotter import output_plotter

# Load the data from the CSV file
df = pd.read_csv('task_4_output.csv')
df["FeatureVector"] = df["FeatureVector"].apply(ast.literal_eval)
# Extract feature vectors and labels
IDs = np.array(df['ImageID'].tolist())
X = np.array(df['FeatureVector'].tolist())
y = np.array(df['relevance'])

# Map labels to integers for training the SVM
label_mapping = {'R+': 0, 'R-': 1, 'I-': 2, 'I+': 3}
y_numeric = np.array([label_mapping[label] if label in label_mapping else -1 for label in y])

# Filter rows with labels for training
train_mask = y_numeric != -1
X_train = X[train_mask]
y_train = y_numeric[train_mask]
print(y_train)
# Filter rows without labels for testing
test_mask = ~train_mask
X_test = X[test_mask]
y_test = y_numeric[test_mask]
IDs = IDs[test_mask]

# Create a SVM classifier
clf = MultiClassSVM(C=10000)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

classIds = [[] for i in range(4)]

y_pred = list(y_pred[0])
print(y_pred)
for i in range(len(y_pred)):
    y_pred[i]
    classIds[int(y_pred[i])].append(IDs[i])
print(len(classIds[0]))
print(classIds[0])
output_plotter(classIds[0])