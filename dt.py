import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from database_connection import connect_to_mongo

client = connect_to_mongo()
dbname = client.cse515_project_phase1
collection = dbname.phase2_features

image_data = []
image_features = collection.find()
document_count = collection.count_documents({})
for image in tqdm(image_features, desc="Loading Progress", unit="images", total=document_count):
    image_data.append({"feature": np.array(image["hog"]).flatten(), "target": image["target"]})

data = np.array(image_data)

X = np.array([image["feature"] for image in data])
y = np.array([image["target"] for image in data])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fit the classifier to the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(y_pred)
# print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

