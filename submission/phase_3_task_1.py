import math
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from database_connection import connect_to_mongo
from helper_functions import top_k_min_indices

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase3_task1_features
phase_2_features = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

feature = str(sys.argv[1])
print("Trying to fetch latent semantics....")

data = collection.find({"feature":  feature})
data_document_count = collection.count_documents({"feature": feature})
print(data_document_count)
if data_document_count == 0:
    choice = str(
        input("No latent semantics found. Do you want to generate them now?(Y/N)"))
    if choice != 'N':
        print("Generating latent semantics")
        for i in range(101):
            print(i)
            rep_image = rep_image_collection.find_one(
                {"target": i, "feature": feature})
            class_image_data = phase_2_features.find({"target": i})
            class_image_data = [image[feature] for image in class_image_data]

            distances = [math.dist(np.array(rep_image["feature_value"]).flatten(
            ), np.array(feat).flatten()) for feat in class_image_data]
            top_ids = top_k_min_indices(distances, 10)

            features = [np.array(class_image_data[i]).flatten().tolist()
                        for i in top_ids]

            collection.insert_one(
                {"target": i, "rep_features": features, "feature": feature})

        print("Generated latent semantics")

    else:
        print("Executing task not possible without latent semantics.")
        time.sleep(1)
        print("Exiting....")
        time.sleep(1)
        exit()

# query_image = str(input("Enter the query image ID"))
query_collection = dbname.phase3_odd_features
query_image_features = query_collection.find()
# query_image_feature = query_image_features[feature]

print("num of odd images ", query_collection.count_documents({}))

prediction = []
actual = []
for image in query_image_features:
    scores = np.zeros(101)
    for i in range(101):
        image_data = collection.find({"feature": feature, "target": i})
        label_rep_scores = [float(math.dist(np.array(j).flatten(), np.array(
            image[feature]).flatten())) for j in image_data[0]["rep_features"]]
        scores[i] += np.sum(label_rep_scores)
    # print(scores)
    result = top_k_min_indices(scores, 2)
    prediction.append(result[1])
    actual.append(image["target"])
    print(prediction, actual)
print(prediction, actual)
precision = precision_score(actual, prediction, average=None)
recall = recall_score(actual, prediction, average=None)
f1 = f1_score(actual, prediction, average=None)

print("| Label | Precision | Recall | F Score |")
for i in range(len(precision)):
    print(
        f"| {i}     | {precision[i]}         | {recall[i]}      | {f1[i]}       |")
# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# y_pred = [item for item in prediction]
# y_true = [item for item in actual]

# # Calculate precision, recall, and F1-score
# precision = precision_score(y_true, y_pred, average='binary')
# recall = recall_score(y_true, y_pred, average='binary')
# f1 = f1_score(y_true, y_pred, average='binary')

# # Print the results
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1-score: {f1:.2f}')
