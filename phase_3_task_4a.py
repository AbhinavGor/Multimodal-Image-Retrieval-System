from collections import defaultdict
import csv
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from database_connection import connect_to_mongo
from helper_functions import top_k_min_indices
from lsh import EuclideanLSHRefined

np.set_printoptions(threshold=np.inf)
mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

num_layers = int(input("Enter the number of layers: "))
num_hashes = int(input("Enter the number of hashes per layer: "))

feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

selected_feature = feature_names[feature-1]

csv_file_name = f"InMemory_Index_Structure_{selected_feature}_layers{num_layers}_hashes{num_hashes}.csv"

image_features = []

image_features = []
image_ids = []
for document in collection.find({}, {selected_feature: 1, "image_id": 1}):
    if selected_feature in document:
        image_features.append(document[selected_feature])
        image_ids.append(document["image_id"])

lsh = EuclideanLSHRefined(num_layers, num_hashes, len(image_features[0]))
ctr = 0
for vec in image_features:
    print(image_ids[ctr])
    lsh.add_vector(vec, image_ids[ctr])
    ctr += 1

index_structure = lsh.get_index_structure()
    
lsh.save_index_to_csv(csv_file_name)

print("In-memory Index Structure is saved to file", csv_file_name)

collection = dbname.phase3_odd_features
query = collection.find_one({"image_id": "2501"})
query = tuple(np.array(query[selected_feature]).flatten())
result_image_ids, result_feature_vectors = lsh.query(query, 100)

print(result_image_ids)
print("Overall images considered: ", len(result_image_ids))
print("Unique images considered: ", len(set(result_image_ids)))

collection = dbname.phase2_features
result_image_features = collection.find({"image_id" : { "$in" :  result_image_ids}})

final_vectors = []
final_ids = []
for i in result_image_features:
    if(len(i[selected_feature]) > 0):
        final_vectors.append(i[selected_feature])
        final_ids.append(i["image_id"])
    else:
        print("No feat: ", i["image_id"])

df = pd.DataFrame({'ImageID': final_ids, 'FeatureVector': final_vectors})
df['relevance'] = ''

# Randomly assign relevances "R+", "R-", "I-", "I+" to 16 rows
random_rows = np.random.choice(df.index, 16, replace=False)
# Assign labels "A", "B", "C", "D" to exactly 4 rows each
label_counts = {'R+': 4, 'R-': 4, 'I-': 4, 'I+': 4}

for label, count in label_counts.items():
    indices = np.random.choice(df.index[df['relevance'] == ''], count, replace=False)
    df.loc[indices, 'relevance'] = label

# Save the DataFrame to a CSV file
df.to_csv('task_4_output.csv', index=False)