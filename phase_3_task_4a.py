from collections import defaultdict
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
from PIL import Image
import torchvision
import torch
from torchvision import datasets, models
import torchvision.transforms as transforms

from database_connection import connect_to_mongo
from helper_functions import top_k_min_indices
from lsh import EuclideanLSHRefined

np.set_printoptions(threshold=np.inf)

def on_button_click(event):
    global feedback_list
    feedback_list.append(text_box.text)
    plt.close()

def on_click(event):
    global current_coords
    current_coords = (event.xdata, event.ydata)
    rect = Rectangle((current_coords[0] - 0.5, current_coords[1] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    plt.draw()

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

feedback_list = []

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
count = collection.count_documents({"image_id" : { "$in" :  result_image_ids}})
similarities = []
similarity_ids = []
similarity_vectors = []
for i in result_image_features:
    similarities.append(math.dist(np.array(query).flatten(), np.array(i[selected_feature]).flatten()))
    similarity_ids.append(int(i["image_id"]))
    similarity_vectors.append(np.array(i[selected_feature]).flatten().tolist())

top_k_indices = top_k_min_indices(similarities)

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101', download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

top_k_ids = []
top_k_vectors = []
i = 0
random_indices = np.random.permutation(np.arange(0, len(similarity_ids)))
for i in random_indices:
    if len(np.unique(feedback_list)) == 4:
        break
    img, label = dataset[similarity_ids[i]]
    top_k_ids.append(similarity_ids[i])
    top_k_vectors.append(similarity_vectors[i])
    
    fig, ax = plt.subplots()
    img = torch.tensor(np.array(img))

    ax.imshow((np.squeeze(img)))

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Add a button for user feedback
    ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
    button = Button(ax_button, 'Submit Feedback')
    button.on_clicked(on_button_click)

    # Add a text box for user input
    ax_textbox = plt.axes([0.1, 0.01, 0.65, 0.05])
    text_box = TextBox(ax_textbox, 'Feedback: ', initial="")

    plt.show()
    i+=1
print(feedback_list)
# final_vectors = []
# final_ids = []
# for i in result_image_features:
#     if(len(i[selected_feature]) > 0):
#         final_vectors.append(i[selected_feature])
#         final_ids.append(i["image_id"])
#     else:
#         print("No feat: ", i["image_id"])

df = pd.DataFrame({'ImageID': similarity_ids, 'FeatureVector': similarity_vectors})
df['relevance'] = ''

# # Randomly assign relevances "R+", "R-", "I-", "I+" to 16 rows
# random_rows = np.random.choice(df.index, 16, replace=False)
# # Assign labels "A", "B", "C", "D" to exactly 4 rows each
# label_counts = {'R+': 4, 'R-': 4, 'I-': 4, 'I+': 4}

# for label, count in label_counts.items():
#     indices = np.random.choice(df.index[df['relevance'] == ''], count, replace=False)
#     df.loc[indices, 'relevance'] = label

for i in range(len(feedback_list)):
    image_id = top_k_ids[i]
    new_relevance = feedback_list[i]
    
    df.loc[df['ImageID'] == image_id, 'relevance'] = new_relevance
# Save the DataFrame to a CSV file
df.to_csv('task_4_output.csv', index=False)