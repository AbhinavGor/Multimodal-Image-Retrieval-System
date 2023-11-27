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
    rect = Rectangle((current_coords[0] - 0.5, current_coords[1] - 0.5),
                     1, 1, linewidth=1, edgecolor='r', facecolor='none')
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

# 4b
collection = dbname.features
query_id = str(input("Enter the odd query image ID: "))
top_k = int(input("Enter k for top_k: "))
query = collection.find_one({"image_id": str(query_id)})
query = tuple(np.array(query[selected_feature]).flatten())
result_image_ids, result_feature_vectors = lsh.query(query, 100)

print(result_image_ids)
print("Overall images considered: ", len(result_image_ids))
print("Unique images considered: ", len(set(result_image_ids)))

collection = dbname.phase2_features
result_image_features = collection.find(
    {"image_id": {"$in":  result_image_ids}})
count = collection.count_documents({"image_id": {"$in":  result_image_ids}})

similarities = []
final_result_image_ids = []
final_result_image_feature_vectors = []
for i in result_image_features:
    similarities.append(math.dist(np.array(query).flatten(),
                        np.array(i[selected_feature]).flatten()))
    final_result_image_ids.append(int(i["image_id"]))
    final_result_image_feature_vectors.append(
        np.array(i[selected_feature]).flatten().tolist())

top_k_indices = top_k_min_indices(similarities)

top_k_image_ids = [final_result_image_ids[id] for id in top_k_indices][:top_k]
for i in top_k_image_ids:
    print(i, end=" ")
top_k_image_feature_vectors = [
    final_result_image_feature_vectors[id] for id in top_k_indices][:top_k]

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

relevance_ids = []
relevance_vectors = []
i = 0

for i in range(min(int((top_k)/2), int((len(top_k_image_ids))/2))):
    img, label = dataset[top_k_image_ids[i]]
    relevance_ids.append(top_k_image_ids[i])
    relevance_vectors.append(top_k_image_feature_vectors[i])

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

df = pd.DataFrame({'ImageID': top_k_image_ids,
                  'FeatureVector': top_k_image_feature_vectors})
df['relevance'] = ''

for i in range(len(feedback_list)):
    image_id = relevance_ids[i]
    new_relevance = feedback_list[i]

    df.loc[df['ImageID'] == image_id, 'relevance'] = new_relevance
# Save the DataFrame to a CSV file
df.to_csv('task_4_output.csv', index=False)
