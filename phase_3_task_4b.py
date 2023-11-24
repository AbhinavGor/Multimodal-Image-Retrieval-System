import ast
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from phase_3_task_4a import LSH

def load_lsh_index(csv_file):
    data = pd.read_csv(csv_file)
    
    # Determine the number of layers from the loaded data
    num_layers = data['Layer'].max() + 1
    num_hashes = 10  # You may need to adjust this based on your actual number of hashes
    num_dimensions = 0  # You may need to adjust this based on your actual dimensions

    lsh_index = LSH(num_layers=num_layers, num_hashes=num_hashes, num_dimensions=num_dimensions)
    query_vectors = []

    for _, row in data.iterrows():
        layer_idx = int(row['Layer'])
        hash_key = ast.literal_eval(row['Hash Key'])
        vectors = eval(row['Vectors'])
        lsh_index.tables[layer_idx][hash_key] = vectors
        query_vectors.extend(vectors)

    # Add query_vectors as an attribute to the LSH object
    lsh_index.query_vectors = query_vectors

    return lsh_index

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_search(query_vector, lsh, threshold=1.0, top_l=5):
    query_vector_flat = np.array(query_vector).flatten()
    print("Query Vector Shape:", query_vector_flat.shape)
    query_vector_flat = query_vector_flat.reshape(1, -1)
    print("Reshaped Query Vector Shape:", query_vector_flat.shape)

    candidates = lsh.query(query_vector_flat.flatten(), threshold)
    candidates.sort(key=lambda x: euclidean(query_vector_flat.flatten(), x[1]))  # Sort by distance

    top_candidates = candidates[:top_l]
    return top_candidates

def main():
    # Load LSH index from the CSV file
    num_layers = int(input("Enter the number of layers: "))
    num_hashes = int(input("Enter the number of hashes: "))

    feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
    print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
    feature = int(input("Select one of the feature space from above:"))

    field_to_extract = feature_names[feature-1]

    csv_file_name = f"InMemory_Index_Structure_{field_to_extract}_layers{num_layers}_hashes{num_hashes}.csv"

    print("Searching file name", csv_file_name)
    lsh = load_lsh_index(csv_file_name)

    # Prompt user for an image ID
    image_id = int(input("Enter an image ID:"))

    # Get the query vector from the LSH index based on the image ID
    query_vector = lsh.query_vectors[image_id]

    # Perform image search using the LSH index structure
    top_candidates = image_search(query_vector, lsh, threshold=1.0, top_l=5)

    # Display the query image and top candidates
    query_image, _ = caltech_dataset[image_id]
    query_image = query_image.permute(1, 2, 0).numpy()
    candidate_images = [caltech_dataset[idx][0].permute(1, 2, 0).numpy() for idx, _ in top_candidates]

    images_to_display = [query_image] + candidate_images
    titles = ["Query Image"] + [f"Candidate {i+1}" for i in range(len(candidate_images))]

    # Display images
    display_images(images_to_display, titles)

if __name__ == "__main__":
    main()
