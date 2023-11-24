import ast
import csv
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from database_connection import connect_to_mongo

num_layers = 0
num_hashes = 0
feature = 99999
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]

class LSH:
    def generate_hash_function(self, num_dimensions):
        random_vector = np.random.randn(1024)
        random_vector = random_vector.reshape(1, -1)
        random_offset = np.random.uniform(0, 0.1)
        return lambda x: int((np.dot(random_vector, x) + random_offset) / 0.1)

    def __init__(self, num_layers, num_hashes, num_dimensions):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.num_dimensions = num_dimensions
        self.tables = [{} for _ in range(num_layers)]

        self.hash_functions = [self.generate_hash_function(num_dimensions) for _ in range(num_layers * num_hashes)]
        #print("Hash functions:", self.hash_functions)

    def hash_vector(self, vector, layer_idx):
        hash_values = [self.hash_functions[layer_idx * self.num_hashes + i](vector) for i in range(self.num_hashes)]
        hash_key = tuple(hash_values)
        return hash_key

    def index_vectors(self, vectors):
        for vector in vectors:
            for layer_idx in range(self.num_layers):
                hash_key = self.hash_vector(vector, layer_idx)
                if hash_key not in self.tables[layer_idx]:
                    self.tables[layer_idx][hash_key] = []
                self.tables[layer_idx][hash_key].append(vector)

    def query(self, query_vector, threshold=1.0):
        candidates = set()
        for layer_idx in range(self.num_layers):
            hash_key = self.hash_vector(query_vector, layer_idx)
            if hash_key in self.tables[layer_idx]:
                candidates.update(self.tables[layer_idx][hash_key])

        # Filter candidates based on actual distance
        filtered_candidates = [(idx, vector) for idx, vector in candidates if euclidean(query_vector, vector) <= threshold]

        return filtered_candidates

    def get_index_structure(self):
        return self.tables
    
    def save_index_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Layer', 'Hash Key', 'Vectors'])

            # Write data
            for layer_idx, table in enumerate(self.tables):
                for hash_key, vectors in table.items():
                    writer.writerow([layer_idx, hash_key, vectors])

client = connect_to_mongo()
db = client.CSE515ProjectDB
features_coll = db.Phase2

num_layers = int(input("Enter the number of layers: "))
num_hashes = int(input("Enter the number of hashes: "))

feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

field_to_extract = feature_names[feature-1]

csv_file_name = f"InMemory_Index_Structure_{field_to_extract}_layers{num_layers}_hashes{num_hashes}.csv"

if (not os.path.isfile(csv_file_name)):

    extracted_items = []

    image_features = []
    match feature:
        case 1:
            for document in features_coll.find({}, {field_to_extract: 1}):
                if field_to_extract in document:
                    extracted_items.append(document[field_to_extract])
        case 2:
            for document in features_coll.find({}, {field_to_extract: 1}):
                if field_to_extract in document:
                    extracted_items.append(document[field_to_extract])
        case 3:
            for document in features_coll.find({}, {field_to_extract: 1}):
                if field_to_extract in document:
                    extracted_items.append(document[field_to_extract])
        case 4:
            for document in features_coll.find({}, {field_to_extract: 1}):
                if field_to_extract in document:
                    extracted_items.append(document[field_to_extract])
        case 5:
            for document in features_coll.find({}, {field_to_extract: 1}):
                if field_to_extract in document:
                    extracted_items.append(document[field_to_extract])

    # np.random.seed(42)
    #image_features = [np.random.randn(1000) for _ in range(8500)]

    # print(image_features)

    lsh = LSH(num_layers, num_hashes, len(extracted_items[0]))
    lsh.index_vectors(extracted_items)

    index_structure = lsh.get_index_structure()
    # for i, table in enumerate(index_structure):
    #     print(f"Layer {i}:", table)
        
    lsh.save_index_to_csv(csv_file_name)

    print("In-memory Index Structure is saved to file", csv_file_name)

def load_lsh_index(csv_file):
    data = pd.read_csv(csv_file)
    
    # Determine the number of layers, hashes, and dimensions from the loaded data
    num_layers = data['Layer'].max() + 1
    num_hashes = data.groupby('Layer').size().max()
    
    # Assuming all vectors have the same dimensions, take the dimensions from the first row
    first_row = data.iloc[0]
    num_dimensions = len(ast.literal_eval(first_row['Hash Key']))

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
    # Handle arrays with different numbers of dimensions
    query_vector_flat = np.concatenate([arr.flatten() if isinstance(arr, np.ndarray) else np.array([arr]).flatten() for arr in query_vector])
    print("Query Vector Shape:", query_vector_flat.shape)
    #query_vector_flat = query_vector_flat.reshape(1, -1)
    print("Reshaped Query Vector Shape:", query_vector_flat.shape)

    candidates = lsh.query_vectors #lsh.query(query_vector_flat.flatten(), threshold)
    candidates.sort(key=lambda x: euclidean(query_vector_flat.flatten(), np.array(x[1]).flatten()))  # Sort by distance

    top_candidates = candidates[:top_l]
    return top_candidates

def main():
    # Load LSH index from the CSV file
    # num_layers = int(input("Enter the number of layers: "))
    # num_hashes = int(input("Enter the number of hashes: "))

    # print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
    # feature = int(input("Select one of the feature space from above:"))

    field_to_extract = feature_names[feature-1]

    csv_file_name = f"InMemory_Index_Structure_{field_to_extract}_layers{num_layers}_hashes{num_hashes}.csv"

    print("Searching file name", csv_file_name)
    lsh = load_lsh_index(csv_file_name)

    # Prompt user for an image ID
    image_id = int(input("Enter an image ID:"))

    # Get the query vector from the LSH index based on the image ID
    query_vector = lsh.query_vectors[image_id]

    # Perform image search using the LSH index structure
    top_k = int(input("Enter the number of top images to retrieve (k):"))
    top_candidates = image_search(query_vector, lsh, threshold=1.0, top_l=top_k)

    print(top_candidates)

    # Output only the matching image IDs
    matching_image_ids = [idx[0] for idx in top_candidates]
    print(f"Matching Image IDs: {matching_image_ids}")

if __name__ == "__main__":
    main()
