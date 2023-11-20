import csv
import numpy as np
from scipy.spatial.distance import euclidean

from database_connection import connect_to_mongo
            
class LSH:
    def __init__(self, num_layers, num_hashes, num_dimensions):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.num_dimensions = num_dimensions
        self.tables = [{} for _ in range(num_layers)]

        self.hash_functions = [self.generate_hash_function() for _ in range(num_layers * num_hashes)]

    def generate_hash_function(self):
        random_vector = np.random.randn(1, self.num_dimensions)
        random_offset = np.random.uniform(0, 1)
        return lambda x: int((np.dot(random_vector, x) + random_offset) / 0.1)

    def hash_vector(self, vector, layer_idx):
        hashes = []
        for i in range(self.num_hashes):
            hash_value = self.hash_functions[layer_idx * self.num_hashes + i](vector)
            hashes.append(hash_value)
        return tuple(hashes)

    def index_vectors(self, vectors):
        for layer_idx in range(self.num_layers):
            for vector_idx, vector in enumerate(vectors):
                hash_key = self.hash_vector(vector, layer_idx)
                if hash_key not in self.tables[layer_idx]:
                    self.tables[layer_idx][hash_key] = []
                self.tables[layer_idx][hash_key].append((vector_idx, vector))

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

csv_file_name = f"InMemory_Index_Structure_{feature_names[feature-1]}.csv"

image_features = []
match feature:
    case 1:
        image_features = list(features_coll.find(
            {
                "color_moment": 1
            }))
    case 2:
        image_features = list(features_coll.find(
            {
                "hog": 1
            }))
    case 3:
        image_features = list(features_coll.find(
            {
                "layer3": 1
            }))
    case 4:
        image_features = list(features_coll.find(
            {
                "avgpool": 1
            }))
    case 5:
        image_features = list(features_coll.find(
            {
                "fc": 1
            }))

# np.random.seed(42)
image_features = [np.random.randn(1000) for _ in range(8500)]

# print(image_features)

lsh = LSH(num_layers, num_hashes, len(image_features[0]))
lsh.index_vectors(image_features)

index_structure = lsh.get_index_structure()
# for i, table in enumerate(index_structure):
#     print(f"Layer {i}:", table)
    
lsh.save_index_to_csv(csv_file_name)

print("In-memory Index Structure is saved to file", csv_file_name)
