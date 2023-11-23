from collections import defaultdict
import random
import numpy as np
import csv
import numpy as np
from scipy.spatial.distance import euclidean
from database_connection import connect_to_mongo
            
class LSH:
    def __init__(self, num_layers, num_hashes, num_dimensions, bucket_width=1.0):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.num_dimensions = num_dimensions
        self.bucket_width = bucket_width
        self.tables = [{} for _ in range(num_layers)]

        self.hash_functions = [self.generate_hash_function() for _ in range(num_layers * num_hashes)]

    def generate_hash_function(self):
        random_vector = np.random.randn(1, self.num_dimensions)
        random_offset = np.random.uniform(0, self.bucket_width)
        return lambda x: int((np.dot(random_vector, x) + random_offset) / self.bucket_width)

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

    def query(self, query_vector, threshold=None):
        threshold = threshold if threshold is not None else self.bucket_width * 5
        candidates = set()
        for layer_idx in range(self.num_layers):
            hash_key = self.hash_vector(query_vector, layer_idx)
            if hash_key in self.tables[layer_idx]:
                candidates.update(self.tables[layer_idx][hash_key])

        # Filter candidates based on actual distance
        filtered_candidates = [(idx, vector) for idx, vector in candidates if euclidean(query_vector, vector) <= threshold]

        return filtered_candidates
    
    def multi_probe_query(self, query_vector, num_probes=3):
        candidates = set()
        for layer_idx in range(self.num_layers):
            primary_hash_key = self.hash_vector(query_vector, layer_idx)
            probe_keys = [primary_hash_key]  # Start with primary hash key
            # Generate additional probe keys (simplified example)
            for _ in range(1, num_probes):
                probe_keys.append(self.generate_probe_key(primary_hash_key))
            
            for key in probe_keys:
                if key in self.tables[layer_idx]:
                    candidates.update(self.tables[layer_idx][key])

        # Filter candidates based on actual distance
        # ... same filtering process as in the query method ...

        return candidates

    def generate_probe_key(self, primary_hash_key):
        # Example implementation to generate a new probe key
        # This should be improved based on specific requirements
        return tuple([x + np.random.choice([-1, 0, 1]) for x in primary_hash_key])

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
