from collections import defaultdict
import csv
import random
import numpy as np
import pandas as pd

from database_connection import connect_to_mongo
from helper_functions import top_k_min_indices

mongo_client = connect_to_mongo()

dbname = mongo_client.cse515_project_phase1
collection = dbname.phase2_features
rep_image_collection = dbname.phase2_representative_images

class EuclideanLSHRefined:
    def __init__(self, L, h, dim):
        self.L = L  # Number of layers
        self.h = h  # Number of hashes per layer
        self.dim = dim  # Dimensionality of the input vectors
        self.hash_tables = [defaultdict(list) for _ in range(L)]
        self.id_hash_tables = [defaultdict(list) for _ in range(L)]
        self._generate_hash_functions()

    def _generate_hash_functions(self):
        self.hash_functions = []
        for _ in range(self.L):
            # Generate 'h' hash functions for each layer
            layer_functions = []
            for _ in range(self.h):
                random_vector = np.random.normal(0, 1, self.dim)
                random_offset = random.uniform(0, 1)
                layer_functions.append((random_vector, random_offset))
            self.hash_functions.append(layer_functions)

    def _hash_vector(self, vector, hash_functions):
        # Compute hash values for a vector using given hash functions
        hash_values = []
        for a, b in hash_functions:
            hash_value = np.floor(np.dot(a, vector) + b).astype(int)
            hash_values.append(hash_value)
        # For a vector we are going to have a hash value for each layer
        return tuple(hash_values)

    def add_vector(self, vector, image_id):
        # Add a vector to the LSH index
        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            # print(hash_val)
            self.hash_tables[i][hash_val].append(vector)
            self.id_hash_tables[i][hash_val].append(image_id)

    def query(self, vector, max_results=None):
        # Find similar vectors for the given query vector
        candidates = []
        candidate_ids = []
        # for i, layer_functions in enumerate(self.hash_functions):
        #     hash_val = self._hash_vector(vector, layer_functions)
        #     res = self.hash_tables[i].get(hash_val, [])
            
        #     for vector in res: candidates.append(tuple(vector))
        
        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            res_ids = self.id_hash_tables[i].get(hash_val, [])
            
            for id in res_ids: candidates.append(id)

        # Optionally limit the number of results
        # if max_results is not None:
        #     candidates = list(candidates)[:max_results]
        return candidates
    
    def get_index_structure(self):
        return self.hash_tables
    
    def save_index_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Layer', 'Hash Key', 'Vectors', 'Image ID'])

            # Write data
            for layer_idx, table in enumerate(self.hash_tables):
                i = 0
                for hash_key, vectors in table.items():
                    writer.writerow([layer_idx, hash_key, vectors, str(int(i*2))])
                    i+=1

    def find_top_k_similar_images(self, query_vector, k, vectors):
        candidate_ids = self.query(query_vector)

        # Calculate actual Euclidean distances and find top k
        distances = [np.linalg.norm(query_vector - vectors[candidate_id]) for candidate_id in candidate_ids[0]]
        sorted_indices = np.argsort(distances)[:k]
        top_k_similar_images = [candidate_ids[idx] for idx in sorted_indices]

        return top_k_similar_images

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
    
# lsh.save_index_to_csv(csv_file_name)

print("In-memory Index Structure is saved to file", csv_file_name)

collection = dbname.phase3_odd_features
query = collection.find_one({"image_id": "2501"})
query = tuple(np.array(query[selected_feature]).flatten())
result = lsh.query(query, 100)

print(result)
print("Overall images considered: ", len(result))
print("Unique images considered: ", len(set(result)))

# ID2501, L1, H1, [6, 28, 56, 64, 74, 84, 92, 106, 122, 126, 162, 166, 182, 216, 278, 284, 286, 300, 322, 348, 376, 400, 402, 410, 414, 426, 434, 444, 446, 470, 472, 474, 488, 490, 536, 578, 580, 584, 598, 618, 626, 656, 662, 664, 672, 688, 692, 694, 696, 712, 720, 722, 728, 744, 746, 750, 770, 822, 846, 852, 858, 1112, 1162, 1200, 1206, 1214, 1240, 1254, 1262, 1272, 1288, 1324, 1332, 1352, 1354, 1356, 1394, 1404, 1416, 1418, 1422, 1432, 1440, 1442, 1450, 1468, 1484, 1498, 1542, 1550, 1566, 1572, 1594, 1608, 1642, 1650, 1676, 1678, 1700, 1714]
# ID2501, L3, H3, [498, 1430, 1504, 1658, 1892, 2246, 4830, 5974, 6418, 7978, 8160, 884, 1036, 1186, 1554, 1644, 2156, 2290, 7312, 1540, 1670, 1792, 2594, 3012, 5352]
# ID2501, L10, H10, []
# ID2501, L10, H3, [400, 742, 1354, 1502, 1890, 2000, 2032, 2048, 2938, 3218, 3510, 3646, 3968, 4436, 4676, 4732, 5694, 5812, 7136, 7290, 7326, 7370, 8284, 8294, 912, 2536, 2500, 7194, 1244, 2540, 6868, 8248, 1410, 1066, 1732, 2376, 2418, 5020, 5396, 6306, 6732, 6946, 1108, 1528, 3844, 4192, 4306, 5078, 5098, 5214, 5260, 56, 1578, 1876, 2866, 2978, 3054, 3250, 3372, 3376, 4132, 5820, 184, 284, 1804, 2102, 2282, 2530, 2540, 2586, 2614, 2698, 3386, 4012, 4164, 4356, 4564, 5622, 5648, 6044, 6354, 6390, 6418, 6554, 7482, 8142]
# ID2501, L20, H4, [8014, 3022, 6814, 1520, 5568, 7850, 10, 902, 2588, 1032, 3416, 2012, 2412, 3462, 5224]
# ID2501, L14, H4, [1482, 6788, 1160, 2514, 972, 3898, 5752, 6670, 4694, 416, 1346, 1842, 2618, 3766, 4092, 7014, 704, 6092]
# ID2501, L16, H4, [2540, 5346, 68, 8032, 1034, 6352, 372, 3252, 4414, 3696, 1916]
# ID2501, L16, H6, [2540, 5346, 68, 8032, 1034, 6352, 372, 3252, 4414, 3696, 1916]
# ID2501, L16, H2, [44, 1660, 2352, 2574, 2594, 3770, 4792, 5846, 6736, 6744, 7394, 7782, 7830, 8012, 50, 314, 508, 686, 838, 876, 958, 990, 994, 1070, 1088, 1128, 1194, 1414, 1510, 1534, 1668, 1688, 1784, 1868, 2122, 2242, 2270, 2298, 2314, 2378, 2380, 2400, 2470, 2480, 2532, 2538, 2600, 2640, 2660, 2952, 2958, 3102, 3258, 3322, 3346, 3482, 3564, 3704, 3792, 3856, 3910, 3972, 4064, 4148, 4160, 4336, 4356, 4382, 4496, 4568, 4680, 4700, 4734, 4868, 4882, 4930, 4994, 5108, 5154, 5172, 5188, 5258, 5368, 5394, 5490, 5686, 5772, 5856, 5908, 5910, 6090, 6554, 6808, 6840, 6842, 6880, 6966, 6974, 7212, 7214]
# ID2501, L20, H2, [36, 52, 140, 170, 242, 254, 302, 326, 334, 1218, 1524, 1620, 1856, 1948, 2028, 2424, 2428, 2660, 2814, 2836, 4430, 4434, 5402, 5866, 5908, 6332, 6794, 7008, 7172, 7230, 7266, 8008, 8234, 18, 200, 292, 356, 358, 362, 398, 408, 412, 418, 426, 590, 656, 732, 760, 812, 884, 928, 946, 952, 956, 988, 1004, 1010, 1038, 1196, 1804, 1834, 1940, 2080, 2092, 2130, 2150, 2192, 2214, 2326, 2476, 2546, 2588, 2618, 2704, 2722, 2724, 2740, 2746, 2808, 2842, 2856, 2946, 2986, 3192, 3206, 3348, 3362, 3462, 3558, 3604, 3760, 3786, 3904, 3940, 3948, 3956, 3978, 4022, 4080, 4232]