import numpy as np
from collections import defaultdict
import random
import csv


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
            self.hash_tables[i][hash_val].append(vector)
            self.id_hash_tables[i][hash_val].append(image_id)

    def query(self, vector, max_results=None):
        # Find similar vectors for the given query vector
        candidates = []
        candidate_ids = []
        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            res = self.hash_tables[i].get(hash_val, [])

            for vector in res:
                candidates.append(tuple(vector))

        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            res_ids = self.id_hash_tables[i].get(hash_val, [])

            for id in res_ids:
                candidate_ids.append(id)

        return candidate_ids, candidates

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
                    writer.writerow(
                        [layer_idx, hash_key, vectors, str(int(i*2))])
                    i += 1

    def find_top_k_similar_images(self, query_vector, k, vectors):
        candidate_ids = self.query(query_vector)

        # Calculate actual Euclidean distances and find top k
        distances = [np.linalg.norm(query_vector - vectors[candidate_id])
                     for candidate_id in candidate_ids[0]]
        sorted_indices = np.argsort(distances)[:k]
        top_k_similar_images = [candidate_ids[idx] for idx in sorted_indices]

        return top_k_similar_images
