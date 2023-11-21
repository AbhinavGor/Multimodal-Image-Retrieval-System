class EuclideanLSHRefined:
    def __init__(self, L, h, dim):
        self.L = L  # Number of layers
        self.h = h  # Number of hashes per layer
        self.dim = dim  # Dimensionality of the input vectors
        self.hash_tables = [defaultdict(list) for _ in range(L)]
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
        return tuple(hash_values)

    def add_vector(self, vector):
        # Add a vector to the LSH index
        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            self.hash_tables[i][hash_val].append(vector)

    def query(self, vector, max_results=None):
        # Find similar vectors for the given query vector
        candidates = set()
        for i, layer_functions in enumerate(self.hash_functions):
            hash_val = self._hash_vector(vector, layer_functions)
            candidates.update(self.hash_tables[i].get(hash_val, []))

        # Optionally limit the number of results
        if max_results is not None:
            candidates = list(candidates)[:max_results]
        return candidates

lsh_refined = EuclideanLSHRefined(L=5, h=10, dim=256)
# Vectors can be added using lsh_refined.add_vector(some_vector) and queried using lsh_refined.query(query_vector)

