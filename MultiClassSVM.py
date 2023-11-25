import numpy as np
import cvxpy as cp

class MultiClassSVM:
    def __init__(self, C=1):
        self.C = C
        self.models = []

    def fit(self, X, y):
        classes = np.unique(y)
        num_classes = len(classes)
        
        num_samples, num_features = X.shape

        for i in range(num_classes):
            # Create a binary classification problem for each class
            binary_labels = np.where(y == classes[i], 1, -1)

            # Define the variables
            w = cp.Variable(num_features)
            b = cp.Variable()

            # Define the objective function
            obj = cp.Minimize(0.5 * cp.norm(w, 2) + self.C * cp.sum(cp.pos(1 - cp.multiply(binary_labels, (X @ w + b)))))

            # Define the problem
            prob = cp.Problem(obj)

            # Solve the problem
            prob.solve()

            # Save the model parameters
            self.models.append((w.value, b.value))

    def predict(self, X):
        scores = []
        for w, b in self.models:
            scores.append(X @ w + b)
            # print(w, b)
        scores = np.array(scores).T
        return [np.argmax(scores, axis=1)]
