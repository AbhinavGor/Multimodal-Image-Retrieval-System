import pandas as pd
import numpy as np
import ast
import math

from sklearn.preprocessing import MinMaxScaler

from MultiClassSVM import MultiClassSVM
from output_plotter import output_plotter

# Load the data from the CSV file
try:
    df = pd.read_csv('task_4_output.csv')
except:
    print("Run task 4 before running task 5. Did not find 'task_4_output.csv'.")
print("1. SVM\n2. Probabilistic Method\n")
option = int(input("Select one from the above to perform re-ranking: "))

match option:
    case 1:
        df["FeatureVector"] = df["FeatureVector"].apply(ast.literal_eval)
        # Extract feature vectors and labels
        IDs = np.array(df['ImageID'].tolist())
        X = np.array(df['FeatureVector'].tolist())
        y = np.array(df['relevance'])

        # Map labels to integers for training the SVM
        label_mapping = {'R+': 0, 'R': 1, 'I': 2, 'I-': 3}
        y_numeric = np.array(
            [label_mapping[label] if label in label_mapping else -1 for label in y])

        # Filter rows with labels for training
        train_mask = y_numeric != -1
        X_train = X[train_mask]
        y_train = y_numeric[train_mask]
        print(y_train)
        print(X_train)
        # Filter rows without labels for testing
        test_mask = ~train_mask
        X_test = X[test_mask]
        y_test = y_numeric[test_mask]
        IDs = IDs[test_mask]
        print("X_test", X_test)
        # Create a SVM classifier
        clf = MultiClassSVM(C=2)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        classIds = [[] for i in range(4)]

        y_pred = list(y_pred[0])
        print(y_pred)
        for i in range(len(y_pred)):
            y_pred[i]
            classIds[int(y_pred[i])].append(IDs[i])

        result_ids = []
        for sublist in classIds:
            if isinstance(sublist, list):
                result_ids.extend(sublist)
            else:
                result_ids.append(sublist)

        output_plotter(result_ids)

    case 2:
        irrelevant_vector_representative = np.zeros(10)
        relevant_vectors = list(df[df["relevance"] ==
                                   "R+"]["FeatureVector"].apply(ast.literal_eval))

        for vector in df[df["relevance"] == "R"]["FeatureVector"].apply(ast.literal_eval):
            relevant_vectors.append(vector)
        relevant_vector_representative = np.zeros(len(relevant_vectors[0]))

        for vector in relevant_vectors:
            relevant_vector_representative += np.array(vector)
        irrelevant_vectors = df[df["relevance"] ==
                                "I-"]["FeatureVector"].apply(ast.literal_eval)
        for vector in df[df["relevance"] == "I"]["FeatureVector"].apply(ast.literal_eval):
            irrelevant_vectors.append(vector)

        irrelevant_vector_representative = np.zeros(len(relevant_vectors[0]))
        if len(irrelevant_vectors):
            irrelevant_vector_representative = np.zeros(
                len(irrelevant_vectors[0]))

            for vector in irrelevant_vectors:
                irrelevant_vector_representative += np.array(vector)
        else:
            print("No irrelevant vectors given!")

        unlabelled_vectors = list(
            df["FeatureVector"].apply(ast.literal_eval))
        unlabelled_ids = df["ImageID"]
        scaler = MinMaxScaler()

        relevant_distances = [-1*math.dist(relevant_vector_representative, np.array(
            vector)) for vector in unlabelled_vectors]

        relevant_distances_normalized_values = scaler.fit_transform(
            np.array(relevant_distances).reshape(-1, 1))

        irrelevant_distances = [math.dist(irrelevant_vector_representative, np.array(
            vector)) for vector in unlabelled_vectors]
        irrelevant_distances_normalized_values = scaler.fit_transform(
            np.array(irrelevant_distances).reshape(-1, 1))
        irrelevant_distances_normalized_values = np.array(
            irrelevant_distances_normalized_values).flatten() + 0.1
        irrelevant_distances_normalized_values = list(
            np.array(irrelevant_distances_normalized_values).flatten())
        relevant_distances_normalized_values = list(
            np.array(relevant_distances_normalized_values).flatten())

        for i in range(len(irrelevant_distances_normalized_values)):
            if irrelevant_distances_normalized_values[i] == 0:
                irrelevant_distances_normalized_values[i] += 0.01

        for i in range(len(relevant_distances_normalized_values)):
            if relevant_distances_normalized_values[i] == 0:
                relevant_distances_normalized_values[i] += 0.01
        # print(relevant_distances_normalized_values,
        #       irrelevant_distances_normalized_values)
        ranks = [-1*math.log(relevant_distances_normalized_values[i]/irrelevant_distances_normalized_values[i])
                 for i in range(len(relevant_distances_normalized_values))]
        ranks = np.array(ranks).reshape(-1, 1)

        image_rank_list = list(zip(unlabelled_ids, ranks))

        # Sort the list of tuples based on the second element (rank)
        sorted_image_rank_list = sorted(image_rank_list, key=lambda x: x[1])

        # Extract the sorted image IDs
        sorted_image_ids = [image_id for image_id, _ in sorted_image_rank_list]
        output_plotter(sorted_image_ids)
    case _:
        print("Please select a valid option!")
