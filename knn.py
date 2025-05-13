import numpy as np
from collections import Counter

num_features = 2

def euclidean_distance_matrix(test_features, train_features):
    # Vectorized Euclidean distance calculation
    return np.sqrt(((test_features[:, None, :] - train_features[None, :, :]) ** 2).sum(axis=2))

def knn_predict(train_data, test_data, k=3):
    # Extract features and labels
    train_features = train_data[:, :num_features].astype(float)
    train_labels = train_data[:, num_features]

    test_features = test_data[:, :num_features].astype(float)
    test_labels = test_data[:, num_features]

    # Compute distances
    dists = euclidean_distance_matrix(test_features, train_features)

    predictions = []
    for i in range(len(test_features)):
        k_indices = np.argsort(dists[i])[:k]
        k_labels = train_labels[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions), test_labels

def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

# Example usage
from data import train, test

# Run classifier
k = 3
predictions, true_labels = knn_predict(train, test, k)
accuracy = calculate_accuracy(predictions, true_labels)

# Output
print("Predictions:", predictions)
print("True labels:", true_labels)
print("Accuracy:", accuracy)
