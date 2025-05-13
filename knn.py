import numpy as np
from collections import Counter
import pandas as pd # It's good practice to import pandas if you expect DataFrames

# Use the predefined num_features as requested
num_features = 5 # Set this to the correct number of features for your data
                 # For your data: acidity, sweetness, alcohol

def weighted_euclidean_distance_matrix(test_features, train_features, weights):
    """
    Calculates the weighted Euclidean distance between test and train features.
    Expects test_features and train_features to be NumPy arrays.
    """
    weights = np.array(weights)
    if weights.shape[0] != num_features:
        raise ValueError(f"Weights array must have {num_features} elements, got {weights.shape[0]}.")
    # These checks assume features are already extracted and are numpy arrays
    if test_features.shape[1] != num_features or train_features.shape[1] != num_features:
        raise ValueError(f"Feature columns in test_features ({test_features.shape[1]}) or train_features ({train_features.shape[1]}) do not match num_features ({num_features}).")

    weighted_diff = (test_features[:, None, :] - train_features[None, :, :]) / weights[None, None, :]
    return np.sqrt((weighted_diff ** 2).sum(axis=2))


def knn_predict(train_data, query_data, k=3, weights=None):
    """
    Predicts labels for query_data using kNN with optional feature weights.
    train_data and query_data can be pandas DataFrames or NumPy arrays.
    """
    # Assumes global num_features is correctly set

    # Convert pandas DataFrame to NumPy array using .iloc and .values
    if isinstance(train_data, pd.DataFrame):
        train_features = train_data.iloc[:, :num_features].values.astype(float)
        train_labels = train_data.iloc[:, num_features].values
    else: # Assumes NumPy array
        train_features = train_data[:, :num_features].astype(float)
        train_labels = train_data[:, num_features]

    if isinstance(query_data, pd.DataFrame):
        query_features = query_data.iloc[:, :num_features].values.astype(float)
        true_labels = query_data.iloc[:, num_features].values
    else: # Assumes NumPy array
        query_features = query_data[:, :num_features].astype(float)
        true_labels = query_data[:, num_features]


    if weights is None:
        weights = np.ones(num_features)
    else:
        weights = np.array(weights)
        if len(weights) != num_features:
            raise ValueError(f"Length of weights array must be equal to num_features ({num_features})")
        if np.any(weights <= 0):
            print("Warning: Weights should be positive. Adjusting non-positive weights to a small value (1e-6).")
            weights[weights <= 0] = 1e-6

    # weighted_euclidean_distance_matrix expects NumPy arrays
    dists = weighted_euclidean_distance_matrix(query_features, train_features, weights)

    predictions = []
    for i in range(len(query_features)):
        k_indices = np.argsort(dists[i])[:k]
        k_labels = train_labels[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions), true_labels

def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

def optimize_weights_random_search(train_data_for_knn, validation_data_for_opt, k=3, num_trials=100, weight_range=(0.1, 10.0)):
    """
    Optimizes feature weights using random search on the validation_data_for_opt.
    The kNN model itself is trained using train_data_for_knn for each trial.
    """
    best_weights = np.ones(num_features) # Assumes global num_features
    best_accuracy = 0.0

    print(f"Starting random search for optimal weights ({num_trials} trials)...")
    for i in range(num_trials):
        current_weights = np.random.uniform(weight_range[0], weight_range[1], num_features)
        current_weights[current_weights <= 1e-6] = 1e-6

        # knn_predict can handle if train_data_for_knn and validation_data_for_opt are DataFrames or NumPy arrays
        predictions, true_labels = knn_predict(train_data_for_knn, validation_data_for_opt, k, weights=current_weights)
        accuracy = calculate_accuracy(predictions, true_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = current_weights
        # Adjusted print frequency for potentially shorter num_trials:
        if (i + 1) % (max(1, num_trials // 10)) == 0 or i == num_trials -1 :
            print(f"Trial {i+1}/{num_trials}: Weights: {current_weights}, Accuracy: {accuracy:.4f}")
            print(f"Trial {i+1}/{num_trials} processed. Current best accuracy on validation: {best_accuracy:.4f}")

    print(f"Finished random search. Best accuracy on validation set: {best_accuracy:.4f}")
    return best_weights, best_accuracy

# --- Example Usage ---
if __name__ == "__main__":
    print(f"Using predefined number of features (num_features): {num_features}")

    if num_features <= 0:
        raise ValueError("Global num_features must be positive and correctly set at the top of the script.")

    train_dataset, validation_dataset, test_dataset = None, None, None # Initialize
    try:
        # It's assumed that 'data.py' provides pandas DataFrames
        # If they are NumPy arrays, the .iloc logic in knn_predict will use the 'else' branch
        from data import train as train_dataset_orig, validation as validation_dataset_orig, test as test_dataset_orig
        print("Successfully loaded train, validation, and test datasets from 'data' module.")

        # If you want to ensure they are DataFrames for this example based on the error:
        # This is just for clarity if 'data' module could return either.
        # If you know 'data' returns DataFrames, this explicit conversion isn't strictly needed
        # but makes the type explicit.
        train_dataset = pd.DataFrame(train_dataset_orig)
        validation_dataset = pd.DataFrame(validation_dataset_orig)
        test_dataset = pd.DataFrame(test_dataset_orig)


        min_cols_required = num_features + 1
        # shape[1] works for both DataFrames and NumPy arrays
        if train_dataset.shape[1] < min_cols_required:
            raise ValueError(f"Train data has {train_dataset.shape[1]} columns, but num_features={num_features} requires at least {min_cols_required} columns (features + label).")
        if validation_dataset.shape[1] < min_cols_required:
            raise ValueError(f"Validation data has {validation_dataset.shape[1]} columns, but num_features={num_features} requires at least {min_cols_required} columns.")
        if test_dataset.shape[1] < min_cols_required:
            raise ValueError(f"Test data has {test_dataset.shape[1]} columns, but num_features={num_features} requires at least {min_cols_required} columns.")

    except ImportError:
        print("Error: Could not import train, validation, or test from 'data' module.")
        print("Please ensure 'data.py' exists and contains these numpy arrays or pandas DataFrames.")
    except ValueError as e:
        print(f"Data loading or validation error: {e}")
    except Exception as e: # Catch any other unexpected error during data loading/conversion
        print(f"An unexpected error occurred during data loading: {e}")


    if train_dataset is not None and validation_dataset is not None and test_dataset is not None:
        k_val = 5

        print("\n--- kNN with default weights (on TEST set) ---")
        predictions_default, true_labels_default = knn_predict(train_dataset, test_dataset, k=k_val)
        accuracy_default = calculate_accuracy(predictions_default, true_labels_default)
        print("Predictions (default weights):", predictions_default)
        print("True labels (default weights):", true_labels_default)
        print(f"Accuracy on TEST set (default weights): {accuracy_default:.4f}")

        print("\n--- Optimizing weights (using TRAIN and VALIDATION sets) ---")
        optimized_weights, best_val_accuracy = optimize_weights_random_search(
            train_dataset,
            validation_dataset,
            k=k_val,
            num_trials=5000, # Original value
            weight_range=(0.1, 10)
        )
        print(f"\nOptimized weights found: {optimized_weights} (achieved {best_val_accuracy:.4f} on VALIDATION set)")

        print("\n--- kNN with optimized weights (on TEST set) ---")
        predictions_optimized, true_labels_optimized = knn_predict(train_dataset, test_dataset, k=k_val, weights=optimized_weights)
        accuracy_optimized = calculate_accuracy(predictions_optimized, true_labels_optimized)

        print("Predictions (optimized weights):", predictions_optimized)
        print("True labels (optimized weights):", true_labels_optimized)
        print(f"Accuracy on TEST set (optimized weights): {accuracy_optimized:.4f}")

        # Append the weights and the accuracy to the results file
        with open("results.txt", "a") as f:
            f.write(f"Optimized weights: {optimized_weights}, Accuracy: {accuracy_optimized:.4f}\n")
        print("\n--- Results saved to results.txt ---")

        if accuracy_optimized > accuracy_default:
            print("\nOptimized weights resulted in a higher accuracy on the test set.")
        elif accuracy_optimized < accuracy_default:
            print("\nOptimized weights resulted in a lower accuracy on the test set...")
        else:
            print("\nOptimized weights resulted in the same accuracy on the test set.")
    else:
        print("Execution cannot continue as data was not properly loaded or validated.")
