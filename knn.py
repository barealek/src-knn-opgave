import numpy as np
from collections import Counter
import pandas as pd # It's good practice to import pandas if you expect DataFrames
import math, random

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
        raise ValueError(f"Der skal være {num_features} vægte, fik {weights.shape[0]}.")
    # Features er et numpy array
    if test_features.shape[1] != num_features or train_features.shape[1] != num_features:
        raise ValueError(f"Feature kolonner i test eller train features matcher ikke num_features ({num_features}).")

    # Beregn afstanden med numpy broadcasting og division med vægtene
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

def optimize_weights_simulated_annealing(train_data_for_knn, validation_data_for_opt, k=3,
                                        initial_weights=None, initial_temp=1.0, cooling_rate=0.95,
                                        n_iterations=5000, step_size=0.1):
    """
    Optimizes feature weights using a simulated annealing-inspired approach.
    """
    num_features = train_data_for_knn.shape[1] - 1 if isinstance(train_data_for_knn, np.ndarray) else train_data_for_knn.shape[1] - 1
    if initial_weights is None:
        current_weights = np.random.uniform(0.1, 10.0, num_features)
    else:
        current_weights = np.array(initial_weights)

    best_weights = np.copy(current_weights)
    best_accuracy = calculate_accuracy(*knn_predict(train_data_for_knn, validation_data_for_opt, k, weights=best_weights))
    current_accuracy = best_accuracy
    temperature = initial_temp

    print("Starting Simulated Annealing for weight optimization...")

    for i in range(n_iterations):
        # Generate a new set of weights by making small random adjustments
        new_weights = current_weights + np.random.normal(0, step_size * temperature, num_features)
        new_weights[new_weights <= 1e-6] = 1e-6 # Ensure weights remain positive

        new_accuracy = calculate_accuracy(*knn_predict(train_data_for_knn, validation_data_for_opt, k, weights=new_weights))

        # Calculate the change in accuracy
        delta_accuracy = new_accuracy - current_accuracy

        # Acceptance probability
        if delta_accuracy > 0:
            # Accept better solutions
            current_accuracy = new_accuracy
            current_weights = new_weights
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weights = np.copy(current_weights)
        else:
            # Accept worse solutions with a probability based on temperature
            probability = math.exp(delta_accuracy / temperature)
            if random.random() < probability:
                current_accuracy = new_accuracy
                current_weights = new_weights

        # Cool down the temperature
        temperature *= cooling_rate

        if (i + 1) % (max(1, n_iterations // 50)) == 0 or i == n_iterations - 1:
            print(f"Iteration {i+1}/{n_iterations}: Temp: {temperature:.4f}, Current Accuracy: {current_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

    print(f"Finished Simulated Annealing. Best accuracy on validation set: {best_accuracy:.4f}")
    print(f"Best weights found: {best_weights}")
    return best_weights, best_accuracy

def optimize_weights_random_search(train_data_for_knn,
                                   validation_data_for_opt,
                                   k=3, num_trials=100,
                                   weight_range=(0.1, 10.0)):

    best_weights = np.ones(num_features) # Vi starter med et array af 1'ere
    best_accuracy = 0.0

    print(f"Starting random search for optimal weights ({num_trials} trials)...")
    for i in range(num_trials):
        # Vi genererer tilfældige vægte inden for det angivne interval
        current_weights = np.random.uniform(weight_range[0], weight_range[1], num_features)
        current_weights[current_weights <= 1e-6] = 1e-6

        # Vi prøver at lave kNN-forespørgsler med den nye vægte
        predictions, true_labels = knn_predict(train_data_for_knn, validation_data_for_opt, k, weights=current_weights)
        accuracy = calculate_accuracy(predictions, true_labels)

        # Vi tjekker om den nuværende vægtkombination er bedre end den bedste hidtil
        if accuracy > best_accuracy:
            # Hvis accuracy er bedre, opdaterer vi den bedste vægt og den bedste nøjagtighed
            best_accuracy = accuracy
            best_weights = current_weights

        # Print status for hver 10% af trials
        if (i + 1) % (num_trials // 10) == 0 or i == num_trials -1 :
            print(f"Trial {i+1}/{num_trials} processed. Current best accuracy on validation: {best_accuracy:.4f}")

    # Print den bedste vægt og den bedste nøjagtighed
    print(f"Finished random search. Best accuracy on validation set: {best_accuracy:.4f}")
    return best_weights, best_accuracy

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

        # Vi dobbelttjekker lige, at dataen er i det rigtige format
        train_dataset = pd.DataFrame(train_dataset_orig)
        validation_dataset = pd.DataFrame(validation_dataset_orig)
        test_dataset = pd.DataFrame(test_dataset_orig)


        # Vi tjekker at dataen er den rigtige størrelse
        min_cols_required = num_features + 1
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
        k_val = 7

        print("\n--- kNN with default weights (on TEST set) ---")
        predictions_default, true_labels_default = knn_predict(train_dataset, test_dataset, k=k_val)
        accuracy_default = calculate_accuracy(predictions_default, true_labels_default)
        print("Predictions (default weights):", predictions_default)
        print("True labels (default weights):", true_labels_default)
        print(f"Accuracy on TEST set (default weights): {accuracy_default:.4f}")

        print("\n--- Optimizing weights (using TRAIN and VALIDATION sets) ---")
        optimized_weights, best_val_accuracy = optimize_weights_simulated_annealing(
            train_dataset,
            validation_dataset,
            k=k_val,
            n_iterations=10000,
            initial_temp=3,
            cooling_rate=.9995,
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
            f.write(f"Optimized weights: {optimized_weights}, Accuracy: {accuracy_optimized:.4f}  Acc. no weights: {accuracy_default:.4f}")
            if accuracy_optimized > accuracy_default:
                f.write(" (Optimized weights improved accuracy)\n")
            else:
                f.write(" (Optimized weights did not improve accuracy)\n")
        print("\n--- Results saved to results.txt ---")

        if accuracy_optimized > accuracy_default:
            print("\nOptimized weights resulted in a higher accuracy on the test set.")
        elif accuracy_optimized < accuracy_default:
            print("\nOptimized weights resulted in a lower accuracy on the test set...")
        else:
            print("\nOptimized weights resulted in the same accuracy on the test set.")
    else:
        print("Execution cannot continue as data was not properly loaded or validated.")
