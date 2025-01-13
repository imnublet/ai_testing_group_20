import numpy as np
import onnxruntime as ort
import random

import pandas as pd

print("Using an ONNX model with tabular/text data...")

# Replace this path with your ONNX model file
MODEL_PATH = 'model_1/model_1.onnx'
original_predictions = None
original_pred = None

# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
num_features = input_shape[1]  # Expected number of features
print(num_features)
# Define the evaluation function using the ONNX model
def evaluate_best(seeds, current_fitness, session):
    best_seed = seeds[0]  # Assume the first seed is the best initially
    best_fitness = current_fitness

    # Prepare input for the ONNX model
    input_name = session.get_inputs()[0].name
    data_batch = np.array(seeds, dtype=np.float32).reshape(-1, seeds[0].shape[1])  # Reshape to (n_samples, num_features)

    # Predict fitness using the ONNX model
    predictions = session.run(None, {input_name: data_batch})

    # The predictions should contain two outputs
    predicted_classes = predictions[0]  # First output: predicted class
    original_pred = predicted_classes
    probabilities = predictions[1]  # Second output: class probabilities

    # Evaluate fitness for each seed
    for idx, seed in enumerate(seeds):
        # Access the probabilities dictionary for the current seed
        prob_dict = probabilities[idx]  # Adjust based on how your model outputs the probabilities

        target_class = 0  # The class you are interested in mutating towards
        current_fitness = prob_dict.get(target_class, float('inf'))  # Get probability for the target class

        # Update best fitness based on your criteria
        if current_fitness < best_fitness:  # Assuming lower fitness is better
            best_fitness = current_fitness
            best_seed = seed

    return best_seed, best_fitness


if __name__ == "__main__":
    # Simulate loading tabular/text data (e.g., 10 features per sample)
    num_features = 315
    DATASET_PATH = 'data/investigation_train_large_checked.csv'  # original file from Brightspace, no changes
    data = pd.read_csv(DATASET_PATH)
    X = data.drop(columns=['Ja', 'Nee', 'checked'])
    original_data = X.iloc[2].values.reshape(1, -1).astype(np.float32)
    print("Original data:", original_data)

    # Initial seed (starting solution)
    seed = original_data.copy()

    # Number of iterations for optimization
    num_iterations = 400

    # Current fitness (using the ONNX model's predict method)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: seed})

    probabilities = output[1][0]  # Get the first element of the list
    original_predictions = probabilities
    original_pred = output[0]
    target_class = 0  # The class you want to mutate towards
    fitness = probabilities[target_class]  # Use the target class probability as fitness
    print(f"Initial fitness for target class {target_class}: {fitness}")
    for iteration in range(1, num_iterations + 1):
        print(f"Iteration: {iteration}")
        new_seed_plus = seed.copy()
        new_seed_minus = seed.copy()

        # Mutate one feature
        feature_index = random.randint(0, num_features - 1)

        # Apply mutation (e.g., ±30% of the original value)
        epsilon = 0.30
        best_seed = None
        best_fitness = None

        if seed[0][feature_index] == 0 or seed[0][feature_index] == 1:
            if seed[0][feature_index] == 0: #assumes boolean
                new_seed_plus[0][feature_index] = 1
            else:
                new_seed_plus[0][feature_index] = 0

            # Evaluate the original seed and both neighbors
            best_seed, best_fitness = evaluate_best(
                [seed, new_seed_plus], fitness, session
            )

        else: #Not boolean
            delta = seed[0][feature_index] * epsilon

            # Apply mutations
            new_seed_plus[0][feature_index] += np.ceil(delta)
            new_seed_minus[0][feature_index] -= np.floor(delta)

            # Evaluate the original seed and both neighbors
            best_seed, best_fitness = evaluate_best(
                [seed, new_seed_plus, new_seed_minus], fitness, session
            )

        # Update the current best seed and fitness
        if best_fitness < fitness:  # Adjust condition based on your fitness goal
            seed = best_seed
            fitness = best_fitness
            print("New fitness value:", fitness)

        if fitness < 0.5:
            break

    # Final result
    final_output = session.run(None, {input_name: seed})
    final_probabilities = final_output[1][0]  # Get the final probabilities
    print("Optimized data:", seed)
    print("Final fitness value:", fitness)
    print("Original predictions", original_pred)  # Print final predicted classes
    print("Final predictions:", final_output[0])  # Print final predicted classes
    print("Original probabilities:", original_predictions)
    print("Final probabilities:", final_probabilities)  # Print final probabilities
