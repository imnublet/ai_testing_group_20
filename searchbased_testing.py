import numpy as np
import onnxruntime as ort
import random

print("Using an ONNX model with tabular/text data...")

# Replace this path with your ONNX model file
MODEL_PATH = 'model_1/model_1.onnx'

# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
num_features = input_shape[1]  # Expected number of features

# Define the evaluation function using the ONNX model
def evaluate_best(seeds, current_fitness, session):
    best_seed = seeds[0]  # Assume the first seed is the best initially
    best_fitness = current_fitness

    # Prepare input for the ONNX model
    input_name = session.get_inputs()[0].name
    data_batch = np.array(seeds, dtype=int)  # Ensure data type matches model input

    # Predict fitness using the ONNX model
    predictions = session.run(None, {input_name: data_batch})[0]

    # Evaluate fitness for each seed (modify fitness comparison as needed)
    for idx, prediction in enumerate(predictions):
        # Assuming lower prediction values indicate better fitness
        if prediction[0] < best_fitness:  # Adjust indexing for your model's output shape
            best_fitness = prediction[0]
            best_seed = seeds[idx]

    return best_seed, best_fitness

if __name__ == "__main__":
    # Simulate loading tabular/text data (e.g., 10 features per sample)
    num_features = 10
    original_data = np.random.rand(1, num_features).astype(np.float32)  # Replace with your actual data
    print("Original data:", original_data)

    # Initial seed (starting solution)
    seed = original_data.copy()

    # Number of iterations for optimization
    num_iterations = 300

    # Current fitness (using the ONNX model's predict method)
    input_name = session.get_inputs()[0].name
    fitness = session.run(None, {input_name: seed})[0][0][0]  # Adjust indexing based on model's output shape

    for iteration in range(1, num_iterations + 1):
        print(f"Iteration: {iteration}")
        new_seed_plus = seed.copy()
        new_seed_minus = seed.copy()

        # Mutate one feature
        feature_index = random.randint(0, num_features - 1)

        # Apply mutation (e.g., ±10% of the original value)
        epsilon = 0.10
        delta = seed[0][feature_index] * epsilon

        # Apply mutations
        new_seed_plus[0][feature_index] += delta
        new_seed_minus[0][feature_index] -= delta

        # Evaluate the original seed and both neighbors
        best_seed, best_fitness = evaluate_best(
            [seed, new_seed_plus, new_seed_minus], fitness, session
        )

        # Update the current best seed and fitness
        if best_fitness < fitness:  # Adjust condition based on your fitness goal
            seed = best_seed
            fitness = best_fitness
            print("New fitness value:", fitness)

    # Final result
    print("Optimized data:", seed)
    print("Final fitness value:", fitness)
