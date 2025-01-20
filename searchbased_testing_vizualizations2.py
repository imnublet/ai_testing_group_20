import random

import numpy as np
import onnxruntime as ort
import pandas as pd

print("Using an ONNX model with tabular/text data...")
np.set_printoptions(suppress=True, precision=0, floatmode='fixed')

# Replace this path with your ONNX model file
MODEL_PATH_1 = 'model_1/model_1.onnx'
MODEL_PATH_2 = 'model_2/model_2.onnx'


# Define the evaluation function using the ONNX model
def evaluate_best(seeds, current_fitness, session):
    best_seed = seeds[0]  # Assume the first seed is the best initially
    best_fitness = current_fitness

    # Prepare input for the ONNX model
    input_name = session.get_inputs()[0].name
    data_batch = np.array(seeds, dtype=np.float32).reshape(-1,
                                                           seeds[0].shape[1])  # Reshape to (n_samples, num_features)

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


def search_based_testing(session, seed, features_to_mutate):
    # Number of iterations for optimization
    num_iterations = 20
    # Current fitness (using the ONNX model's predict method)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: seed})

    probabilities = output[1][0]  # Get the first element of the list
    target_class = 0  # The class you want to mutate towards
    fitness = probabilities[target_class]  # Use the target class probability as fitness
    print(f"Initial fitness for target class {target_class}: {fitness}")
    final_iteration = 0
    for iteration in range(1, num_iterations + 1):
        final_iteration = final_iteration + 1
        # print(f"Iteration: {iteration}")
        new_seed_plus = seed.copy()
        new_seed_minus = seed.copy()

        # Mutate one feature
        feature_index = random.randint(0, len(features_to_mutate) - 1)
        feature_index = features_to_mutate[feature_index]

        # Apply mutation (e.g., ±30% of the original value)
        epsilon = 0.20
        best_seed = None
        best_fitness = None

        if seed[0][feature_index] == 0.0 or seed[0][feature_index] == 1.0:
            if seed[0][feature_index] == 0.0:  # assumes boolean
                new_seed_plus[0][feature_index] = 1.0
            else:
                new_seed_plus[0][feature_index] = 0.0

            # Evaluate the original seed and both neighbors
            best_seed, best_fitness = evaluate_best(
                [seed, new_seed_plus], fitness, session
            )

        else:  # Not boolean
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
            print("Number of iteration ", final_iteration)
            break

    print("Number of iteration ", final_iteration )
    return fitness, seed


def run_search_based_on_both_models(model_path_1, model_path_2):
    # Load the ONNX model
    session_1 = ort.InferenceSession(model_path_1)
    session_2 = ort.InferenceSession(model_path_2)

    input_name = session_1.get_inputs()[0].name
    input_shape = session_1.get_inputs()[0].shape
    num_features = input_shape[1]  # Expected number of features
    print(num_features)

    # Simulate loading tabular/text data (e.g., 10 features per sample)
    DATASET_PATH = 'data/investigation_train_large_checked.csv'  # original file from Brightspace, no changes
    data = pd.read_csv(DATASET_PATH)
    X = data.drop(columns=['Ja', 'Nee', 'checked'])

    random_selected_data = random.sample(range(len(data)), int(len(data) * 0.05))
    i = 0

    amount_flipped_model_1 = 0
    amount_flipped_model_2 = 0
    for x in random_selected_data:

        print(" ---- Run " + str(i) + " out of " + str(len(data)*0.2) + "----")
        i = i +1
        original_data = X.iloc[x].values.reshape(1, -1).astype(np.float32)
        #print("Original data:", original_data)

        # Initial seed (starting solution)
        seed = original_data.copy()

        # Current fitness (using the ONNX model's predict method)
        # feature_index = data.columns.get_loc('relatie_kind_heeft_kinderen')
        # print(feature_index)
        # print(data.columns[feature_index])
        #
        # features_to_mutate = [feature_index]

        features_to_mutate = [index for index, column in enumerate(data.columns) if
                              "adres" in column]

        # Print the indices and corresponding column names
        #print(features_to_mutate)
        #print([data.columns[index] for index in features_to_mutate])
        print("Model 1:")

        output = session_1.run(None, {input_name: seed})
        fitness1, seed1 = search_based_testing(session_1, seed, features_to_mutate)
        final_output = session_1.run(None, {input_name: seed1})
        #print("Optimized data 1:", seed1)
        print("Final fitness value 1:", fitness1)
        print("Original predictions 1", output[0])  # Print final predicted classes
        print("Final predictions 1:", final_output[0])  # Print final predicted classes
        print("Original probabilities 1:", output[1][0])
        print("Final probabilities 1:", final_output[1][0])  # Print final probabilities
        if output[0] != final_output[0]:
            amount_flipped_model_1 += 1
        print("Model 2:")

        output = session_2.run(None, {input_name: seed})
        fitness2, seed2 = search_based_testing(session_2, seed, features_to_mutate)
        final_output = session_2.run(None, {input_name: seed2})
        #print("Optimized data 2:", seed2)
        print("Final fitness value 2:", fitness2)
        print("Original predictions 2", output[0])  # Print final predicted classes
        print("Final predictions 2:", final_output[0])  # Print final predicted classes
        print("Original probabilities 2:", output[1][0])
        print("Final probabilities 2:", final_output[1][0])  # Print final probabilities
        if output[0] != final_output[0]:
            amount_flipped_model_2 += 1

    print("Amount flipped model 1", amount_flipped_model_1)
    print("Amount flipped model 2", amount_flipped_model_2)



# Final result
run_search_based_on_both_models(MODEL_PATH_1, MODEL_PATH_2)
