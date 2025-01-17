import numpy as np
import onnxruntime as ort
import random
import matplotlib.pyplot as plt
from statistics import mean

import pandas as pd

print("Using an ONNX model with tabular/text data...")

# Replace this path with your ONNX model file
MODEL_PATH = 'For Group 20/model_1/model_1.onnx'
original_predictions = None
original_pred = None

# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
num_features = input_shape[1]  # Expected number of features
print(num_features)

# Load model 2
MODEL_PATH_2 = 'For Group 20/model_2/model_2.onnx'
original_predictions_2 = None
original_pred_2 = None

# Load the ONNX model
session_2 = ort.InferenceSession(MODEL_PATH_2)
input_name_2 = session_2.get_inputs()[0].name
input_shape_2 = session_2.get_inputs()[0].shape
num_features_2 = input_shape_2[1]  # Expected number of features

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

def mutate(seed, session):
    #original_data = data
    # Initial seed (starting solution)
    #seed = data.copy()

    # Number of iterations for optimization
    num_iterations = 3 # start at 10?

    # Current fitness (using the ONNX model's predict method)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: seed})

    probabilities = output[1][0]  # Get the first element of the list
    original_predictions = probabilities
    original_pred = output[0]
    target_class = 0  # The class you want to mutate towards
    fitness = probabilities[target_class]  # Use the target class probability as fitness
    print(f"Initial fitness for target class {target_class}: {fitness}")
    
    iteration = 0 # Inner loop iteration number
    fitness_list = [] # Keeps track of best fitness
    features_list = [] # Keeps track of the randomly selected features
        
    while fitness > 0.50:
        print(f"Iteration: {iteration}")
        prev_seed = seed # Previous best seed
        new_seed_plus = seed.copy()
        new_seed_minus = seed.copy()

        # Mutate one feature
        feature_index = random.randint(0, num_features - 1)
        features_list.append(feature_index)

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
            
        fitness_list.append(best_fitness)
        iteration += 1
    #fitness_dict[iteration] = fitness_list
    fitness_feature_dict = dict(zip(features_list, fitness_list))
    return fitness_feature_dict

def visualize_fitness(dict): 
    # Create a separate subplot for each outer key
    num_subplots = len(dict)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), squeeze=False)  
    for i, (outer_key, inner_dict) in enumerate(dict.items()):
        ax = axes[i][0] 
        print(inner_dict.items())
        features, fitnesses = zip(*inner_dict.items())
        fitness_list = list(fitnesses)

        ax.plot(fitness_list)
        
        ax.set_title(f"Iteration {outer_key}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid()

    # Adjust layout
    #plt.tight_layout()
    plt.show()

def visualize_mean_iterations(dict_1, dict_2): 
    total_iters_1 = []
    total_iters_2 = []

    for i, (outer_key, inner_dict) in enumerate(dict_1.items()): # feature + fitness of model 1
        total_iters_1.append(len(inner_dict.items()))
    for i, (outer_key, inner_dict) in enumerate(dict_2.items()): # feature + fitness of model 2
        total_iters_2.append(len(inner_dict.items()))

    box_colors = ['skyblue', 'plum']  

    box_plot = plt.boxplot([total_iters_1, total_iters_2], patch_artist=True, meanline=True, showmeans=True)

    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)

    plt.title("Iterations untill Label Change")
    plt.xticks([1,2], ['Model 1', 'Model 2'])
    plt.ylabel("Iteration")
    plt.show()

def find_influential_features(dict, session, threshold): # Gather feature value mutations that caused change of at least 'threshold' in fitness
    # for each item in main dict, subtract current value from previous (take absolute value) 
    feature_list = [] 
    major_differences = {}
    for i, (outer_key, inner_dict) in enumerate(dict.items()): 
        for i, (feature, fitness) in enumerate(inner_dict.items()):
            if i == 0:
                prev = fitness
                continue
            else:
                difference = abs(fitness-prev)
                # Compare difference to threshold, if equal or larger then save feature + difference 
                if difference >= threshold:
                    #print("fitness before sub: ", fitness)
                    #print("prev fitness: ", prev)
                    #print("difference: ", difference)
                    feature_list.append(feature)

                    # Add feature and corresponding change in fitness to a dict 
                    if feature not in major_differences.keys():
                        major_differences[feature] = [difference]
                    else:
                        major_differences[feature].append(difference)
                        major_differences[feature].sort(reverse = True)
                prev = fitness
    print("major differences: ", major_differences)
    return major_differences

def visualize_influential_features(dict, data, top_n, model_name): # model_name is 'Model 1' or 'Model 2'
    # Count occurences of each feature that passed threshold along with effect on fitness
    dict = {feature: (len(differences), mean(differences)) for feature, differences in dict.items()}

    sorted_by_occurrences = sorted(dict.items(), key=lambda x: x[1][0], reverse=True)
    sorted_by_avg_score = sorted(dict.items(), key=lambda x: x[1][1], reverse=True)
    
    # Sort in descending order and select top_n features
    top_occurrences = sorted_by_occurrences[:top_n]
    top_avg_scores = sorted_by_avg_score[:top_n]
    keys_occurrences, values_occurrences = zip(*[(k, v[0]) for k, v in top_occurrences])
    keys_avg_scores, values_avg_scores = zip(*[(k, v[1]) for k, v in top_avg_scores])

    # Get column names
    columns_occurrences = [data.columns[k] for k in keys_occurrences]
    columns_mean_diff = [data.columns[k] for k in keys_avg_scores]
    
    # Plot the top-N by occurrences
    #plt.figure(figsize=(12, 6))
    plt.barh(columns_occurrences, values_occurrences, color='cadetblue')
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.title(f"Top-{top_n} Influential Features in {model_name} by Occurrences")
    plt.xlabel("Occurences")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    # Plot the top-N by mean scores
    #plt.figure(figsize=(12, 6))
    plt.barh(columns_mean_diff, values_avg_scores, color='palevioletred')
    plt.title(f"Top-{top_n} Influential Features in {model_name} by Mean Scores") # TODO: choose a more descriptive title
    plt.xlabel("Mean Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simulate loading tabular/text data (e.g., 10 features per sample)
    num_features = num_features
    DATASET_PATH = 'data/investigation_train_large_checked.csv'  # original file from Brightspace, no changes
    data = pd.read_csv(DATASET_PATH)
    X = data.drop(columns=['Ja', 'Nee', 'checked'])
    original_data = X.iloc[100].values.reshape(1, -1).astype(np.float32)
    print("Original data:", original_data)
    
    # Initial seed (starting solution)
    seed = original_data.copy()

    num_iterations = 5
    model1_fitness_feature_dict = {}
    model2_fitness_feature_dict = {}
    for i in range(1, num_iterations+1):
        model1_fitness_list = mutate(original_data, session)
        model1_fitness_feature_dict[i] = model1_fitness_list
        model2_fitness_list = mutate(original_data, session_2) # TODO: apply same mutations to model 2 as to model 1
        model2_fitness_feature_dict[i] = model2_fitness_list

    #visualize_fitness(fitness_feature_dict) # Don't run this without changing num_iterations to low number!
    visualize_mean_iterations(model1_fitness_feature_dict, model2_fitness_feature_dict)
    major_differences = find_influential_features(model1_fitness_feature_dict, session, 0.2)
    visualize_influential_features(dict=major_differences, data=data, top_n=10, model_name="Model 1")

    major_differences_2 = find_influential_features(model2_fitness_feature_dict, session_2, 0.2)
    visualize_influential_features(dict=major_differences_2, data=data, top_n=10, model_name="Model 2")
    
    # Final result
    final_output = session.run(None, {input_name: seed})
    print("final output: ", final_output)
    final_probabilities = final_output[1][0]  # Get the final probabilities
