import random

import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

print("Using an ONNX model with tabular/text data...")
np.set_printoptions(suppress=True, precision=0, floatmode='fixed')

# Replace this path with your ONNX model file
MODEL_PATH_1 = 'model_1/model_1.onnx'
MODEL_PATH_2 = 'model_2/model_2.onnx'

feature_labels = pd.read_csv('data/feature_labels_v2.csv')
# print(len(feature_labels))
discriminatory_features = feature_labels[feature_labels['Category'] == 'D']
# print(len(discriminatory_features))

subjective_features = feature_labels[feature_labels['Category'] == 'S']
# print(len(subjective_features))

non_relevant_features = feature_labels[feature_labels['Category'] == 'NR']
# print(len(non_relevant_features))

unclear_features = feature_labels[feature_labels['Category'] == 'U']
# print(len(unclear_features))


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
    num_iterations = 30
    # Current fitness (using the ONNX model's predict method)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: seed})

    probabilities = output[1][0]  # Get the first element of the list
    target_class = 0  # The class you want to mutate towards
    fitness = probabilities[target_class]  # Use the target class probability as fitness
    print(f"Initial fitness for target class {target_class}: {fitness}")
    final_iteration = 0
    for iteration in range(1, num_iterations + 1):
        print(final_iteration)
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

    print("Number of iteration ", final_iteration)
    return final_iteration, fitness, seed


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

    random_selected_data = random.sample(range(len(data)), int(len(data) * 0.010))
    i = 0
    MAX_ITERATIONS = 30
    amount_flipped_model_1 = 0
    amount_flipped_model_2 = 0
    amount_iterations_1 = []  # Number of iterations it took to make a change
    probability_difference_1 = []  # Difference in probability before and after
    amount_iterations_2 = []  # Number of iterations it took to make a change
    probability_difference_2 = []  # Difference in probability before and after
    for x in random_selected_data:

        print(" ---- Run " + str(i) + " out of " + str(len(data) * 0.2) + "----")
        i = i + 1
        original_data = X.iloc[x].values.reshape(1, -1).astype(np.float32)
        # print("Original data:", original_data)

        # Initial seed (starting solution)
        seed = original_data.copy()

        # Current fitness (using the ONNX model's predict method)
        # feature_index = data.columns.get_loc('relatie_kind_heeft_kinderen')
        # print(feature_index)
        # print(data.columns[feature_index])
        #
        # features_to_mutate = [feature_index]

        # features_to_mutate = [index for index, column in enumerate(data.columns) if
        #                       "_" in column]

        #features_to_mutate = [index for index, column in enumerate(discriminatory_features)]

        # print(discriminatory_features)
        #
        # print(discriminatory_features._data)
        # features_to_mutate = [index for index, column in enumerate(data.columns) if
        #                       discriminatory_features['Feature'].tolist() in column]

        features_to_mutate = discriminatory_features.index.tolist()
        print(discriminatory_features['Feature'].tolist())
        print(features_to_mutate)

        print("LENGTH OF FEATURS TO MUTATE")
        print(len(features_to_mutate))


        # Print the indices and corresponding column names
        # print(features_to_mutate)
        # print([data.columns[index] for index in features_to_mutate])
        print("Model 1:")

        output = session_1.run(None, {input_name: seed})
        iterations1, fitness1, seed1 = search_based_testing(session_1, seed, features_to_mutate)
        final_output = session_1.run(None, {input_name: seed1})
        # print("Optimized data 1:", seed1)
        print("Final fitness value 1:", fitness1)
        print("Original predictions 1", output[0])  # Print final predicted classes
        print("Final predictions 1:", final_output[0])  # Print final predicted classes
        print("Original probabilities 1:", output[1][0])
        print("Final probabilities 1:", final_output[1][0])  # Print final probabilities
        if output[0] != final_output[0]:
            amount_flipped_model_1 += 1
        amount_iterations_1.append(iterations1)

        probability_difference_1.append(np.abs(final_output[1][0][0] - output[1][0][0]))
        print("Model 2:")

        output = session_2.run(None, {input_name: seed})
        iterations2, fitness2, seed2 = search_based_testing(session_2, seed, features_to_mutate)
        final_output = session_2.run(None, {input_name: seed2})
        # print("Optimized data 2:", seed2)
        print("Final fitness value 2:", fitness2)
        print("Original predictions 2", output[0])  # Print final predicted classes
        print("Final predictions 2:", final_output[0])  # Print final predicted classes
        print("Original probabilities 2:", output[1][0])
        print("Final probabilities 2:", final_output[1][0])  # Print final probabilities
        if output[0] != final_output[0]:
            amount_flipped_model_2 += 1
        probability_difference_2.append(np.abs(final_output[1][0][0] - output[1][0][0]))
        amount_iterations_2.append(iterations2)

    print("Amount flipped model 1", amount_flipped_model_1)
    print("Average iterations model 1", np.mean(amount_iterations_1))
    print("Median iterations model 1", np.median(amount_iterations_1))

    print("Average change in probabilities model 1", np.mean(probability_difference_1))
    print("Median change in probabilities model 1", np.median(probability_difference_1))

    print("Amount flipped model 2", amount_flipped_model_2)
    print("Average iterations model 2", np.mean(amount_iterations_2))
    print("Median iterations model 2", np.median(amount_iterations_2))

    print("Average change in probabilities model 2", np.mean(probability_difference_2))
    print("Median change in probabilities model 2", np.median(probability_difference_2))
    
    bar_graph_iterations(amount_iterations_1, amount_iterations_2, MAX_ITERATIONS)
    scatter_plot_side_by_side(probability_difference_1, amount_iterations_1, probability_difference_2, amount_iterations_2)


import matplotlib.pyplot as plt
import numpy as np

def bar_graph_iterations(iterations1, iterations2, max_iterations):
    # Define the buckets
    buckets = [1, (2,10), (11,20), (21,29), (30)]

    # Count the number of iterations in each bucket for both models
    counts1 = [
        sum(1 for i in iterations1 if i == 1),
        sum(1 for i in iterations1 if 2 <= i <= 11),
        sum(1 for i in iterations1 if 11 <= i <= 20),
        sum(1 for i in iterations1 if 21 <= i <= 29),
        sum(1 for i in iterations1 if i >= max_iterations)
    ]
    
    counts2 = [
        sum(1 for i in iterations2 if i == 1),
        sum(1 for i in iterations2 if 2 <= i <= 11),
        sum(1 for i in iterations2 if 11 <= i <= 20),
        sum(1 for i in iterations2 if 21 <= i <= 29),
        sum(1 for i in iterations2 if i >= max_iterations)
    ]
    
    x = np.arange(len(buckets))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, counts1, width, label='Model 1', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, counts2, width, label='Model 2', color='plum', edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Iterations', fontsize=12)
    ax.set_title('Number of Iterations for each model - Discriminatory Features', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=10)
    ax.legend(fontsize=10)

    # Annotate bars with their heights
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
            
    plt.tight_layout()
    plt.show()


def scatter_plot_side_by_side(prob_diff1, iterations1, prob_diff2, iterations2):
    """
    Create two side-by-side scatter plots for model 1 and model 2.

    Parameters:
    - prob_diff1, iterations1: Probability differences and iterations for Model 1.
    - prob_diff2, iterations2: Probability differences and iterations for Model 2.
    - title1, title2: Titles for the respective plots.
    """
    # Validate inputs
    if len(prob_diff1) != len(iterations1):
        print("Error: Model 1 data lengths must match.")
        return
    if len(prob_diff2) != len(iterations2):
        print("Error: Model 2 data lengths must match.")
        return

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle("Discriminatory Features")
    # Scatter plot for Model 1

    axes[0].scatter(prob_diff1, iterations1, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0].set_title("Model 1", fontsize=14, weight='bold')
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel('Probability Difference', fontsize=12)
    axes[0].set_ylabel('Number of Iterations', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Scatter plot for Model 2
    axes[1].scatter(prob_diff2, iterations2, color='plum', alpha=0.7, edgecolor='black')
    axes[1].set_title("Model 2", fontsize=14, weight='bold')
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel('Probability Difference', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


# Final result
run_search_based_on_both_models(MODEL_PATH_1, MODEL_PATH_2)
