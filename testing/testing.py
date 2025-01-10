from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def mutationTest(model1, model2,X,y,n_runs):
    model1_accuracies = []
    model2_accuracies= []
    model1_mutated_accuracies = []
    model2_mutated_accuracies = []

    # Evaluate on noisy test data across multiple runs
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        y_pred_model1 = model1.predict(X_test)
        model1_acc = accuracy_score(y_test, y_pred_model1)
        model1_accuracies.append(model1_acc)
        y_pred_model2 = model2.predict(X_test)
        model2_acc = accuracy_score(y_test, y_pred_model2)
        model2_accuracies.append(model2_acc)

        print(f"Initial accuracies: {model1_acc:.4f} for model 1, {model2_acc:.4f} for model 2")
        
        noise_test = np.random.normal(0, 10, size=X_test.shape)
        X_test_noisy = X_test + noise_test

        y_pred_model1_mutated = model1.predict(X_test_noisy)
        model1_mutated_acc = accuracy_score(y_test, y_pred_model1_mutated)
        model1_mutated_accuracies.append(model1_mutated_acc)

        y_pred_model2_mutated = model2.predict(X_test_noisy)
        model2_mutated_acc = accuracy_score(y_test, y_pred_model2_mutated)
        model2_mutated_accuracies.append(model2_mutated_acc)

        print(f"Model 1 Accuracy altered: {model1_mutated_acc:.4f}, Model 2 Accuracy altered: {model2_mutated_acc:.4f}")

    return {
        "model1": model1_accuracies,
        "model1_mutated": model1_mutated_accuracies,
        "model2": model2_accuracies,
        "model2_mutated": model2_mutated_accuracies
    }

def compareAccuracies(results):
    # Prepare data for Plotly
    data = pd.DataFrame({
        'Accuracy': results['model1'] + results['model1_mutated'] + results['model2'] + results['model2_mutated'],
        'Model': ['Model 1'] * len(results['model1']) + ['Model 1 Mutated'] * len(results['model1_mutated']) + ['Model 2 '] * len(results['model2']) + ['Model 2 Mutated'] * len(results['model2_mutated'])
    })

    # Create the boxplot
    fig = px.box(data, x='Model', y='Accuracy', title='Accuracy Comparison', labels={'Accuracy': 'Accuracy', 'Model': 'Model Type'})
    fig.show()

def differentationTesting(model1,model2,X,y,outlier_percentage,features_to_modify,n_runs):

    metrics = {"model1": {"before": [], "after": []}, "model2": {"before": [], "after": []}}

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        predictionsOne = model1.predict(X_test)
        accuracyOne = accuracy_score(y_test, predictionsOne)
        tn, fp, fn, tp = confusion_matrix(y_test, predictionsOne).ravel()
        metrics["model1"]["before"].append({"accuracy": accuracyOne, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

        predictionsTwo = model2.predict(X_test)
        accuracyTwo = accuracy_score(y_test, predictionsTwo)
        tnt, fpt, fnt, tpt = confusion_matrix(y_test, predictionsTwo).ravel()
        metrics["model2"]["before"].append({"accuracy": accuracyTwo, "tp": tpt, "fp": fpt, "tn": tnt, "fn": fnt})

        # here we introduce outliers 
        
        num_outliers = int(outlier_percentage * len(X_test))
        outlier_indices = np.random.choice(X_test.index, size=num_outliers, replace=False)

        for feature in features_to_modify:
            X_test.loc[outlier_indices, feature] += np.random.choice([-100, 100], size=num_outliers)  

        predictionsOne = model1.predict(X_test)
        accuracyOne = accuracy_score(y_test, predictionsOne)
        tn, fp, fn, tp = confusion_matrix(y_test, predictionsOne).ravel()
        metrics["model1"]["after"].append({"accuracy": accuracyOne, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

        predictionsTwo = model2.predict(X_test)
        accuracyTwo = accuracy_score(y_test, predictionsTwo)
        tnt, fpt, fnt, tpt = confusion_matrix(y_test, predictionsTwo).ravel()
        metrics["model2"]["after"].append({"accuracy": accuracyTwo, "tp": tpt, "fp": fpt, "tn": tnt, "fn": fnt})
    
    return metrics


def print_average_metrics(metrics):
    def calculate_averages(model_key):
        before = metrics[model_key]["before"]
        after = metrics[model_key]["after"]

        averages = {
            "accuracy_before": np.mean([run["accuracy"] for run in before]),
            "accuracy_after": np.mean([run["accuracy"] for run in after]),
            "tp_before": np.mean([run["tp"] for run in before]),
            "tp_after": np.mean([run["tp"] for run in after]),
            "fp_before": np.mean([run["fp"] for run in before]),
            "fp_after": np.mean([run["fp"] for run in after]),
            "tn_before": np.mean([run["tn"] for run in before]),
            "tn_after": np.mean([run["tn"] for run in after]),
            "fn_before": np.mean([run["fn"] for run in before]),
            "fn_after": np.mean([run["fn"] for run in after]),
        }
        return averages

    model1_averages = calculate_averages("model1")
    model2_averages = calculate_averages("model2")

    print("Average Metrics Across Runs:\n")
    print("Model 1:")
    for key, value in model1_averages.items():
        print(f"{key}: {value:.2f}")

    print("\nModel 2:")
    for key, value in model2_averages.items():
        print(f"{key}: {value:.2f}")


def equivalencePartitioning(model1,model2,X,y,partitions,run):
    results = []
    print(f"\nRun {run}")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

    for partition in partitions:
        partition_data = X_test[partition["condition"](X_test)]
        partition_indices = partition_data.index  # Get the indices of the partition
        partition_labels = y_test.loc[partition_indices]  # Get the actual labels for the partition

        if not partition_data.empty:
            # Predictions using the model
            predictions = model1.predict(partition_data)
            # Calculate accuracy for this partition
            accuracy = accuracy_score(partition_labels, predictions)
            tn, fp, fn, tp = confusion_matrix(partition_labels, predictions).ravel()
            results.append({"accuracy": accuracy, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

            predictions1 = model2.predict(partition_data)
            # Calculate accuracy for this partition
            accuracy1 = accuracy_score(partition_labels, predictions1)
            tn1, fp1, fn1, tp1 = confusion_matrix(partition_labels, predictions1).ravel()
            results.append({"accuracy": accuracy1, "tp": tp1, "fp": fp1, "tn": tn1, "fn": fn1})


    return results
