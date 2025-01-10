from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import onnxruntime as rt


def mutationTest(model1,X,y,n_runs):
    model1_accuracies = []
    model1_mutated_accuracies = []

    # Evaluate on noisy test data across multiple runs
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        y_test = y_test.astype(int)

        y_pred_model1 = model1.run(None, {'X': X_test.values.astype(np.float32)})
        y_pred_model1_binary = pred_to_binary(y_pred_model1[0])
        model1_acc = accuracy_score(y_test, y_pred_model1_binary)
        model1_accuracies.append(model1_acc)
        print(f"Initial accuracy: {model1_acc:.4f} for model")
        
        noise_test = np.random.normal(0, 10, size=X_test.shape)
        X_test_noisy = X_test + noise_test

        y_pred_model1_mutated = model1.run(None, {'X': X_test_noisy.values.astype(np.float32)})
        y_pred_model1_mutated_binary = pred_to_binary(y_pred_model1_mutated[0])
        model1_mutated_acc = accuracy_score(y_test, y_pred_model1_mutated_binary)
        model1_mutated_accuracies.append(model1_mutated_acc)

        print(f"Model Accuracy altered: {model1_mutated_acc:.4f}")

    return {
        "model": model1_accuracies,
        "model_mutated": model1_mutated_accuracies,
    }

def compareAccuracies(results1,results2):
    # Prepare data for Plotly
    data = pd.DataFrame({
        'Accuracy': results1['model'] + results1['model_mutated'] + results2['model'] + results2['model_mutated'],
        'Model': ['Model 1'] * len(results1['model']) + ['Model 1 Mutated'] * len(results1['model_mutated']) + ['Model 2 '] * len(results2['model']) + ['Model 2 Mutated'] * len(results2['model_mutated'])
    })

    # Create the boxplot
    fig = px.box(data, x='Model', y='Accuracy', title='Accuracy Comparison', labels={'Accuracy': 'Accuracy', 'Model': 'Model Type'})
    fig.update_yaxes(range=[0, 1])

    fig.show()

def differentiationTesting(model1,X,y,outlier_percentage,features_to_modify,n_runs):

    metrics = {"model1": {"before": [], "after": []}}

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        y_test = y_test.astype(int)

        predictionsOne = model1.run(None, {'X': X_test.values.astype(np.float32)})
        predictionsOne_binary = pred_to_binary(predictionsOne[0])
        accuracyOne = accuracy_score(y_test, predictionsOne_binary)
        tn, fp, fn, tp = confusion_matrix(y_test, predictionsOne_binary).ravel()
        metrics["model1"]["before"].append({"accuracy": accuracyOne, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

        # here we introduce outliers 
        
        num_outliers = int(outlier_percentage * len(X_test))
        outlier_indices = np.random.choice(X_test.index, size=num_outliers, replace=False)

        for feature in features_to_modify:
            X_test.loc[outlier_indices, feature] += np.random.choice([-100, 100], size=num_outliers)  

        predictionsOne = model1.run(None, {'X': X_test.values.astype(np.float32)})
        predictionsOne_binary = pred_to_binary(predictionsOne[0])
        accuracyOne = accuracy_score(y_test, predictionsOne_binary)
        tn, fp, fn, tp = confusion_matrix(y_test, predictionsOne_binary).ravel()
        metrics["model1"]["after"].append({"accuracy": accuracyOne, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    
    return metrics


def print_average_metrics(metrics,metrics2):
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
    
    def calculate_averages2(model_key):
        before = metrics2[model_key]["before"]
        after = metrics2[model_key]["after"]

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
    model2_averages = calculate_averages2("model1")

    print("Average Metrics Across Runs:\n")
    print("Model 1:")
    for key, value in model1_averages.items():
        print(f"{key}: {value:.2f}")

    print("\nModel 2:")
    for key, value in model2_averages.items():
        print(f"{key}: {value:.2f}")


def equivalencePartitioning(model1,X,y,partitions,run):
    results = []
    print(f"\nRun {run}")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
    y_test = y_test.astype(int)

    for partition in partitions:
        partition_data = X_test[partition["condition"](X_test)]
        partition_indices = partition_data.index  # Get the indices of the partition
        partition_labels = y_test.loc[partition_indices]  # Get the actual labels for the partition

        if not partition_data.empty:
            # Predictions using the model
            predictions = model1.run(None, {'X': partition_data.values.astype(np.float32)})
            predictions_binary = pred_to_binary(predictions[0])
            # Calculate accuracy for this partition
            accuracy = accuracy_score(partition_labels, predictions_binary)
            tn, fp, fn, tp = confusion_matrix(partition_labels, predictions_binary).ravel()
            results.append({"accuracy": accuracy, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

    return results

def plot_EP_results(results,results2,metric):
    model_1 = results
    model_2 = results2

    model_1_values = [res[metric] for res in model_1]
    model_2_values = [res[metric] for res in model_2]

    x = np.arange(len(model_1)) 
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, model_1_values, bar_width, label='Model 1')
    plt.bar(x + bar_width / 2, model_2_values, bar_width, label='Model 2')

    # Formatting
    plt.xlabel('Partition', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Compare models based on {metric.capitalize()}', fontsize=14)
    plt.xticks(x, [f'Partition {i + 1}' for i in range(len(model_1))])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def calculateEPHighestDifference(df, model, n_partitions=5):
    results = []
    values_to_remove = ['checked', 'Ja', 'Nee']
    features = [col for col in df.columns if col not in values_to_remove]    
    for feature in features:
        feature_values = df[feature]
        if pd.api.types.is_numeric_dtype(feature_values):
            if (feature_values.nunique() <= 1): continue
            bins = np.linspace(feature_values.min(), feature_values.max(), n_partitions + 1)
            df['partition'] = pd.cut(feature_values, bins, labels=False, include_lowest=True)
        else:
            unique_values = feature_values.unique()
            if len(unique_values) > n_partitions:
                unique_values = unique_values[:n_partitions]  
            df['partition'] = feature_values.apply(lambda x: np.where(unique_values == x)[0][0] if x in unique_values else -1)

        accuracies = []
        for partition in df['partition'].unique():
            if partition == -1:
                continue
            partition_data = df[df['partition'] == partition]
            X_partition = partition_data[features]
            y_partition = partition_data['checked']
            y_partition = y_partition.astype(int)
            y_pred = model.run(None, {'X': X_partition.values.astype(np.float32)})
            y_pred_binary = pred_to_binary(y_pred[0])
            accuracy = accuracy_score(y_partition, y_pred_binary)
            accuracies.append(accuracy)

        if len(accuracies) > 1:
            accuracy_diff = max(accuracies) - min(accuracies)
            results.append({'feature': feature, 'accuracy_difference': accuracy_diff, 'accuracies': accuracies})

        df.drop(columns=['partition'], inplace=True)

    # Return top 5 features with highest accuracy difference
    results_df = pd.DataFrame(results).sort_values(by='accuracy_difference', ascending=False)
    return results_df.head(5)

def pred_to_binary(predictions, threshold=0.5):
    """Converts risk scores to binary values."""
    return (predictions >= threshold).astype(int)

