import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =============================
# Configuration
# =============================
DIRECTIONS_4 = ["UP", "RIGHT", "BOTTOM", "LEFT"]
DIRECTIONS_8 = [
    "UP", "UP_RIGHT", "RIGHT", "BOTTOM_RIGHT",
    "BOTTOM", "BOTTOM_LEFT", "LEFT", "UP_LEFT"
]

# =============================
# Parse one experiment file
# =============================
def parse_experiment_file(filepath):
    """
    Returns:
        accuracies: dict {direction: accuracy}
        confusion_data: dict {target: [predictions]}
    """
    accuracy_data = defaultdict(list)
    confusion_data = defaultdict(list)
    current_target = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("TARGET:"):
                current_target = line.split(":")[1].strip()

            elif line.startswith("NEXT") or line == "EXPERIMENT_FINISHED":
                current_target = None

            elif current_target:
                accuracy_data[current_target].append(line)
                confusion_data[current_target].append(line)

    accuracies = {}
    for target, predictions in accuracy_data.items():
        correct = sum(1 for p in predictions if p == target)
        accuracies[target] = correct / len(predictions)

    return accuracies, confusion_data


# =============================
# Average accuracy
# =============================
def average_accuracies(filepaths, directions):
    sums = defaultdict(float)
    counts = defaultdict(int)

    for path in filepaths:
        acc, _ = parse_experiment_file(path)
        for d in directions:
            sums[d] += acc.get(d, 0.0)
            counts[d] += 1

    return {d: sums[d] / counts[d] for d in directions}


# =============================
# Confusion matrix (counts)
# =============================
def build_confusion_counts(filepaths, directions):
    index = {d: i for i, d in enumerate(directions)}
    matrix = np.zeros((len(directions), len(directions)))

    for path in filepaths:
        _, confusion_data = parse_experiment_file(path)

        for target, predictions in confusion_data.items():
            if target not in index:
                continue
            t = index[target]
            for p in predictions:
                if p in index:
                    matrix[t, index[p]] += 1

    return matrix


# =============================
# Normalize confusion matrix
# =============================
def normalize_confusion(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


# =============================
# Metrics computation
# =============================
def compute_metrics(confusion_matrix, directions):
    metrics = {}

    for i, d in enumerate(directions):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        total = confusion_matrix[i, :].sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        accuracy = TP / total if total > 0 else 0

        metrics[d] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


# =============================
# Print metrics table
# =============================
def print_metrics(metrics, title):
    print("\n" + title)
    print("=" * len(title))
    print(f"{'Region':<15} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")

    for region, m in metrics.items():
        print(
            f"{region:<15} "
            f"{m['accuracy']:.2f} "
            f"{m['precision']:.2f} "
            f"{m['recall']:.2f} "
            f"{m['f1']:.2f}"
        )


# =============================
# Plot accuracy
# =============================
def plot_accuracy(avg_acc, title):
    plt.figure()
    plt.bar(avg_acc.keys(), avg_acc.values())
    plt.ylim(0, 1)
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

DIRECTIONS_4_new = ["U", "R", "B", "L"]
DIRECTIONS_8_new = [
    "U", "UR", "R", "BR",
    "B", "BL", "L", "UL"
]

# =============================
# Plot confusion heatmap
# =============================
def plot_confusion(matrix, directions, title):
    plt.figure()
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(label="Probability")

    plt.xticks(range(len(directions)), directions, rotation=45)
    plt.yticks(range(len(directions)), directions)

    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.title(title)

    plt.tight_layout()
    plt.show()


# =============================
# Example usage
# =============================
experiment_4_files = [
    "miren_gaze_experiment_4.txt",
    "eneko_gaze_experiment_4.txt",
    "markel_gaze_experiment_4.txt",
    "unai_gaze_experiment_4.txt",
    "aitor_gaze_experiment_4.txt",
    "itziar_gaze_experiment_4.txt",
    "itziar2_gaze_experiment_4.txt",
    "elena_gaze_experiment_4.txt",
    "igor_gaze_experiment_4.txt",
    "maitane_gaze_experiment_4.txt",
]

experiment_8_files = [
    "miren2_gaze_experiment_8.txt",
    "markel_gaze_experiment_8.txt",
    "unai_gaze_experiment_8.txt",
    "aitor_gaze_experiment_8.txt",
    "itziar_gaze_experiment_8.txt",
    "itziar2_gaze_experiment_8.txt",
    "elena_gaze_experiment_8.txt",
    "maitane_gaze_experiment_8.txt",
]

# ---- Accuracy plots ----
avg_4 = average_accuracies(experiment_4_files, DIRECTIONS_4)
avg_8 = average_accuracies(experiment_8_files, DIRECTIONS_8)

plot_accuracy(avg_4, "Average Accuracy – 4 Directions")
plot_accuracy(avg_8, "Average Accuracy – 8 Directions")

# ---- Confusion matrices ----
conf_4_counts = build_confusion_counts(experiment_4_files, DIRECTIONS_4)
conf_8_counts = build_confusion_counts(experiment_8_files, DIRECTIONS_8)

conf_4 = normalize_confusion(conf_4_counts)
conf_8 = normalize_confusion(conf_8_counts)

plot_confusion(conf_4, DIRECTIONS_4_new, "Confusion Matrix – 4 Directions")
plot_confusion(conf_8, DIRECTIONS_8_new, "Confusion Matrix – 8 Directions")

# ---- Metrics ----
metrics_4 = compute_metrics(conf_4_counts, DIRECTIONS_4)
metrics_8 = compute_metrics(conf_8_counts, DIRECTIONS_8)

print_metrics(metrics_4, "4-Direction Experiment Metrics")
print_metrics(metrics_8, "8-Direction Experiment Metrics")
