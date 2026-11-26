import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Data preparation same as in explore_dataset_trial.ipynb

data_train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
data_test = np.loadtxt("ECG5000/ECG5000_TEST.txt")

df_train = pd.DataFrame(data_train, columns=["class"] + [f"value_{i}" for i in range(1, data_train.shape[1])])
df_test = pd.DataFrame(data_test, columns=["class"] + [f"value_{i}" for i in range(1, data_test.shape[1])])

y_min = df_train.iloc[:, 1:].min().min()
y_max = df_train.iloc[:, 1:].max().max()
labels = df_train["class"].unique().astype(int)

signals_by_class = {
    label: df_train[df_train["class"] == label].iloc[:, 1:].values
    for label in labels
}

# Partition the dataset into normal and abnormal classes

df_train["binary_label"] = (df_train["class"] != 1).astype(int)

df_normal = df_train[df_train["binary_label"] == 0]
df_abnormal = df_train[df_train["binary_label"] == 1]

X_normal = df_normal.iloc[:, 1:-1].values
X_abnormal = df_abnormal.iloc[:, 1:-1].values

df_test["binary_label"] = (df_test["class"] != 1).astype(int)

df_test_normal = df_test[df_test["binary_label"] == 0]
df_test_abnormal = df_test[df_test["binary_label"] == 1]

X_test_normal = df_test_normal.iloc[:, 1:-1].values
X_test_abnormal = df_test_abnormal.iloc[:, 1:-1].values

# Fit PCA on the normal training data and visualize the explained variance and principal components

pca = PCA()
pca.fit(X_normal)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_[:21]), marker="o", linestyle="-", color="b")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.xticks(np.arange(0, 21))
plt.yticks(np.linspace(0, 1, 21))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.plot(pca.components_[i])
    ax.set_ylim(-0.35, 0.45)
    ax.set_title(f"Principal Component {i+1}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Loading")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Reconstruct the signals using the first n principal components and visualize the original and reconstructed signals

n_relevant_components = 4

X_train_projected_normal = pca.transform(X_normal)[:, :n_relevant_components]
X_train_reconstructed_normal = pca.inverse_transform(
    np.pad(X_train_projected_normal, ((0, 0), (0, pca.n_components_ - n_relevant_components)))
)

X_test_projected_normal = pca.transform(X_test_normal)[:, :n_relevant_components]
X_test_reconstructed_normal = pca.inverse_transform(
    np.pad(X_test_projected_normal, ((0, 0), (0, pca.n_components_ - n_relevant_components)))
)

X_test_projected_abnormal = pca.transform(X_test_abnormal)[:, :n_relevant_components]
X_test_reconstructed_abnormal = pca.inverse_transform(
    np.pad(X_test_projected_abnormal, ((0, 0), (0, pca.n_components_ - n_relevant_components)))
)

n_shown_components = 5

fig, axes = plt.subplots(n_shown_components, 3, figsize=(14, 2.5 * n_shown_components), sharex=True)

for i in range(n_shown_components):
    axes[i, 0].plot(X_normal[i], label="Original")
    axes[i, 0].plot(X_train_reconstructed_normal[i], label="Reconstructed", linestyle="--")
    axes[i, 0].set_title(f"Normal heartbeat {i+1}")
    axes[i, 0].legend()
    axes[i, 0].grid(alpha=0.3)
    
    axes[i, 1].plot(X_test_normal[i], label="Original")
    axes[i, 1].plot(X_test_reconstructed_normal[i], label="Reconstructed", linestyle="--")
    axes[i, 1].set_title(f"Normal heartbeat {i+1}")
    axes[i, 1].legend()
    axes[i, 1].grid(alpha=0.3)

    axes[i, 2].plot(X_test_abnormal[i], label="Original")
    axes[i, 2].plot(X_test_reconstructed_abnormal[i], label="Reconstructed", linestyle="--")
    axes[i, 2].set_title(f"Abnormal heartbeat {i+1}")
    axes[i, 2].legend()
    axes[i, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate the reconstruction errors for normal and abnormal signals and visualize their distributions

normal_training_errors = np.mean((X_normal - X_train_reconstructed_normal)**2, axis=1)
normal_errors = np.mean((X_test_normal - X_test_reconstructed_normal)**2, axis=1)
abnormal_errors = np.mean((X_test_abnormal - X_test_reconstructed_abnormal)**2, axis=1)

threshold_95 = np.percentile(normal_training_errors, 95)
threshold_99 = np.percentile(normal_training_errors, 99)

plt.figure(figsize=(10, 5))

plt.hist(normal_training_errors, bins=50, alpha=0.5,
         density=True, label="Normal training")

plt.hist(normal_errors, bins=50, alpha=0.5,
         density=True, label="Normal test")

plt.hist(abnormal_errors, bins=50, alpha=0.5,
         density=True, label="Abnormal test")

plt.axvline(threshold_95, color="orange",
            linestyle="--", label="95th percentile")

plt.axvline(threshold_99, color="red",
            linestyle="--", label="99th percentile")

plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.title("PCA Reconstruction Error Distributions")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Choose 95th percentile as the threshold for classification and evaluate the model's performance

y_true = np.concatenate([
    np.zeros(len(normal_errors)),
    np.ones(len(abnormal_errors))
]).astype(int)

all_errors = np.concatenate([normal_errors, abnormal_errors])
y_pred = (all_errors > threshold_95).astype(int)

TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, all_errors)

print(f"Threshold: {threshold_95:.4f}")
print(f"TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}")

print("\nEvaluation Metrics:")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Normal", "Abnormal"],
    zero_division=0
))

# Visualize missclassified signals (false positives and false negatives) along with their reconstruction errors

false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

fp_indices = false_positives
fn_indices = false_negatives - len(normal_errors)

n_reconstructions = 4
fig, axes = plt.subplots(2, n_reconstructions, figsize=(14, 7), sharey=True)

for i in range(n_reconstructions):
    if i < len(fp_indices):
        idx = fp_indices[i]

        axes[0, i].plot(X_test_normal[idx], label="Original")
        axes[0, i].plot(X_test_reconstructed_normal[idx],
                        label="Reconstructed", linestyle="--")
        axes[0, i].set_title(f"False Positive\nMSE={normal_errors[idx]:.4f}")

    if i < len(fn_indices):
        idx = fn_indices[i]

        axes[1, i].plot(X_test_abnormal[idx], label="Original")
        axes[1, i].plot(X_test_reconstructed_abnormal[idx],
                        label="Reconstructed", linestyle="--")
        axes[1, i].set_title(f"False Negative\nMSE={abnormal_errors[idx]:.4f}")

    for ax in axes[:, i]:
        ax.set_xlabel("Sample")
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=8)

axes[0, 0].set_ylabel("False Positives\nAmplitude")
axes[1, 0].set_ylabel("False Negatives\nAmplitude")

plt.suptitle(f"PCA Misclassifications (Threshold = {threshold_95:.4f})")
plt.tight_layout()
plt.show()

# Lets look at how subclasses perform where does the reconstruction error come from for each subclass
