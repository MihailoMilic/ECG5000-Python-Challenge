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

def explained_variance_null_test(X, n_null=999, alpha=0.05, seed=42,
                                 null_model="permutation"):
    """Compare observed variance shares with the null MAXIMUM share.

    permutation: independently shuffle rows within each feature, preserving
    its empirical marginal distribution (including variance and higher moments).
    gaussian: simulate iid N(0,1) entries, the classical isotropic MP null.

    Rows must be independent/exchangeable for the permutation interpretation.
    This tests against a complete independence null; the selected count is a
    screening rule, not a confidence-certified estimate of latent rank.
    No feature standardization is applied, matching the original covariance PCA.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("X must contain at least 3 rows and 1 feature.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if not 0 < alpha < 1 or not isinstance(n_null, (int, np.integer)) or n_null < 1:
        raise ValueError("Require 0 < alpha < 1 and a positive integer n_null.")
    if null_model not in {"permutation", "gaussian"}:
        raise ValueError("null_model must be 'permutation' or 'gaussian'.")
    if 1 / (n_null + 1) > alpha:
        raise ValueError("Increase n_null to resolve the requested significance level.")

    n_samples, n_features = X.shape
    centered = X - X.mean(axis=0)

    def variance_shares(A):
        # The covariance divisor cancels in lambda / trace. Use the smaller
        # Gram matrix; omitted eigenvalues are zero when features exceed rows.
        gram = A.T @ A if A.shape[1] <= A.shape[0] else A @ A.T
        eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
        total = eigenvalues.sum()
        if total <= 0:
            raise ValueError("Total variance must be positive.")
        return eigenvalues / total

    observed = variance_shares(centered)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_null)
    for b in range(n_null):
        if null_model == "permutation":
            null_X = np.empty_like(centered)
            for j in range(n_features):
                null_X[:, j] = rng.permutation(centered[:, j])
        else:
            null_X = rng.standard_normal(X.shape)
            null_X -= null_X.mean(axis=0)
        null_max[b] = variance_shares(null_X)[0]

    # Every component is compared to the SAME distribution of null maxima,
    # never to its corresponding rank in the null spectrum.
    p_values = (1 + (null_max[:, None] >= observed[None, :]).sum(axis=0)) / (n_null + 1)
    # Order-statistic cutoff agrees with the add-one Monte Carlo p-value rule
    # (strict exceedance is needed at the cutoff, including ties).
    cutoff_rank = int(np.ceil((1 - alpha) * (n_null + 1)))
    threshold = np.sort(null_max)[cutoff_rank - 1]
    significant = p_values <= alpha

    # Classical equal-variance MP edge divided by asymptotic total variance:
    # sigma^2 * (1 + sqrt(p/(N-1)))^2 / (p * sigma^2).
    # Reference ONLY for heterogeneous marginals, not an alpha-level cutoff.
    mp_reference = (1 + np.sqrt(n_features / (n_samples - 1)))**2 / n_features
    return {
        "explained_variance_ratio": observed,
        "null_max": null_max,
        "threshold": threshold,
        "p_values": p_values,
        "significant": significant,
        "n_significant": int(significant.sum()),
        "mp_reference": mp_reference,
        "alpha": alpha,
        "null_model": null_model,
    }


def plot_null_scree(result):
    ratios = result["explained_variance_ratio"]
    components = np.arange(1, len(ratios) + 1)
    selected = result["significant"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(components, 100 * ratios, "o-", markersize=3,
            color="steelblue", label="Observed explained variance")
    ax.scatter(components[selected], 100 * ratios[selected], color="crimson",
               zorder=3, label="Exceeds null-maximum threshold")
    ax.axhline(100 * result["threshold"], color="crimson", linestyle="--",
               label=f"{100 * (1 - result['alpha']):g}% null-maximum cutoff")
    ax.axhline(100 * result["mp_reference"], color="darkorange", linestyle=":",
               label="Normalized MP edge (equal-variance reference)")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)",
           title="Scree plot: explained variance versus null maximum")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


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

df_train["binary_label"] = (df_train["class"] != 1).astype(int)

df_normal = df_train[df_train["binary_label"] == 0]
df_abnormal = df_train[df_train["binary_label"] == 1]

X_normal = df_normal.iloc[:, 1:-1].values
X_abnormal = df_abnormal.iloc[:, 1:-1].values

pca = PCA()
pca.fit(X_normal)

# Train-only, moment-preserving null comparison. Use null_model="gaussian"
# for the classical iid MP null instead.
mp_result = explained_variance_null_test(X_normal, n_null=999, alpha=0.05)
print(f"Null maximum explained-variance cutoff: {mp_result['threshold']:.2%}")
print(f"Components exceeding the cutoff: {mp_result['n_significant']}")
print(pd.DataFrame({
    "component": np.arange(1, len(mp_result["p_values"]) + 1),
    "explained_variance_ratio": mp_result["explained_variance_ratio"],
    "p_vs_null_max": mp_result["p_values"],
    "exceeds_null_max": mp_result["significant"],
}).to_string(index=False))
plot_null_scree(mp_result)
plt.show()
# Reconstruction below keeps your original n=4. To use this screening rule,
# replace that assignment with n = mp_result["n_significant"].

df_test["binary_label"] = (df_test["class"] != 1).astype(int)

df_test_normal = df_test[df_test["binary_label"] == 0]
df_test_abnormal = df_test[df_test["binary_label"] == 1]

X_test_normal = df_test_normal.iloc[:, 1:-1].values
X_test_abnormal = df_test_abnormal.iloc[:, 1:-1].values

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

n=4

X_train_projected_normal = pca.transform(X_normal)[:, :n]
X_train_reconstructed_normal = pca.inverse_transform(
    np.pad(X_train_projected_normal, ((0, 0), (0, pca.n_components_ - n)))
)

X_test_projected_normal = pca.transform(X_test_normal)[:, :n]
X_test_reconstructed_normal = pca.inverse_transform(
    np.pad(X_test_projected_normal, ((0, 0), (0, pca.n_components_ - n)))
)

X_test_projected_abnormal = pca.transform(X_test_abnormal)[:, :n]
X_test_reconstructed_abnormal = pca.inverse_transform(
    np.pad(X_test_projected_abnormal, ((0, 0), (0, pca.n_components_ - n)))
)

n = 5

fig, axes = plt.subplots(n, 3, figsize=(14, 2.5 * n), sharex=True)

for i in range(n):
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

normal_training_errors = np.mean((X_normal - X_train_reconstructed_normal)**2, axis=1)
normal_errors = np.mean((X_test_normal - X_test_reconstructed_normal)**2, axis=1)
abnormal_errors = np.mean((X_test_abnormal - X_test_reconstructed_abnormal)**2, axis=1)

print("Mean normal training error:", normal_training_errors.mean())
print("Mean normal reconstruction error:", normal_errors.mean())
print("Mean abnormal reconstruction error:", abnormal_errors.mean())

threshold_95 = np.percentile(normal_training_errors, 95)
threshold_99 = np.percentile(normal_training_errors, 99)

print("95th percentile:", threshold_95)
print("99th percentile:", threshold_99)

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

# Identify misclassified observations
false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

# Separate indices for normal and abnormal test observations
fp_indices = false_positives
fn_indices = false_negatives - len(normal_errors)

n = 4
fig, axes = plt.subplots(2, n, figsize=(14, 7), sharey=True)

for i in range(n):
    # False positives: normal signals classified as abnormal
    if i < len(fp_indices):
        idx = fp_indices[i]

        axes[0, i].plot(X_test_normal[idx], label="Original")
        axes[0, i].plot(X_test_reconstructed_normal[idx],
                        label="Reconstructed", linestyle="--")
        axes[0, i].set_title(f"False Positive\nMSE={normal_errors[idx]:.4f}")

    # False negatives: abnormal signals classified as normal
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

# Lets look at how subclasses perform
