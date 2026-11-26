import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Configuration with help from AI.

RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10
THRESHOLD_PERCENTILE = 95

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Data preparation same as in pca_reconstruction_trial.py parts 1 and 2.

data_train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
data_test = np.loadtxt("ECG5000/ECG5000_TEST.txt")

df_train = pd.DataFrame(data_train, columns=["class"] + [f"value_{i}" for i in range(1, data_train.shape[1])])
df_test = pd.DataFrame(data_test, columns=["class"] + [f"value_{i}" for i in range(1, data_test.shape[1])])

df_train["binary_label"] = (df_train["class"] != 1).astype(int)

df_normal = df_train[df_train["binary_label"] == 0]
df_abnormal = df_train[df_train["binary_label"] == 1]

X_normal = df_normal.iloc[:, 1:-1].values
X_abnormal = df_abnormal.iloc[:, 1:-1].values

df_test["binary_label"] = (df_test["class"] != 1).astype(int)
y_test_original = df_test["class"].to_numpy()

df_test_normal = df_test[df_test["binary_label"] == 0]
df_test_abnormal = df_test[df_test["binary_label"] == 1]

X_test_normal = df_test_normal.iloc[:, 1:-1].values
X_test_abnormal = df_test_abnormal.iloc[:, 1:-1].values


# Split the normal training data into training and validation sets.

X_train_normal, X_val_normal = train_test_split(
    X_normal,
    test_size=0.2,
    random_state=RANDOM_SEED,
)

input_dim = X_train_normal.shape[1]

print("\nData shapes:")
print("Normal training:", X_train_normal.shape)
print("Abnormal training (not used for fitting):", X_abnormal.shape)
print("Normal validation:", X_val_normal.shape)
print("Normal test:", X_test_normal.shape)
print("Abnormal test:", X_test_abnormal.shape)

X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_normal, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, X_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val_tensor, X_val_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Define the autoencoder architecture with help from AI.

mixed_experiments = {
    "A": {"hidden_layers": [64], "latent_dim": 32},
    "B": {"hidden_layers": [64, 32], "latent_dim": 16},
    "C": {"hidden_layers": [64, 32], "latent_dim": 8},
    "D": {"hidden_layers": [128, 64], "latent_dim": 16},
    "E": {"hidden_layers": [128, 64], "latent_dim": 4},
}

latent_experiments = {
    "L4":  {"hidden_layers": [64, 32], "latent_dim": 4},
    "L8":  {"hidden_layers": [64, 32], "latent_dim": 8},
    "L16": {"hidden_layers": [64, 32], "latent_dim": 16},
    "L32": {"hidden_layers": [64, 32], "latent_dim": 32},
}

architecture_experiments = {
    "Shallow": {"hidden_layers": [64], "latent_dim": 16},
    "Deep":    {"hidden_layers": [64, 32], "latent_dim": 16},
    "Wide":    {"hidden_layers": [128, 64], "latent_dim": 16},
    "Deeper":  {"hidden_layers": [128, 64, 32], "latent_dim": 16},
}

interaction_experiments = {
    "A": {"hidden_layers": [64], "latent_dim": 8},
    "B": {"hidden_layers": [64], "latent_dim": 32},
    "C": {"hidden_layers": [128, 64], "latent_dim": 8},
    "D": {"hidden_layers": [128, 64], "latent_dim": 32},
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim):
        super().__init__()

        # Encoder.
        encoder_layers = []
        previous_dim = input_dim

        for units in hidden_layers:
            encoder_layers.extend([
                nn.Linear(previous_dim, units),
                nn.ReLU(),
            ])
            previous_dim = units

        encoder_layers.extend([
            nn.Linear(previous_dim, latent_dim),
            nn.ReLU(),
        ])

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: mirror the encoder architecture.
        decoder_layers = []
        previous_dim = latent_dim

        for units in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(previous_dim, units),
                nn.ReLU(),
            ])
            previous_dim = units

        # Linear output layer for signal reconstruction.
        decoder_layers.append(nn.Linear(previous_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_autoencoder(input_dim, hidden_layers, latent_dim):
    return Autoencoder(input_dim, hidden_layers, latent_dim).to(device)


# ============================================================
# 4. Train all experiments with early stopping
# ============================================================

EXPERIMENTS_DICT = mixed_experiments

autoencoders = {}
histories = {}
experiment_results = []

for name, config in EXPERIMENTS_DICT.items():
    print("\n" + "=" * 55)
    print(f"Experiment {name}")
    print("Hidden layers:", config["hidden_layers"])
    print("Latent dimension:", config["latent_dim"])
    print("=" * 55)

    # Reset the seed before each model for reproducibility.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = build_autoencoder(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        latent_dim=config["latent_dim"],
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    counter = 0

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss_sum = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * inputs.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- Training loss: {train_loss:.6f} "
            f"- Validation loss: {val_loss:.6f}"
        )

        # Save the best validation model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        if counter >= PATIENCE:
            print("Early stopping.")
            break

    # Restore the best model weights.
    model.load_state_dict(best_weights)
    autoencoders[name] = model

    histories[name] = {
        "loss": train_losses,
        "val_loss": val_losses,
    }

    architecture = (
        [input_dim]
        + config["hidden_layers"]
        + [config["latent_dim"]]
        + list(reversed(config["hidden_layers"]))
        + [input_dim]
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    experiment_results.append({
        "Experiment": name,
        "Architecture": " -> ".join(map(str, architecture)),
        "Latent Dimension": config["latent_dim"],
        "Parameters": parameter_count,
        "Best Epoch": best_epoch + 1,
        "Training Loss": train_losses[best_epoch],
        "Validation Loss": best_val_loss,
    })


# ============================================================
# 5. Compare architectures and training histories
# ============================================================

results_df = pd.DataFrame(experiment_results).set_index("Experiment")

print("\nArchitecture comparison:")
print(results_df.round(6).to_string())

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, (name, history) in zip(axes, histories.items()):
    ax.plot(history["loss"], label="Training loss")
    ax.plot(history["val_loss"], label="Validation loss")

    ax.set_title(
        f"Experiment {name} "
        f"(latent dimension = {EXPERIMENTS_DICT[name]['latent_dim']})"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.3)

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()


# ============================================================
# 6. Calculate reconstruction errors
# ============================================================

def reconstruct(model, X):
    """Return reconstructed signals as a NumPy array."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()

    return reconstructed


reconstruction_errors = {}

for name, model in autoencoders.items():
    reconstructed_train = reconstruct(model, X_train_normal)
    reconstructed_val = reconstruct(model, X_val_normal)
    reconstructed_normal = reconstruct(model, X_test_normal)
    reconstructed_abnormal = reconstruct(model, X_test_abnormal)

    reconstruction_errors[name] = {
        "train": np.mean(
            (X_train_normal - reconstructed_train) ** 2,
            axis=1,
        ),
        "val": np.mean(
            (X_val_normal - reconstructed_val) ** 2,
            axis=1,
        ),
        "normal": np.mean(
            (X_test_normal - reconstructed_normal) ** 2,
            axis=1,
        ),
        "abnormal": np.mean(
            (X_test_abnormal - reconstructed_abnormal) ** 2,
            axis=1,
        ),
    }


# ============================================================
# 7. Visualize normal and abnormal error distributions
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, (name, errors) in zip(axes, reconstruction_errors.items()):
    ax.hist(errors["normal"], bins=40, alpha=0.6, label="Normal test")
    ax.hist(errors["abnormal"], bins=40, alpha=0.6, label="Abnormal test")

    ax.set_title(f"Experiment {name}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Detect anomalies and evaluate performance
# ============================================================

evaluation_results = []

for name, errors in reconstruction_errors.items():
    # Threshold is determined only from normal validation errors.
    threshold = np.percentile(
        errors["val"],
        THRESHOLD_PERCENTILE,
    )

    y_true = np.concatenate([
        np.zeros(len(errors["normal"]), dtype=int),
        np.ones(len(errors["abnormal"]), dtype=int),
    ])

    all_errors = np.concatenate([
        errors["normal"],
        errors["abnormal"],
    ])

    # Higher reconstruction error means more likely to be abnormal.
    y_pred = (all_errors > threshold).astype(int)

    TN, FP, FN, TP = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, all_errors)

    evaluation_results.append({
        "Experiment": name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "ROC-AUC": auc,
    })

evaluation_df = pd.DataFrame(evaluation_results).set_index("Experiment")

print("\nAnomaly detection performance:")
print(evaluation_df.round(4).to_string())


# ============================================================
# 9. Compare reconstruction errors across original classes
# ============================================================

X_test_all = df_test.iloc[:, 1:141].to_numpy(dtype=np.float32)

class_error_results = {}

for name, model in autoencoders.items():
    reconstructed_all = reconstruct(model, X_test_all)

    errors_all = np.mean(
        (X_test_all - reconstructed_all) ** 2,
        axis=1,
    )

    class_errors = pd.DataFrame({
        "class": y_test_original,
        "error": errors_all,
    })

    class_error_results[name] = class_errors

    print(f"\nExperiment {name}: reconstruction errors by original class")
    print(
        class_errors.groupby("class")["error"]
        .agg(["mean", "median", "std", "count"])
        .round(5)
    )


# Plot mean reconstruction error for each original class.
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, (name, class_errors) in zip(axes, class_error_results.items()):
    class_means = class_errors.groupby("class")["error"].mean()

    ax.bar(class_means.index.astype(str), class_means.values)
    ax.set_title(f"Experiment {name}")
    ax.set_xlabel("Original ECG class")
    ax.set_ylabel("Mean reconstruction MSE")
    ax.grid(axis="y", alpha=0.3)

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()


# ============================================================
# 10. Compare reconstruction quality across experiments
# ============================================================

reconstruction_comparison = pd.DataFrame({
    name: {
        "Normal train": errors["train"].mean(),
        "Normal validation": errors["val"].mean(),
        "Normal test": errors["normal"].mean(),
        "Abnormal test": errors["abnormal"].mean(),
    }
    for name, errors in reconstruction_errors.items()
}).T

print("\nReconstruction quality comparison:")
print(reconstruction_comparison.round(6).to_string())


# ============================================================
# 11. Store individual model references
# ============================================================

autoencoder_A = autoencoders["A"]
autoencoder_B = autoencoders["B"]
autoencoder_C = autoencoders["C"]
autoencoder_D = autoencoders["D"]
autoencoder_E = autoencoders["E"]

print("\nAll experiments completed.")
