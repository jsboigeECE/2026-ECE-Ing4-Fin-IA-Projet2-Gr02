"""Phase 3.2 - Autoencoder detection using reconstruction error."""

from __future__ import annotations

import json
import os
import sys
import importlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_score, recall_score

# Allow imports from project root when running this script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_autoencoder_utils():
    module = importlib.import_module("src.models.autoencoder")
    return module.reconstruction_error, module.train_autoencoder


def main() -> None:
    reconstruction_error, train_autoencoder = _load_autoencoder_utils()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")
    data_path = os.path.join(results_dir, "preprocessed_data.npz")
    model_path = os.path.join(results_dir, "autoencoder_model.keras")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing file: {data_path}")

    # Train the model if it does not exist yet.
    if not os.path.exists(model_path):
        print("Autoencoder model not found. Training now...")
        train_autoencoder(data_path=data_path, output_dir=results_dir)

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"].astype(int)
    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)

    X_train_normal = X_train[y_train == 0]

    model = tf.keras.models.load_model(model_path)

    # Compute reconstruction error for thresholding and test detection.
    train_error_normal = reconstruction_error(model, X_train_normal)
    test_error = reconstruction_error(model, X_test)

    # Unsupervised threshold: 95th percentile of normal-train errors.
    threshold = float(np.quantile(train_error_normal, 0.95))

    # Fraud label convention: 1 = fraud/anomaly, 0 = normal
    y_pred_test = (test_error >= threshold).astype(int)

    precision = float(precision_score(y_test, y_pred_test, zero_division=0))
    recall = float(recall_score(y_test, y_pred_test, zero_division=0))
    auprc = float(average_precision_score(y_test, test_error))

    mean_error_normal = float(np.mean(test_error[y_test == 0]))
    mean_error_fraud = float(np.mean(test_error[y_test == 1]))

    outputs_path = os.path.join(results_dir, "autoencoder_test_outputs.npz")
    np.savez_compressed(
        outputs_path,
        y_pred_test=y_pred_test,
        reconstruction_error_test=test_error,
        threshold=np.array([threshold], dtype=float),
    )

    metrics = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
        "mean_error_normal": mean_error_normal,
        "mean_error_fraud": mean_error_fraud,
    }

    metrics_path = os.path.join(results_dir, "autoencoder_detection_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(test_error[y_test == 0], bins=100, alpha=0.6, label="Normal", density=True)
    plt.hist(test_error[y_test == 1], bins=100, alpha=0.6, label="Fraud", density=True)
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Autoencoder Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "autoencoder_error_distribution.png")
    plt.savefig(plot_path, dpi=150)

    print("Phase 3.2 completed successfully.")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Mean error (normal): {mean_error_normal:.6f}")
    print(f"Mean error (fraud):  {mean_error_fraud:.6f}")
    print(f"Saved: {outputs_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
