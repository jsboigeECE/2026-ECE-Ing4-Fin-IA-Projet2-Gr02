"""Phase 2.2 - Evaluate Isolation Forest on test set."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score


def main() -> None:
    results_dir = "results"
    data_path = os.path.join(results_dir, "preprocessed_data.npz")
    outputs_path = os.path.join(results_dir, "isolation_forest_test_outputs.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing file: {data_path}")
    if not os.path.exists(outputs_path):
        raise FileNotFoundError(f"Missing file: {outputs_path}")

    data = np.load(data_path)
    outputs = np.load(outputs_path)

    y_test = data["y_test"]
    y_pred_raw = outputs["y_pred_test"]  # sklearn format: -1 anomaly, 1 normal
    anomaly_score = outputs["anomaly_score_test"]

    # Convert to fraud label format: 1 = fraud/anomaly, 0 = normal
    y_pred = (y_pred_raw == -1).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, anomaly_score)

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, anomaly_score)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "auprc": float(auprc),
    }

    metrics_path = os.path.join(results_dir, "isolation_forest_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(7, 5))
    plt.plot(pr_recall, pr_precision, label=f"Isolation Forest (AUPRC={auprc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Isolation Forest")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    pr_curve_path = os.path.join(results_dir, "isolation_forest_pr_curve.png")
    plt.savefig(pr_curve_path, dpi=150)

    print("Phase 2.2 evaluation completed.")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"PR curve saved to: {pr_curve_path}")


if __name__ == "__main__":
    main()
