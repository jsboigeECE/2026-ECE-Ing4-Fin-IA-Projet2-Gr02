"""Phase 2.3 - Threshold selection and cost analysis for Isolation Forest."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


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

    y_test = data["y_test"].astype(int)
    anomaly_score = outputs["anomaly_score_test"]

    # Business cost assumptions (editable).
    fp_cost = 1.0   # cost of blocking a legitimate transaction
    fn_cost = 25.0  # cost of missing a fraud

    thresholds = np.quantile(anomaly_score, np.linspace(0.70, 0.999, 150))

    rows = []
    best = None

    for t in thresholds:
        y_pred = (anomaly_score >= t).astype(int)
        tp, fp, tn, fn = confusion_counts(y_test, y_pred)

        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        total_cost = fp_cost * fp + fn_cost * fn

        row = {
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_cost": float(total_cost),
        }
        rows.append(row)

        if best is None or row["total_cost"] < best["total_cost"]:
            best = row

    assert best is not None

    # Save table as CSV (without pandas dependency).
    csv_path = os.path.join(results_dir, "isolation_forest_threshold_analysis.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["threshold", "precision", "recall", "tp", "fp", "tn", "fn", "total_cost"]
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(
                f"{r['threshold']},{r['precision']},{r['recall']},{r['tp']},{r['fp']},{r['tn']},{r['fn']},{r['total_cost']}\n"
            )

    best_payload = {
        "best_threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "tp": best["tp"],
        "fp": best["fp"],
        "tn": best["tn"],
        "fn": best["fn"],
        "total_cost": best["total_cost"],
        "cost_assumptions": {"fp_cost": fp_cost, "fn_cost": fn_cost},
    }

    best_path = os.path.join(results_dir, "isolation_forest_best_threshold.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    # Plot cost vs threshold and mark best point.
    th = [r["threshold"] for r in rows]
    costs = [r["total_cost"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(th, costs, label="Total Cost")
    plt.scatter([best["threshold"]], [best["total_cost"]], color="red", label="Best threshold")
    plt.xlabel("Threshold on anomaly score")
    plt.ylabel("Total Cost")
    plt.title("Cost vs Threshold - Isolation Forest")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    cost_curve_path = os.path.join(results_dir, "isolation_forest_cost_curve.png")
    plt.savefig(cost_curve_path, dpi=150)

    print("Phase 2.3 threshold analysis completed.")
    print(f"Best threshold: {best['threshold']:.6f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"Cost: {best['total_cost']:.2f}")
    print(f"Saved: {best_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {cost_curve_path}")


if __name__ == "__main__":
    main()
