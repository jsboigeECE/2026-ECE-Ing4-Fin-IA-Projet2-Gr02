"""Phase 3.3 - Comparison: Autoencoder vs Isolation Forest."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def main() -> None:
    # --- Load shared labels ---
    data = np.load(os.path.join(RESULTS, "preprocessed_data.npz"))
    y_test = data["y_test"].astype(int)

    # --- Load Isolation Forest results ---
    if_outputs = np.load(os.path.join(RESULTS, "isolation_forest_test_outputs.npz"))
    if_score = if_outputs["anomaly_score_test"]
    if_best = load_json(os.path.join(RESULTS, "isolation_forest_best_threshold.json"))
    if_pred = (if_score >= if_best["best_threshold"]).astype(int)

    # --- Load Autoencoder results ---
    ae_outputs = np.load(os.path.join(RESULTS, "autoencoder_test_outputs.npz"))
    ae_score = ae_outputs["reconstruction_error_test"]
    ae_threshold = float(ae_outputs["threshold"][0])
    ae_pred = ae_outputs["y_pred_test"].astype(int)

    # --- Metrics ---
    models = {
        "Isolation Forest": (if_score, if_pred),
        "Autoencoder":      (ae_score, ae_pred),
    }

    print("=" * 55)
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'AUPRC':>10}")
    print("-" * 55)

    pr_data = {}
    metrics_summary = {}

    for name, (score, pred) in models.items():
        tp, fp, tn, fn = confusion_counts(y_test, pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        auprc     = float(average_precision_score(y_test, score))
        pr_p, pr_r, _ = precision_recall_curve(y_test, score)
        pr_data[name] = (pr_r, pr_p)
        metrics_summary[name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "auprc":     round(auprc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        print(f"{name:<20} {precision:>10.4f} {recall:>10.4f} {auprc:>10.4f}")

    print("=" * 55)

    # --- PR curve comparison ---
    plt.figure(figsize=(8, 5))
    colors = {"Isolation Forest": "steelblue", "Autoencoder": "darkorange"}
    for name, (pr_r, pr_p) in pr_data.items():
        auprc = metrics_summary[name]["auprc"]
        plt.plot(pr_r, pr_p, label=f"{name} (AUPRC={auprc:.4f})", color=colors[name])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: AE vs Isolation Forest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(RESULTS, "comparison_pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, (_, pred)) in zip(axes, models.items()):
        m = metrics_summary[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
        ax.set_title(name)
    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS, "comparison_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # --- Save summary JSON ---
    summary_path = os.path.join(RESULTS, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved: {pr_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
