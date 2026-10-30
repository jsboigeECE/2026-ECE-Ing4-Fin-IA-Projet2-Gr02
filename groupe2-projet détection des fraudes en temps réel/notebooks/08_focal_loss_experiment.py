"""Phase 4.3 - Focal Loss classifier for fraud detection.

Focal Loss down-weights easy examples (normal transactions) and focuses
training on hard examples (fraud), naturally handling class imbalance.

Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
  - p_t : predicted probability for the true class
  - gamma : focusing parameter (default 2)
  - alpha : class balancing factor
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def compute_cost(counts: dict[str, int], fp_cost: float = 1.0, fn_cost: float = 25.0) -> float:
    return fp_cost * counts["fp"] + fn_cost * counts["fn"]


# ---------------------------------------------------------------------------
# Focal Loss (custom Keras loss)
# ---------------------------------------------------------------------------

def make_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Return a focal loss function for binary classification.

    Args:
        gamma: focusing parameter. Higher = more focus on hard examples.
        alpha: weight for the positive class (fraud).
    """
    def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross-entropy for each sample
        ce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # p_t: probability of the correct class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # alpha_t: balancing factor per class
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        # Focal weight
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)

        return tf.reduce_mean(focal_weight * ce)

    focal_loss.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return focal_loss


# ---------------------------------------------------------------------------
# Classifier architecture
# ---------------------------------------------------------------------------

def build_classifier(input_dim: int) -> tf.keras.Model:
    """Simple dense classifier for binary fraud detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="fraud_classifier")
    return model


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss_fn: Any,
    loss_name: str,
    sample_weight: np.ndarray | None = None,
    epochs: int = 20,
    batch_size: int = 1024,
) -> tuple[tf.keras.Model, Any]:
    model = build_classifier(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print(f"\n  Training classifier with {loss_name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        sample_weight=sample_weight,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ],
    )
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, Any]:
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    auprc = float(average_precision_score(y_test, y_prob))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    counts = confusion_counts(y_test, y_pred)
    cost = compute_cost(counts)

    print(f"  [{label}] precision={prec:.4f}  recall={rec:.4f}  AUPRC={auprc:.4f}  cost={cost:.0f}€")
    return {
        "model": label,
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "auprc": auprc,
        "cost": cost,
        **counts,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def find_best_threshold(y_prob: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Find threshold that minimises cost FP*1 + FN*25."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        counts = confusion_counts(y_true, y_p)
        c = compute_cost(counts)
        if c < best_cost:
            best_cost = c
            best_t = t
    return best_t, best_cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 4.3 - Focal Loss Experiment")
    print("=" * 60)

    # --- Load data ---
    data = load_npz(os.path.join(RESULTS, "preprocessed_data.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    input_dim = X_train.shape[1]

    print(f"\nTrain: {X_train.shape[0]} samples  ({int(y_train.sum())} frauds)")
    print(f"Test : {X_test.shape[0]} samples  ({int(y_test.sum())} frauds)")

    # --- Class weights (for comparison baseline) ---
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"\nClass weights: {class_weight_dict}")

    sample_weight = np.where(y_train == 1, cw[1], cw[0])

    # ---------------------------------------------------------------
    # Experiment 1 — Baseline: Binary Cross-Entropy (no weighting)
    # ---------------------------------------------------------------
    print("\n--- Experiment 1: Binary Cross-Entropy (baseline) ---")
    model_bce, _ = train_classifier(
        X_train, y_train,
        loss_fn="binary_crossentropy",
        loss_name="Binary Cross-Entropy",
        epochs=20,
    )
    t_bce, _ = find_best_threshold(
        model_bce.predict(X_test, verbose=0).ravel(), y_test
    )
    results_bce = evaluate_classifier(model_bce, X_test, y_test, threshold=t_bce, label="BCE baseline")

    # ---------------------------------------------------------------
    # Experiment 2 — Weighted Binary Cross-Entropy
    # ---------------------------------------------------------------
    print("\n--- Experiment 2: Weighted Binary Cross-Entropy ---")
    model_wbce, _ = train_classifier(
        X_train, y_train,
        loss_fn="binary_crossentropy",
        loss_name="Weighted BCE",
        sample_weight=sample_weight,
        epochs=20,
    )
    t_wbce, _ = find_best_threshold(
        model_wbce.predict(X_test, verbose=0).ravel(), y_test
    )
    results_wbce = evaluate_classifier(model_wbce, X_test, y_test, threshold=t_wbce, label="Weighted BCE")

    # ---------------------------------------------------------------
    # Experiment 3 — Focal Loss (gamma=2, alpha=0.25)
    # ---------------------------------------------------------------
    gamma, alpha = 2.0, 0.25
    print(f"\n--- Experiment 3: Focal Loss (gamma={gamma}, alpha={alpha}) ---")
    focal_fn = make_focal_loss(gamma=gamma, alpha=alpha)
    model_fl, _ = train_classifier(
        X_train, y_train,
        loss_fn=focal_fn,
        loss_name=f"Focal Loss g={gamma} a={alpha}",
        epochs=20,
    )
    t_fl, _ = find_best_threshold(
        model_fl.predict(X_test, verbose=0).ravel(), y_test
    )
    results_fl = evaluate_classifier(model_fl, X_test, y_test, threshold=t_fl, label=f"Focal Loss g{gamma}")

    # ---------------------------------------------------------------
    # Experiment 4 — Focal Loss with higher gamma (gamma=5)
    # ---------------------------------------------------------------
    gamma2, alpha2 = 5.0, 0.25
    print(f"\n--- Experiment 4: Focal Loss (gamma={gamma2}, alpha={alpha2}) ---")
    focal_fn2 = make_focal_loss(gamma=gamma2, alpha=alpha2)
    model_fl2, _ = train_classifier(
        X_train, y_train,
        loss_fn=focal_fn2,
        loss_name=f"Focal Loss g={gamma2} a={alpha2}",
        epochs=20,
    )
    t_fl2, _ = find_best_threshold(
        model_fl2.predict(X_test, verbose=0).ravel(), y_test
    )
    results_fl2 = evaluate_classifier(model_fl2, X_test, y_test, threshold=t_fl2, label=f"Focal Loss g{gamma2}")

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    all_results = [results_bce, results_wbce, results_fl, results_fl2]

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Model':<28} {'Precision':>10} {'Recall':>8} {'AUPRC':>8} {'Cost €':>8}"
    print(header)
    print("-" * 65)
    for r in all_results:
        print(f"{r['model']:<28} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['auprc']:>8.4f} {r['cost']:>8.0f}")

    # ---------------------------------------------------------------
    # Precision-Recall curves comparison
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colours = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
    labels_for_plot = [r["model"] for r in all_results]

    ax = axes[0]
    for r, c, lbl in zip(all_results, colours, labels_for_plot):
        prec_c, rec_c, _ = precision_recall_curve(y_test, r["y_prob"])
        ax.plot(rec_c, prec_c, color=c, label=f"{lbl} (AUPRC={r['auprc']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Focal Loss Comparison")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Bar chart AUPRC
    ax2 = axes[1]
    x = np.arange(len(all_results))
    auprcs = [r["auprc"] for r in all_results]
    bars = ax2.bar(x, auprcs, color=colours, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["model"] for r in all_results], rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("AUPRC")
    ax2.set_title("AUPRC Comparison")
    for bar, val in zip(bars, auprcs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.4f}", ha="center", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_pr = os.path.join(RESULTS, "focal_loss_comparison.png")
    plt.savefig(out_pr, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_pr}")

    # ---------------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    cmap = plt.cm.Blues

    for ax_i, r in zip(axes2, all_results):
        cm = confusion_matrix(y_test, r["y_pred"], labels=[0, 1])
        im = ax_i.imshow(cm, interpolation="nearest", cmap=cmap)
        fig2.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
        tick_marks = np.arange(2)
        ax_i.set_xticks(tick_marks)
        ax_i.set_yticks(tick_marks)
        ax_i.set_xticklabels(["Normal", "Fraud"])
        ax_i.set_yticklabels(["Normal", "Fraud"])
        ax_i.set_xlabel("Predicted")
        ax_i.set_ylabel("True")
        ax_i.set_title(r["model"], fontsize=9)
        thresh_cm = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax_i.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                          color="white" if cm[i, j] > thresh_cm else "black", fontsize=10)

    plt.suptitle("Confusion Matrices — Focal Loss Experiment", fontsize=12)
    plt.tight_layout()
    out_cm = os.path.join(RESULTS, "focal_loss_confusion_matrices.png")
    plt.savefig(out_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_cm}")

    # ---------------------------------------------------------------
    # Save JSON summary
    # ---------------------------------------------------------------
    summary = []
    for r in all_results:
        summary.append({
            "model": r["model"],
            "threshold": float(r["threshold"]),
            "precision": r["precision"],
            "recall": r["recall"],
            "auprc": r["auprc"],
            "cost": r["cost"],
            "tp": r["tp"],
            "fp": r["fp"],
            "tn": r["tn"],
            "fn": r["fn"],
        })

    out_json = os.path.join(RESULTS, "focal_loss_summary.json")
    with open(out_json, "w") as f:
        json.dump({"gamma_tested": [gamma, gamma2], "alpha": alpha, "results": summary}, f, indent=2)
    print(f"Saved: {out_json}")

    # Save raw probabilities for full ROC reconstruction in Phase 7.
    out_scores = os.path.join(RESULTS, "focal_loss_test_scores.npz")
    np.savez_compressed(
        out_scores,
        y_test=y_test.astype(np.int8),
        bce_prob=results_bce["y_prob"].astype(np.float32),
        wbce_prob=results_wbce["y_prob"].astype(np.float32),
        fl2_prob=results_fl["y_prob"].astype(np.float32),
        fl5_prob=results_fl2["y_prob"].astype(np.float32),
        bce_threshold=np.float32(results_bce["threshold"]),
        wbce_threshold=np.float32(results_wbce["threshold"]),
        fl2_threshold=np.float32(results_fl["threshold"]),
        fl5_threshold=np.float32(results_fl2["threshold"]),
    )
    print(f"Saved: {out_scores}")

    # Best model by AUPRC
    best = max(all_results, key=lambda r: r["auprc"])
    print(f"\nBest model by AUPRC: {best['model']} (AUPRC={best['auprc']:.4f})")

    print("\nPhase 4.3 completed successfully.")


if __name__ == "__main__":
    main()
