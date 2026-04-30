"""Phase 4.2 - Class weights impact on Isolation Forest and Autoencoder."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")


def load_npz(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def best_cost_threshold(
    score: np.ndarray,
    y_true: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 25.0,
) -> tuple[float, np.ndarray, float]:
    thresholds = np.quantile(score, np.linspace(0.70, 0.999, 120))
    best_t = float(thresholds[0])
    best_cost = float("inf")
    best_pred = np.zeros_like(y_true)

    for t in thresholds:
        y_pred = (score >= t).astype(int)
        counts = confusion_counts(y_true, y_pred)
        cost = fp_cost * counts["fp"] + fn_cost * counts["fn"]
        if cost < best_cost:
            best_cost = float(cost)
            best_t = float(t)
            best_pred = y_pred

    return best_t, best_pred, best_cost


def build_autoencoder(input_dim: int, learning_rate: float = 1e-3) -> Any:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def reconstruction_error(model: Any, X: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    counts = confusion_counts(y_true, y_pred)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, score)),
        **counts,
    }


def save_csv(rows: list[dict[str, Any]], path: str) -> None:
    headers = ["model", "variant", "precision", "recall", "auprc", "tp", "fp", "tn", "fn"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['variant']},{r['precision']},{r['recall']},{r['auprc']},"
                f"{r['tp']},{r['fp']},{r['tn']},{r['fn']}\n"
            )


def main() -> None:
    data = load_npz(os.path.join(RESULTS, "preprocessed_data.npz"))
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    classes = np.array([0, 1], dtype=int)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_array)}
    sample_weight = np.where(y_train == 1, class_weights[1], class_weights[0]).astype(float)

    # -------------------------
    # Baselines (from saved outputs)
    # -------------------------
    if_outputs = load_npz(os.path.join(RESULTS, "isolation_forest_test_outputs.npz"))
    if_score_base = if_outputs["anomaly_score_test"]
    if_best = json.load(open(os.path.join(RESULTS, "isolation_forest_best_threshold.json"), "r", encoding="utf-8"))
    if_pred_base = (if_score_base >= float(if_best["best_threshold"])).astype(int)

    ae_outputs = load_npz(os.path.join(RESULTS, "autoencoder_test_outputs.npz"))
    ae_score_base = ae_outputs["reconstruction_error_test"]
    ae_pred_base = ae_outputs["y_pred_test"].astype(int)

    # -------------------------
    # Isolation Forest with class weights
    # -------------------------
    contamination = max(1e-4, float(np.mean(y_train)))
    if_model_w = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    # Some sklearn versions accept sample_weight; fallback if not supported.
    try:
        if_model_w.fit(X_train, sample_weight=sample_weight)
    except TypeError:
        if_model_w.fit(X_train)

    if_score_w = -if_model_w.score_samples(X_test)
    if_t_w, if_pred_w, if_cost_w = best_cost_threshold(if_score_w, y_test)

    # -------------------------
    # Autoencoder with class weights
    # -------------------------
    tf.keras.utils.set_random_seed(42)
    ae_model_w = build_autoencoder(input_dim=X_train.shape[1])

    ae_model_w.fit(
        X_train,
        X_train,
        sample_weight=sample_weight,
        epochs=8,
        batch_size=256,
        validation_split=0.1,
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=0,
    )

    train_err_normal_w = reconstruction_error(ae_model_w, X_train[y_train == 0])
    ae_t_w = float(np.quantile(train_err_normal_w, 0.95))
    ae_score_w = reconstruction_error(ae_model_w, X_test)
    ae_pred_w = (ae_score_w >= ae_t_w).astype(int)

    # Save weighted autoencoder model
    ae_model_w.save(os.path.join(RESULTS, "autoencoder_model_class_weights.keras"))

    rows = []

    if_base = metrics_block(y_test, if_pred_base, if_score_base)
    if_w = metrics_block(y_test, if_pred_w, if_score_w)
    ae_base = metrics_block(y_test, ae_pred_base, ae_score_base)
    ae_w = metrics_block(y_test, ae_pred_w, ae_score_w)

    rows.append({"model": "IsolationForest", "variant": "baseline", **if_base})
    rows.append({"model": "IsolationForest", "variant": "class_weights", **if_w})
    rows.append({"model": "Autoencoder", "variant": "baseline", **ae_base})
    rows.append({"model": "Autoencoder", "variant": "class_weights", **ae_w})

    summary = {
        "class_weights": class_weights,
        "thresholds": {
            "isolation_forest_class_weights": if_t_w,
            "autoencoder_class_weights": ae_t_w,
        },
        "cost_analysis": {
            "isolation_forest_class_weights_best_cost": if_cost_w,
            "cost_assumptions": {"fp_cost": 1.0, "fn_cost": 25.0},
        },
        "results": rows,
    }

    summary_path = os.path.join(RESULTS, "class_weights_impact_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(RESULTS, "class_weights_impact_table.csv")
    save_csv(rows, csv_path)

    # Confusion matrices for weighted models only (easy visual for impact).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mats = [
        ("Isolation Forest (class weights)", confusion_counts(y_test, if_pred_w)),
        ("Autoencoder (class weights)", confusion_counts(y_test, ae_pred_w)),
    ]
    for ax, (title, c) in zip(axes, mats):
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.suptitle("Class Weights Impact - Confusion Matrices")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS, "class_weights_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("Phase 4.2 completed successfully.")
    print(f"Class weights: {class_weights}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
