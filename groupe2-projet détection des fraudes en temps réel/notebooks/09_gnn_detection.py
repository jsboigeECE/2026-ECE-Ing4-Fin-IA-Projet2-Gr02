"""Phase 5 - GNN (Graph Neural Network) pour la détection de fraude.

Architecture : GCN à 2 couches (Graph Convolutional Network, Kipf & Welling 2017)
Implémenté avec PyTorch Geometric (GCNConv).

Formule GCN :
    H^(l+1) = sigma( A_hat_norm * H^(l) * W^(l) )
    - A_hat_norm = D^(-1/2)(A+I)D^(-1/2)  (adjacence normalisée)
    - H^(0) = X  (features des nœuds)
    - Couche finale : sigmoid -> probabilité de fraude

Étapes :
    5.1  Construction du graphe de transactions
    5.2  Entraînement du GCN
    5.3  Comparaison avec les méthodes précédentes
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import joblib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")
sys.path.insert(0, PROJECT_ROOT)

# Dynamic import of graph_builder (avoids static resolution issues)
spec = importlib.util.spec_from_file_location(
    "graph_builder",
    os.path.join(PROJECT_ROOT, "src", "models", "graph_builder.py"),
)
gb = importlib.util.module_from_spec(spec)
sys.modules["graph_builder"] = gb
spec.loader.exec_module(gb)
build_transaction_graph = gb.build_transaction_graph
graph_stats = gb.graph_stats
save_graph = gb.save_graph


# ---------------------------------------------------------------------------
# PyTorch Geometric GCN model and training
# ---------------------------------------------------------------------------

class PyGGCN(nn.Module):
    """2-layer GCN with a sigmoid fraud head (via logits)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        emb_dim = hidden_dim // 2
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        emb = self.conv2(h, edge_index)
        emb = F.relu(emb)
        logits = self.head(emb).squeeze(-1)
        return logits, emb


def train_pyg_gcn(
    model: PyGGCN,
    data: Data,
    epochs: int = 35,
    learning_rate: float = 5e-4,
    class_weight: dict[int, float] | None = None,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    y = data.y.float()
    if class_weight is not None:
        sample_weight = torch.where(
            data.y == 1,
            torch.tensor(class_weight[1], dtype=torch.float32),
            torch.tensor(class_weight[0], dtype=torch.float32),
        )
    else:
        sample_weight = torch.ones_like(y, dtype=torch.float32)

    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = loss_fn(logits, y)
        loss = (loss * sample_weight).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{epochs}  loss={losses[-1]:.4f}")

    return losses


def predict_proba_pyg(model: PyGGCN, data: Data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        return torch.sigmoid(logits).cpu().numpy()


def generate_embeddings_pyg(model: PyGGCN, data: Data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, emb = model(data.x, data.edge_index)
        return emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def compute_cost(counts: dict[str, int], fp_cost: float = 1.0, fn_cost: float = 25.0) -> float:
    return fp_cost * counts["fp"] + fn_cost * counts["fn"]


def find_best_threshold(y_prob: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
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


def find_threshold_with_precision_floor(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    min_precision: float = 0.5,
) -> tuple[float, float, float]:
    """Find threshold with minimum cost under a precision constraint.

    Returns:
        (threshold, cost, precision)
    """
    thresholds = np.quantile(y_prob, np.linspace(0.60, 0.999, 220))
    best_t = float(thresholds[0])
    best_cost = float("inf")
    best_precision = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        if p < min_precision:
            continue
        counts = confusion_counts(y_true, y_pred)
        cost = compute_cost(counts)
        if cost < best_cost:
            best_cost = float(cost)
            best_t = float(t)
            best_precision = p

    # Fallback when no threshold reaches the precision floor.
    if not np.isfinite(best_cost):
        return 0.5, float("inf"), 0.0
    return best_t, best_cost, best_precision


def metrics_from_scores(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, Any]:
    """Compute classification metrics from anomaly/probability scores."""
    y_pred = (score >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    return {
        "model": model_name,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, score)),
        "cost": float(compute_cost(counts)),
        **counts,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 5 - GNN (Graph Convolutional Network)")
    print("=" * 60)

    # --- Load preprocessed data ---
    data = np.load(os.path.join(RESULTS, "preprocessed_data.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Combine train + test for transductive graph learning
    X_all = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y_all = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    n_train = len(X_train)

    print(f"\nTotal nodes: {len(y_all)}  (train={n_train}, test={len(X_test)})")
    print(f"Fraud nodes: {y_all.sum()}")

    # -----------------------------------------------------------------------
    # Step 5.1 — Build the transaction graph
    # -----------------------------------------------------------------------
    print("\n--- Step 5.1: Building Transaction Graph ---")
    g = build_transaction_graph(
        X_all, y_all,
        time_window=0.03,
        amount_window=0.03,
        max_edges_per_node=8,
        subsample=8_000,    # keep tractable for CPU training
        random_state=42,
    )
    stats = graph_stats(g)
    print(f"\n  Graph statistics:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Save the graph
    save_graph(g, os.path.join(RESULTS, "transaction_graph.npz"))

    # Save stats
    with open(os.path.join(RESULTS, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 5.2 — Train GCN
    # -----------------------------------------------------------------------
    print("\n--- Step 5.2: Training GCN ---")

    input_dim = g.x.shape[1]

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=g.y)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print(f"  Class weights: {class_weight}")

    pyg_data = Data(
        x=torch.tensor(g.x, dtype=torch.float32),
        edge_index=torch.tensor(g.edge_index, dtype=torch.long),
        y=torch.tensor(g.y.astype(np.int64), dtype=torch.long),
    )

    torch.manual_seed(42)
    model = PyGGCN(input_dim=input_dim, hidden_dim=64)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PyGGCN(params={total_params:,}, layers=2)")

    losses = train_pyg_gcn(
        model,
        pyg_data,
        epochs=35,
        learning_rate=5e-4,
        class_weight=class_weight,
    )

    # -----------------------------------------------------------------------
    # Step 5.3 — Evaluation
    # -----------------------------------------------------------------------
    print("\n--- Step 5.3: Evaluation ---")
    y_prob = predict_proba_pyg(model, pyg_data).ravel()

    node_embeddings = generate_embeddings_pyg(model, pyg_data)
    emb_path = os.path.join(RESULTS, "gnn_node_embeddings.npz")
    np.savez_compressed(emb_path, embeddings=node_embeddings, labels=g.y)
    print(f"  Saved: {emb_path}")

    best_t, best_cost = find_best_threshold(y_prob, g.y)
    y_pred = (y_prob >= best_t).astype(int)

    auprc = float(average_precision_score(g.y, y_prob))
    prec = float(precision_score(g.y, y_pred, zero_division=0))
    rec = float(recall_score(g.y, y_pred, zero_division=0))
    counts = confusion_counts(g.y, y_pred)
    cost = compute_cost(counts)

    print(f"\n  GCN Results (threshold={best_t:.3f}):")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    AUPRC     : {auprc:.4f}")
    print(f"    Cost      : {cost:.0f}€  (FP×1 + FN×25)")

    # Operational threshold: reduce false positives via precision floor.
    target_precision = 0.50
    op_t, op_cost, op_precision = find_threshold_with_precision_floor(
        y_prob,
        g.y,
        min_precision=target_precision,
    )
    if np.isfinite(op_cost):
        op_metrics = metrics_from_scores(g.y, y_prob, op_t, "GNN_operational")
        print(
            f"\n  Operational threshold (precision>={target_precision:.2f}): "
            f"t={op_t:.3f}, precision={op_metrics['precision']:.4f}, "
            f"recall={op_metrics['recall']:.4f}, cost={op_metrics['cost']:.0f}€"
        )
    else:
        op_metrics = metrics_from_scores(g.y, y_prob, best_t, "GNN_operational")
        print("\n  No threshold reached the target precision; using cost-optimal threshold.")

    # -----------------------------------------------------------------------
    # Step 5.3 (required): compare GNN vs Isolation Forest vs Autoencoder
    # on the exact same nodes used in the graph sub-sample.
    # -----------------------------------------------------------------------
    print("\n--- Step 5.3: Comparison with Forest and Autoencoder ---")

    # GNN metrics on graph nodes
    gnn_metrics = metrics_from_scores(g.y, y_prob, op_metrics["threshold"], "GNN")

    # Isolation Forest on the same nodes
    if_model = joblib.load(os.path.join(RESULTS, "isolation_forest_model.pkl"))
    if_score = -if_model.score_samples(g.x)
    with open(os.path.join(RESULTS, "isolation_forest_best_threshold.json"), "r", encoding="utf-8") as f:
        if_best = json.load(f)
    if_threshold = float(if_best["best_threshold"])
    if_metrics = metrics_from_scores(g.y, if_score, if_threshold, "IsolationForest")

    # Autoencoder on the same nodes (optional: TensorFlow can fail to import on some setups)
    ae_metrics = None
    try:
        import tensorflow as tf
        from src.models.autoencoder import reconstruction_error

        ae_model = tf.keras.models.load_model(os.path.join(RESULTS, "autoencoder_model.keras"), compile=False)
        ae_score = reconstruction_error(ae_model, g.x)
        with open(os.path.join(RESULTS, "autoencoder_detection_metrics.json"), "r", encoding="utf-8") as f:
            ae_info = json.load(f)
        ae_threshold = float(ae_info["threshold"])
        ae_metrics = metrics_from_scores(g.y, ae_score, ae_threshold, "Autoencoder")
    except Exception as exc:
        print(f"  Autoencoder comparison skipped (TensorFlow unavailable): {exc}")

    comparison = [gnn_metrics, if_metrics]
    if ae_metrics is not None:
        comparison.append(ae_metrics)

    print("\n  Comparison table (same graph nodes):")
    print("  Model            Precision   Recall    AUPRC   Cost €")
    for r in comparison:
        print(f"  {r['model']:<16} {r['precision']:>9.4f} {r['recall']:>8.4f} {r['auprc']:>8.4f} {r['cost']:>8.0f}")

    # Analyze cases detected only by GNN
    if ae_metrics is not None:
        gnn_only_mask = (
            (gnn_metrics["y_pred"] == 1)
            & (if_metrics["y_pred"] == 0)
            & (ae_metrics["y_pred"] == 0)
        )
    else:
        gnn_only_mask = (gnn_metrics["y_pred"] == 1) & (if_metrics["y_pred"] == 0)
    gnn_only_idx = np.where(gnn_only_mask)[0]
    gnn_only_labels = g.y[gnn_only_idx]
    gnn_only_fraud = int(gnn_only_labels.sum())
    gnn_only_total = int(len(gnn_only_idx))

    in_train = int(np.sum(g.node_ids[gnn_only_idx] < n_train))
    in_test = gnn_only_total - in_train

    unique_analysis = {
        "total_gnn_only": gnn_only_total,
        "fraud_in_gnn_only": gnn_only_fraud,
        "normal_in_gnn_only": gnn_only_total - gnn_only_fraud,
        "fraud_rate_gnn_only": float(gnn_only_fraud / max(gnn_only_total, 1)),
        "from_train_split": in_train,
        "from_test_split": in_test,
        "top_gnn_only_node_ids": g.node_ids[gnn_only_idx][:20].tolist(),
    }

    print("\n  GNN-only detections analysis:")
    print(f"    total={unique_analysis['total_gnn_only']}")
    print(f"    fraud={unique_analysis['fraud_in_gnn_only']}  normal={unique_analysis['normal_in_gnn_only']}")
    print(f"    fraud_rate={unique_analysis['fraud_rate_gnn_only']:.4f}")

    # Save comparison table as csv
    cmp_csv = os.path.join(RESULTS, "gnn_comparison_table.csv")
    with open(cmp_csv, "w", encoding="utf-8") as f:
        f.write("model,precision,recall,auprc,cost,threshold,tp,fp,tn,fn\n")
        for r in comparison:
            f.write(
                f"{r['model']},{r['precision']:.6f},{r['recall']:.6f},{r['auprc']:.6f},{r['cost']:.1f},"
                f"{r['threshold']:.6f},{r['tp']},{r['fp']},{r['tn']},{r['fn']}\n"
            )
    print(f"  Saved: {cmp_csv}")

    unique_json = os.path.join(RESULTS, "gnn_unique_cases_analysis.json")
    with open(unique_json, "w", encoding="utf-8") as f:
        json.dump(unique_analysis, f, indent=2)
    print(f"  Saved: {unique_json}")

    # Save threshold analysis for reporting and operational tuning.
    threshold_csv = os.path.join(RESULTS, "gnn_threshold_analysis.csv")
    th_values = np.quantile(y_prob, np.linspace(0.60, 0.999, 180))
    with open(threshold_csv, "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,tp,fp,tn,fn,cost\n")
        for t in th_values:
            yp = (y_prob >= t).astype(int)
            cts = confusion_counts(g.y, yp)
            p = precision_score(g.y, yp, zero_division=0)
            r = recall_score(g.y, yp, zero_division=0)
            c = compute_cost(cts)
            f.write(
                f"{float(t):.6f},{float(p):.6f},{float(r):.6f},"
                f"{cts['tp']},{cts['fp']},{cts['tn']},{cts['fn']},{float(c):.2f}\n"
            )
    print(f"  Saved: {threshold_csv}")

    threshold_json = os.path.join(RESULTS, "gnn_best_threshold.json")
    with open(threshold_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cost_optimal": {
                    "threshold": float(best_t),
                    "precision": float(prec),
                    "recall": float(rec),
                    "cost": float(cost),
                },
                "operational_precision_constrained": {
                    "target_precision": float(target_precision),
                    "threshold": float(op_metrics["threshold"]),
                    "precision": float(op_metrics["precision"]),
                    "recall": float(op_metrics["recall"]),
                    "cost": float(op_metrics["cost"]),
                    "tp": int(op_metrics["tp"]),
                    "fp": int(op_metrics["fp"]),
                    "tn": int(op_metrics["tn"]),
                    "fn": int(op_metrics["fn"]),
                },
            },
            f,
            indent=2,
        )
    print(f"  Saved: {threshold_json}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1 - Training loss curve
    axes[0].plot(losses, color="#E91E63", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("GCN Training Loss")
    axes[0].grid(alpha=0.3)

    # 2 - Precision-Recall curve
    prec_c, rec_c, _ = precision_recall_curve(g.y, y_prob)
    axes[1].plot(rec_c, prec_c, color="#E91E63", label=f"GCN (AUPRC={auprc:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — GCN")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3 - Confusion matrix
    cm = confusion_matrix(g.y, y_pred, labels=[0, 1])
    im = axes[2].imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Normal", "Fraud"])
    axes[2].set_yticklabels(["Normal", "Fraud"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    axes[2].set_title("Confusion Matrix — GCN")
    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > thresh_val else "black", fontsize=11)

    plt.suptitle("GCN Fraud Detection — Phase 5", fontsize=13)
    plt.tight_layout()
    out_fig = os.path.join(RESULTS, "gnn_results.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_fig}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    result = {
        "model": "GCN",
        "num_nodes": g.num_nodes,
        "num_edges": g.num_edges,
        "graph_stats": stats,
        "threshold": float(best_t),
        "operational_threshold": float(op_metrics["threshold"]),
        "precision": prec,
        "recall": rec,
        "auprc": auprc,
        "cost": cost,
        "operational_precision": float(op_metrics["precision"]),
        "operational_recall": float(op_metrics["recall"]),
        "operational_cost": float(op_metrics["cost"]),
        "comparison_same_nodes": [
            {
                "model": r["model"],
                "threshold": r["threshold"],
                "precision": r["precision"],
                "recall": r["recall"],
                "auprc": r["auprc"],
                "cost": r["cost"],
                "tp": r["tp"],
                "fp": r["fp"],
                "tn": r["tn"],
                "fn": r["fn"],
            }
            for r in comparison
        ],
        "gnn_only_cases": unique_analysis,
        **counts,
    }
    out_json = os.path.join(RESULTS, "gnn_metrics.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json}")

    print("\nPhase 5 completed successfully.")


if __name__ == "__main__":
    main()
