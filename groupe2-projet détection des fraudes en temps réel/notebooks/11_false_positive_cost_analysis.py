"""Phase 7.2 - False positive cost analysis across all methods.

Produces:
- Cost estimation per false positive for each method
- Mapping of FP by amount range, time of day
- Optimal threshold proposals to reduce FP cost
- Saved to results/fp_cost_analysis_*.{json,png}
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve  # noqa: F401 (available in global Python)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> dict:
    data = np.load(os.path.join(RESULTS_DIR, "preprocessed_data.npz"))
    X_test = data["X_test"]
    y_test  = data["y_test"]

    if_out = np.load(os.path.join(RESULTS_DIR, "isolation_forest_test_outputs.npz"))
    ae_out = np.load(os.path.join(RESULTS_DIR, "autoencoder_test_outputs.npz"))

    with open(os.path.join(RESULTS_DIR, "isolation_forest_best_threshold.json"), encoding="utf-8") as f:
        if_thresh = json.load(f)["best_threshold"]
    with open(os.path.join(RESULTS_DIR, "autoencoder_detection_metrics.json"), encoding="utf-8") as f:
        ae_thresh = json.load(f)["threshold"]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "if_scores": if_out["anomaly_score_test"],
        "ae_scores": ae_out["reconstruction_error_test"],
        "if_thresh": if_thresh,
        "ae_thresh": ae_thresh,
    }


# ---------------------------------------------------------------------------
# 2. Cost functions
# ---------------------------------------------------------------------------
FP_COST = 1.0   # coût d'un faux positif (transaction bloquée à tort)
FN_COST = 25.0  # coût d'un faux négatif (fraude non détectée)


def compute_costs_at_threshold(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    total = len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cost = fp * FP_COST + fn * FN_COST
    cost_per_fp = cost / fp if fp > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "total_cost": round(cost, 2),
        "fp_count": fp,
        "fn_count": fn,
        "fp_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        "avg_cost_per_transaction": round(cost / total, 4),
    }


# ---------------------------------------------------------------------------
# 3. FP cartography by Amount and Time
# ---------------------------------------------------------------------------
def get_fp_indices(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> np.ndarray:
    preds = (scores >= threshold).astype(int)
    return np.where((preds == 1) & (y_true == 0))[0]


def cartography_plot(X_test: np.ndarray, y_test: np.ndarray,
                     if_scores: np.ndarray, ae_scores: np.ndarray,
                     if_thresh: float, ae_thresh: float) -> None:
    """Plot FP distribution by Amount (col index 29 in original = last col after normalization).
    Note: X_test columns are PCA-transformed (V1-V28) + normalized Amount + normalized Time.
    Amount = last col, Time = second-to-last col."""
    amount_col = X_test.shape[1] - 1  # normalized Amount
    time_col   = X_test.shape[1] - 2  # normalized Time

    if_fp_idx = get_fp_indices(if_scores, y_test, if_thresh)
    ae_fp_idx = get_fp_indices(ae_scores, y_test, ae_thresh)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Histogram: FP amounts (normalized) ---
    ax = axes[0, 0]
    ax.hist(X_test[if_fp_idx, amount_col], bins=40, alpha=0.6, color="tab:blue", label=f"IF FPs (n={len(if_fp_idx)})")
    ax.hist(X_test[ae_fp_idx, amount_col], bins=40, alpha=0.6, color="tab:orange", label=f"AE FPs (n={len(ae_fp_idx)})")
    ax.set_xlabel("Montant normalisé")
    ax.set_ylabel("Nombre de transactions FP")
    ax.set_title("Distribution des FP par montant")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Histogram: FP by time ---
    ax = axes[0, 1]
    ax.hist(X_test[if_fp_idx, time_col], bins=40, alpha=0.6, color="tab:blue", label=f"IF FPs")
    ax.hist(X_test[ae_fp_idx, time_col], bins=40, alpha=0.6, color="tab:orange", label=f"AE FPs")
    ax.set_xlabel("Temps normalisé")
    ax.set_ylabel("Nombre de transactions FP")
    ax.set_title("Distribution des FP par temps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Scatter: Amount vs Score for FPs ---
    ax = axes[1, 0]
    ax.scatter(X_test[if_fp_idx, amount_col], if_scores[if_fp_idx],
               alpha=0.4, s=15, color="tab:blue", label="IF FPs")
    ax.axhline(if_thresh, color="tab:blue", linestyle="--", linewidth=1.2, label=f"Seuil IF={if_thresh:.3f}")
    ax.set_xlabel("Montant normalisé")
    ax.set_ylabel("Score Isolation Forest")
    ax.set_title("IF — Score vs Montant pour les FPs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(X_test[ae_fp_idx, amount_col], ae_scores[ae_fp_idx],
               alpha=0.4, s=15, color="tab:orange", label="AE FPs")
    ax.axhline(ae_thresh, color="tab:orange", linestyle="--", linewidth=1.2, label=f"Seuil AE={ae_thresh:.3f}")
    ax.set_xlabel("Montant normalisé")
    ax.set_ylabel("Score Autoencoder (reconstruction error)")
    ax.set_title("AE — Score vs Montant pour les FPs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cartographie des Faux Positifs — Phase 7.2", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fp_cost_cartography.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"FP cartography saved: {out}")


# ---------------------------------------------------------------------------
# 4. Optimal threshold search
# ---------------------------------------------------------------------------
def find_optimal_thresholds(scores: np.ndarray, y_true: np.ndarray,
                             method_name: str, n_steps: int = 200) -> dict:
    """Sweep thresholds and find cost-optimal + precision-constrained optima."""
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    costs, precisions, recalls, fps, fns = [], [], [], [], []

    for t in thresholds:
        r = compute_costs_at_threshold(scores, y_true, t)
        costs.append(r["total_cost"])
        precisions.append(r["precision"])
        recalls.append(r["recall"])
        fps.append(r["fp_count"])
        fns.append(r["fn_count"])

    costs = np.array(costs)
    precisions = np.array(precisions)

    # Cost-optimal
    best_cost_idx = int(np.argmin(costs))
    # Precision ≥ 0.5 constraint
    valid_prec = np.where(precisions >= 0.5)[0]
    best_prec_idx = int(valid_prec[np.argmin(costs[valid_prec])]) if len(valid_prec) > 0 else best_cost_idx

    return {
        "method": method_name,
        "cost_optimal_threshold": round(float(thresholds[best_cost_idx]), 4),
        "cost_optimal_cost": round(float(costs[best_cost_idx]), 2),
        "cost_optimal_precision": round(float(precisions[best_cost_idx]), 4),
        "cost_optimal_recall": round(float(recalls[best_cost_idx]), 4),
        "cost_optimal_fp": int(fps[best_cost_idx]),
        "cost_optimal_fn": int(fns[best_cost_idx]),
        "precision_constrained_threshold": round(float(thresholds[best_prec_idx]), 4),
        "precision_constrained_cost": round(float(costs[best_prec_idx]), 2),
        "precision_constrained_precision": round(float(precisions[best_prec_idx]), 4),
        "precision_constrained_recall": round(float(recalls[best_prec_idx]), 4),
        "precision_constrained_fp": int(fps[best_prec_idx]),
        "precision_constrained_fn": int(fns[best_prec_idx]),
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
        "precisions": precisions.tolist(),
        "recalls": list(recalls),
    }


def plot_threshold_sweep(if_res: dict, ae_res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res, color, name in [
        (axes[0], if_res, "tab:blue", "Isolation Forest"),
        (axes[1], ae_res, "tab:orange", "Autoencoder"),
    ]:
        thresholds = res["thresholds"]
        costs = res["costs"]
        ax.plot(thresholds, costs, color=color, linewidth=2)
        ax.axvline(res["cost_optimal_threshold"], color="green", linestyle="--",
                   label=f"Coût-opt: t={res['cost_optimal_threshold']:.3f} ({res['cost_optimal_cost']:.0f}€)")
        ax.axvline(res["precision_constrained_threshold"], color="red", linestyle="--",
                   label=f"Prec≥0.5: t={res['precision_constrained_threshold']:.3f} ({res['precision_constrained_cost']:.0f}€)")
        ax.set_xlabel("Seuil")
        ax.set_ylabel("Coût total (€)")
        ax.set_title(f"{name} — Coût vs Seuil")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Analyse des Seuils Optimaux — Phase 7.2", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fp_threshold_analysis.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Threshold analysis saved: {out}")


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
def print_and_save_summary(if_res: dict, ae_res: dict, if_base: dict, ae_base: dict) -> None:
    summary = {
        "cost_assumptions": {"fp_cost": FP_COST, "fn_cost": FN_COST},
        "isolation_forest": {
            "current": if_base,
            "optimal": {
                "cost_optimal": {
                    "threshold": if_res["cost_optimal_threshold"],
                    "cost": if_res["cost_optimal_cost"],
                    "precision": if_res["cost_optimal_precision"],
                    "recall": if_res["cost_optimal_recall"],
                    "fp": if_res["cost_optimal_fp"],
                    "fn": if_res["cost_optimal_fn"],
                },
                "precision_constrained": {
                    "threshold": if_res["precision_constrained_threshold"],
                    "cost": if_res["precision_constrained_cost"],
                    "precision": if_res["precision_constrained_precision"],
                    "recall": if_res["precision_constrained_recall"],
                    "fp": if_res["precision_constrained_fp"],
                    "fn": if_res["precision_constrained_fn"],
                },
            },
        },
        "autoencoder": {
            "current": ae_base,
            "optimal": {
                "cost_optimal": {
                    "threshold": ae_res["cost_optimal_threshold"],
                    "cost": ae_res["cost_optimal_cost"],
                    "precision": ae_res["cost_optimal_precision"],
                    "recall": ae_res["cost_optimal_recall"],
                    "fp": ae_res["cost_optimal_fp"],
                    "fn": ae_res["cost_optimal_fn"],
                },
                "precision_constrained": {
                    "threshold": ae_res["precision_constrained_threshold"],
                    "cost": ae_res["precision_constrained_cost"],
                    "precision": ae_res["precision_constrained_precision"],
                    "recall": ae_res["precision_constrained_recall"],
                    "fp": ae_res["precision_constrained_fp"],
                    "fn": ae_res["precision_constrained_fn"],
                },
            },
        },
    }

    out_path = os.path.join(RESULTS_DIR, "fp_cost_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"FP cost analysis saved: {out_path}")

    print("\n=== Résumé Analyse des FP — Isolation Forest ===")
    print(f"  Seuil actuel : {if_base['threshold']:.4f} | FP={if_base['fp_count']} | Coût={if_base['total_cost']:.0f}€")
    print(f"  Seuil coût-optimal     : {if_res['cost_optimal_threshold']:.4f} | FP={if_res['cost_optimal_fp']} | Coût={if_res['cost_optimal_cost']:.0f}€")
    print(f"  Seuil precision≥0.5    : {if_res['precision_constrained_threshold']:.4f} | FP={if_res['precision_constrained_fp']} | Coût={if_res['precision_constrained_cost']:.0f}€")

    print("\n=== Résumé Analyse des FP — Autoencoder ===")
    print(f"  Seuil actuel : {ae_base['threshold']:.4f} | FP={ae_base['fp_count']} | Coût={ae_base['total_cost']:.0f}€")
    print(f"  Seuil coût-optimal     : {ae_res['cost_optimal_threshold']:.4f} | FP={ae_res['cost_optimal_fp']} | Coût={ae_res['cost_optimal_cost']:.0f}€")
    print(f"  Seuil precision≥0.5    : {ae_res['precision_constrained_threshold']:.4f} | FP={ae_res['precision_constrained_fp']} | Coût={ae_res['precision_constrained_cost']:.0f}€")


def main() -> None:
    print("=== Phase 7.2 — Analyse des Faux Positifs par Coût ===")
    d = load_data()

    X_test   = d["X_test"]
    y_test   = d["y_test"]
    if_scores = d["if_scores"]
    ae_scores = d["ae_scores"]
    if_thresh = d["if_thresh"]
    ae_thresh = d["ae_thresh"]

    # Baseline metrics at current thresholds
    if_base = compute_costs_at_threshold(if_scores, y_test, if_thresh)
    ae_base = compute_costs_at_threshold(ae_scores, y_test, ae_thresh)

    # Optimal threshold search
    print("Searching optimal thresholds for Isolation Forest...")
    if_res = find_optimal_thresholds(if_scores, y_test, "Isolation Forest")
    print("Searching optimal thresholds for Autoencoder...")
    ae_res = find_optimal_thresholds(ae_scores, y_test, "Autoencoder")

    # Cartography
    cartography_plot(X_test, y_test, if_scores, ae_scores, if_thresh, ae_thresh)

    # Threshold sweep plots
    plot_threshold_sweep(if_res, ae_res)

    # Summary
    print_and_save_summary(if_res, ae_res, if_base, ae_base)

    print("\nPhase 7.2 completed successfully.")


if __name__ == "__main__":
    main()
