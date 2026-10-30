"""SMOTE handler for class imbalance (Phase 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "results", "preprocessed_data.npz")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "preprocessed_data_smote.npz")


@dataclass
class SmoteArtifacts:
    output_path: str
    n_train_before: int
    n_train_after: int
    fraud_before: int
    fraud_after: int


def _class_stats(y: np.ndarray) -> tuple[int, int, int]:
    """Return (total, normal_count, fraud_count)."""
    y = y.astype(int)
    total = int(len(y))
    fraud = int(np.sum(y == 1))
    normal = int(np.sum(y == 0))
    return total, normal, fraud


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float | str = "auto",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE on train split only."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn"
        ) from exc

    sampler: Any = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return np.asarray(X_resampled), np.asarray(y_resampled)


def rebalance_train_with_smote(
    data_path: str = DEFAULT_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    sampling_strategy: float | str = "auto",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> SmoteArtifacts:
    """Load preprocessed data, apply SMOTE to train set, and save new dataset."""
    data_path = os.path.abspath(data_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data not found: {data_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    n_before, _, fraud_before = _class_stats(y_train)

    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    n_after, _, fraud_after = _class_stats(y_train_smote)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=X_train_smote,
        X_test=X_test,
        y_train=y_train_smote,
        y_test=y_test,
    )

    return SmoteArtifacts(
        output_path=output_path,
        n_train_before=n_before,
        n_train_after=n_after,
        fraud_before=fraud_before,
        fraud_after=fraud_after,
    )


if __name__ == "__main__":
    artifacts = rebalance_train_with_smote()
    print("Phase 4.1 SMOTE completed.")
    print(f"Saved: {artifacts.output_path}")
    print(
        "Train size: "
        f"{artifacts.n_train_before} -> {artifacts.n_train_after} | "
        f"Fraud count: {artifacts.fraud_before} -> {artifacts.fraud_after}"
    )
