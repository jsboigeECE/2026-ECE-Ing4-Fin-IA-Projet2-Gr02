"""Isolation Forest baseline for fraud detection (Phase 2.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestArtifacts:
    model_path: str
    predictions_path: str


def train_isolation_forest(
    data_path: str = "results/preprocessed_data.npz",
    output_dir: str = "results",
    random_state: int = 42,
    n_estimators: int = 200,
) -> IsolationForestArtifacts:
    """Train Isolation Forest on normal train transactions and save artifacts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data not found: {data_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    # Learn "normal" behavior from non-fraud transactions only.
    X_train_normal = X_train[y_train == 0]

    contamination = max(1e-4, float(np.mean(y_train)))

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_normal)

    # sklearn: -1 = anomaly, 1 = normal
    y_pred_test = model.predict(X_test)
    anomaly_score_test = -model.score_samples(X_test)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    predictions_path = os.path.join(output_dir, "isolation_forest_test_outputs.npz")

    joblib.dump(model, model_path)
    np.savez_compressed(
        predictions_path,
        y_pred_test=y_pred_test,
        anomaly_score_test=anomaly_score_test,
    )

    return IsolationForestArtifacts(
        model_path=model_path,
        predictions_path=predictions_path,
    )


if __name__ == "__main__":
    artifacts = train_isolation_forest()
    print("Isolation Forest trained successfully.")
    print(f"Model saved to: {artifacts.model_path}")
    print(f"Predictions saved to: {artifacts.predictions_path}")
