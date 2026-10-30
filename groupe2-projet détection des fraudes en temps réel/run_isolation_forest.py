"""Run Phase 2.1 Isolation Forest training."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.isolation_forest import train_isolation_forest


if __name__ == "__main__":
    artifacts = train_isolation_forest(
        data_path="results/preprocessed_data.npz",
        output_dir="results",
    )
    print("Phase 2.1 completed successfully.")
    print(f"Model: {artifacts.model_path}")
    print(f"Outputs: {artifacts.predictions_path}")
