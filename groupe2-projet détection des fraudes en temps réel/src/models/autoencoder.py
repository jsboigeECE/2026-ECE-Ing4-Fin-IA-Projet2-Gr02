"""Autoencoder baseline for fraud detection (Phase 3.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass
class AutoencoderArtifacts:
    model_path: str
    history_path: str


def build_autoencoder(input_dim: int, learning_rate: float = 1e-3) -> Any:
    """Build and compile a dense autoencoder: input -> 32 -> 16 -> 32 -> output."""
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    x = tf.keras.layers.Dense(32, activation="relu", name="encoder_dense_32")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu", name="bottleneck_16")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="decoder_dense_32")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fraud_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def reconstruction_error(model: Any, X: np.ndarray) -> np.ndarray:
    """Return per-sample reconstruction MSE."""
    reconstructed = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def train_autoencoder(
    data_path: str = "results/preprocessed_data.npz",
    output_dir: str = "results",
    random_state: int = 42,
    epochs: int = 20,
    batch_size: int = 256,
    validation_split: float = 0.1,
) -> AutoencoderArtifacts:
    """Train autoencoder on normal train transactions only and save artifacts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data not found: {data_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]

    # Learn normal behavior from non-fraud transactions only.
    X_train_normal = X_train[y_train == 0]
    if len(X_train_normal) == 0:
        raise ValueError("No normal transactions found in training data (y_train == 0).")

    tf.keras.utils.set_random_seed(random_state)
    model = build_autoencoder(input_dim=X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_normal,
        X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "autoencoder_model.keras")
    history_path = os.path.join(output_dir, "autoencoder_history.npz")

    model.save(model_path)
    np.savez_compressed(
        history_path,
        loss=np.asarray(history.history.get("loss", []), dtype=float),
        val_loss=np.asarray(history.history.get("val_loss", []), dtype=float),
    )

    return AutoencoderArtifacts(model_path=model_path, history_path=history_path)


if __name__ == "__main__":
    artifacts = train_autoencoder()
    print("Autoencoder trained successfully.")
    print(f"Model saved to: {artifacts.model_path}")
    print(f"History saved to: {artifacts.history_path}")
