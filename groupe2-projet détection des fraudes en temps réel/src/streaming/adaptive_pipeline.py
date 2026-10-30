"""Phase 6.2 - Adaptive decision pipeline for streaming fraud detection.

This module provides:
- Adaptive thresholds based on short-term volatility
- Lightweight concept drift detection on anomaly-rate shift
- Incremental model update (autoencoder fine-tuning on confident normal samples)
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import tensorflow as tf


@dataclass
class AdaptivePipelineConfig:
    if_base_threshold: float
    ae_base_threshold: float
    volatility_window: int = 200
    drift_window: int = 400
    update_batch_size: int = 128
    min_confidence_margin: float = 0.20
    # Strict latency mode: disables heavy online updates by default.
    strict_latency_mode: bool = True
    enable_incremental_updates: bool = False


class AdaptiveDecisionPipeline:
    """Adaptive ensemble of Isolation Forest + Autoencoder for streaming.

    Decision rule:
    - Score each transaction with IF and AE
    - Adapt thresholds using rolling volatility
    - Combine model votes with weighted score
    - Track drift and run incremental AE updates on confident-normal samples
    """

    def __init__(self, project_root: str, config: AdaptivePipelineConfig | None = None) -> None:
        self.project_root = project_root
        self.results_dir = os.path.join(project_root, "results")

        if config is None:
            config = self._load_default_config()
        self.config = config

        self.if_model = joblib.load(os.path.join(self.results_dir, "isolation_forest_model.pkl"))
        self.ae_model = tf.keras.models.load_model(
            os.path.join(self.results_dir, "autoencoder_model.keras"),
            compile=False,
        )
        # Required for train_on_batch during incremental updates.
        self.ae_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="mse",
        )

        self.if_scores_hist: deque[float] = deque(maxlen=config.volatility_window)
        self.ae_scores_hist: deque[float] = deque(maxlen=config.volatility_window)
        self.decisions_hist: deque[int] = deque(maxlen=config.drift_window)
        self.recent_normals: deque[np.ndarray] = deque(maxlen=4 * config.update_batch_size)

        self.baseline_anomaly_rate = 0.02
        self.last_if_threshold = config.if_base_threshold
        self.last_ae_threshold = config.ae_base_threshold
        self.event_count = 0

    def _load_default_config(self) -> AdaptivePipelineConfig:
        if_best_path = os.path.join(self.results_dir, "isolation_forest_best_threshold.json")
        ae_best_path = os.path.join(self.results_dir, "autoencoder_detection_metrics.json")

        with open(if_best_path, "r", encoding="utf-8") as f:
            if_info = json.load(f)
        with open(ae_best_path, "r", encoding="utf-8") as f:
            ae_info = json.load(f)

        return AdaptivePipelineConfig(
            if_base_threshold=float(if_info["best_threshold"]),
            ae_base_threshold=float(ae_info["threshold"]),
        )

    def _compute_scores(self, x: np.ndarray) -> tuple[float, float]:
        x2d = x.reshape(1, -1)
        if_score = float(-self.if_model.score_samples(x2d)[0])
        # Direct forward call is faster than model.predict for single-sample inference.
        recon = self.ae_model(tf.convert_to_tensor(x2d, dtype=tf.float32), training=False).numpy()
        ae_score = float(np.mean(np.square(x2d - recon), axis=1)[0])
        return if_score, ae_score

    def warmup(self, feature_dim: int) -> None:
        """Warm up model paths to avoid first-event latency spikes."""
        dummy = np.zeros((feature_dim,), dtype=np.float32)
        for _ in range(3):
            _ = self._compute_scores(dummy)

    def _estimate_volatility(self) -> float:
        if len(self.if_scores_hist) < 10 or len(self.ae_scores_hist) < 10:
            return 0.0
        if_arr = np.asarray(self.if_scores_hist, dtype=float)
        ae_arr = np.asarray(self.ae_scores_hist, dtype=float)

        if_vol = float(np.std(if_arr) / (np.mean(if_arr) + 1e-8))
        ae_vol = float(np.std(ae_arr) / (np.mean(ae_arr) + 1e-8))
        return 0.5 * (if_vol + ae_vol)

    def _adaptive_thresholds(self) -> tuple[float, float, float]:
        vol = self._estimate_volatility()
        # More volatility -> stricter thresholds to reduce burst false positives.
        scale = 1.0 + min(max(vol, 0.0), 1.5) * 0.35
        if_t = self.config.if_base_threshold * scale
        ae_t = self.config.ae_base_threshold * scale

        self.last_if_threshold = float(if_t)
        self.last_ae_threshold = float(ae_t)
        return float(if_t), float(ae_t), float(vol)

    def _detect_drift(self) -> dict[str, Any]:
        if len(self.decisions_hist) < max(50, self.config.drift_window // 4):
            return {"drift": False, "anomaly_rate": float(np.mean(self.decisions_hist)) if self.decisions_hist else 0.0}

        current_rate = float(np.mean(self.decisions_hist))
        delta = current_rate - self.baseline_anomaly_rate
        drift = bool(delta > 0.08)

        # EWMA baseline update
        self.baseline_anomaly_rate = 0.98 * self.baseline_anomaly_rate + 0.02 * current_rate
        return {
            "drift": drift,
            "anomaly_rate": current_rate,
            "baseline_rate": float(self.baseline_anomaly_rate),
            "delta": float(delta),
        }

    def _incremental_update(self) -> bool:
        if self.config.strict_latency_mode or not self.config.enable_incremental_updates:
            return False
        if len(self.recent_normals) < self.config.update_batch_size:
            return False

        batch = np.asarray(list(self.recent_normals)[-self.config.update_batch_size :], dtype=np.float32)
        self.ae_model.train_on_batch(batch, batch)
        return True

    def decide(self, x: np.ndarray) -> dict[str, Any]:
        self.event_count += 1
        if_score, ae_score = self._compute_scores(x)

        self.if_scores_hist.append(if_score)
        self.ae_scores_hist.append(ae_score)
        if_t, ae_t, volatility = self._adaptive_thresholds()

        if_flag = int(if_score >= if_t)
        ae_flag = int(ae_score >= ae_t)

        # Weighted ensemble score in [0, 1]
        if_norm = float(min(1.0, if_score / (if_t + 1e-8)))
        ae_norm = float(min(1.0, ae_score / (ae_t + 1e-8)))
        ensemble_score = 0.55 * if_norm + 0.45 * ae_norm
        final_pred = int((if_flag + ae_flag) >= 1 or ensemble_score >= 1.0)

        self.decisions_hist.append(final_pred)

        # Confident normal sample for incremental adaptation.
        margin_if = (if_t - if_score) / (if_t + 1e-8)
        margin_ae = (ae_t - ae_score) / (ae_t + 1e-8)
        if final_pred == 0 and margin_if > self.config.min_confidence_margin and margin_ae > self.config.min_confidence_margin:
            self.recent_normals.append(np.asarray(x, dtype=np.float32))

        updated = self._incremental_update()
        drift_info = self._detect_drift()

        # Drift response: temporarily tighten thresholds.
        if drift_info.get("drift", False):
            self.last_if_threshold *= 1.05
            self.last_ae_threshold *= 1.05

        return {
            "if_score": float(if_score),
            "ae_score": float(ae_score),
            "if_threshold": float(self.last_if_threshold),
            "ae_threshold": float(self.last_ae_threshold),
            "if_flag": int(if_flag),
            "ae_flag": int(ae_flag),
            "ensemble_score": float(ensemble_score),
            "prediction": int(final_pred),
            "volatility": float(volatility),
            "drift": bool(drift_info.get("drift", False)),
            "anomaly_rate": float(drift_info.get("anomaly_rate", 0.0)),
            "model_updated": bool(updated),
        }
