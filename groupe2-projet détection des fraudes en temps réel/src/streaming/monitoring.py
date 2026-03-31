"""Phase 6.3 - Streaming monitoring and alerting utilities."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class MonitoringConfig:
    alert_min_ensemble_score: float = 1.15
    alert_latency_ms: float = 100.0
    alert_fraud_streak: int = 3


class StreamMonitor:
    """Real-time logging, dashboard snapshots and alerts for stream decisions."""

    def __init__(self, output_dir: str, config: MonitoringConfig | None = None) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.config = config or MonitoringConfig()

        self.log_csv = os.path.join(self.output_dir, "stream_decisions_log.csv")
        self.alerts_jsonl = os.path.join(self.output_dir, "alerts.jsonl")
        self.dashboard_json = os.path.join(self.output_dir, "dashboard_snapshot.json")

        self.total = 0
        self.flagged = 0
        self.high_latency = 0
        self.fraud_streak = 0
        self.max_latency = 0.0
        self.sum_latency = 0.0

        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "event_id",
                        "prediction",
                        "if_score",
                        "ae_score",
                        "ensemble_score",
                        "if_threshold",
                        "ae_threshold",
                        "volatility",
                        "drift",
                        "latency_ms",
                        "model_updated",
                    ]
                )

    def _append_alert(self, payload: dict[str, Any]) -> None:
        with open(self.alerts_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_decision(self, event_id: int, decision: dict[str, Any], latency_ms: float) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()

        self.total += 1
        pred = int(decision["prediction"])
        self.flagged += pred
        self.sum_latency += float(latency_ms)
        self.max_latency = max(self.max_latency, float(latency_ms))

        if latency_ms > self.config.alert_latency_ms:
            self.high_latency += 1

        if pred == 1:
            self.fraud_streak += 1
        else:
            self.fraud_streak = 0

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    now,
                    event_id,
                    pred,
                    decision["if_score"],
                    decision["ae_score"],
                    decision["ensemble_score"],
                    decision["if_threshold"],
                    decision["ae_threshold"],
                    decision["volatility"],
                    int(decision["drift"]),
                    float(latency_ms),
                    int(decision["model_updated"]),
                ]
            )

        alerts: list[dict[str, Any]] = []

        if pred == 1 and float(decision["ensemble_score"]) >= self.config.alert_min_ensemble_score:
            alerts.append(
                {
                    "type": "fraud_high_confidence",
                    "timestamp": now,
                    "event_id": event_id,
                    "ensemble_score": float(decision["ensemble_score"]),
                }
            )

        if latency_ms > self.config.alert_latency_ms:
            alerts.append(
                {
                    "type": "latency_violation",
                    "timestamp": now,
                    "event_id": event_id,
                    "latency_ms": float(latency_ms),
                }
            )

        if self.fraud_streak >= self.config.alert_fraud_streak:
            alerts.append(
                {
                    "type": "fraud_streak",
                    "timestamp": now,
                    "event_id": event_id,
                    "streak": int(self.fraud_streak),
                }
            )

        if bool(decision.get("drift", False)):
            alerts.append(
                {
                    "type": "concept_drift",
                    "timestamp": now,
                    "event_id": event_id,
                    "anomaly_rate": float(decision.get("anomaly_rate", 0.0)),
                }
            )

        for a in alerts:
            self._append_alert(a)

        return alerts

    def dashboard_snapshot(self) -> dict[str, Any]:
        avg_latency = self.sum_latency / self.total if self.total > 0 else 0.0
        flagged_rate = self.flagged / self.total if self.total > 0 else 0.0
        latency_violation_rate = self.high_latency / self.total if self.total > 0 else 0.0

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "throughput_total_events": int(self.total),
            "fraud_flagged": int(self.flagged),
            "fraud_flagged_rate": float(flagged_rate),
            "latency_avg_ms": float(avg_latency),
            "latency_max_ms": float(self.max_latency),
            "latency_violation_rate": float(latency_violation_rate),
            "current_fraud_streak": int(self.fraud_streak),
        }

        with open(self.dashboard_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload
