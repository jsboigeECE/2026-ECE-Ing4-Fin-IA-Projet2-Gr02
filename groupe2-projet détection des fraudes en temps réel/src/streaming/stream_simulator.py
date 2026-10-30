"""Phase 6.1 - Real-time transaction stream simulator.

Simulates transaction arrival and enforces low-latency scoring using the
AdaptiveDecisionPipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .adaptive_pipeline import AdaptiveDecisionPipeline
from .monitoring import StreamMonitor


@dataclass
class StreamConfig:
    n_events: int = 1500
    max_buffer_size: int = 500
    target_latency_ms: float = 100.0
    realtime: bool = False
    mean_interarrival_ms: float = 8.0


def _project_root_from_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TransactionStreamSimulator:
    """Event loop that simulates transaction arrivals and online decisions."""

    def __init__(self, project_root: str, config: StreamConfig | None = None) -> None:
        self.project_root = project_root
        self.results_dir = os.path.join(project_root, "results")
        self.streaming_dir = os.path.join(self.results_dir, "streaming")
        os.makedirs(self.streaming_dir, exist_ok=True)

        self.config = config or StreamConfig()
        self.pipeline = AdaptiveDecisionPipeline(project_root)
        self.monitor = StreamMonitor(self.streaming_dir)

        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.config.max_buffer_size)
        self.latencies_ms: list[float] = []

        data = np.load(os.path.join(self.results_dir, "preprocessed_data.npz"))
        self.X_test = data["X_test"].astype(np.float32)
        self.y_test = data["y_test"].astype(np.int64)

        # Warm-up reduces cold-start latency spikes and helps satisfy strict SLA.
        self.pipeline.warmup(self.X_test.shape[1])

    def _sleep_interarrival(self) -> None:
        if not self.config.realtime:
            return
        delay_s = max(0.0, np.random.exponential(self.config.mean_interarrival_ms / 1000.0))
        time.sleep(delay_s)

    def run(self) -> dict[str, Any]:
        n = min(self.config.n_events, len(self.X_test))
        target = self.config.target_latency_ms

        for i in range(n):
            self._sleep_interarrival()

            x = self.X_test[i]
            y_true = int(self.y_test[i])

            t0 = time.perf_counter()
            decision = self.pipeline.decide(x)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.latencies_ms.append(latency_ms)

            alerts = self.monitor.log_decision(i, decision, latency_ms)

            self.buffer.append(
                {
                    "event_id": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "y_true": y_true,
                    "prediction": int(decision["prediction"]),
                    "ensemble_score": float(decision["ensemble_score"]),
                    "latency_ms": float(latency_ms),
                    "alerts": alerts,
                }
            )

            if (i + 1) % 250 == 0:
                avg_l = float(np.mean(self.latencies_ms))
                print(
                    f"Processed {i+1}/{n} events | avg_latency={avg_l:.2f}ms | "
                    f"buffer={len(self.buffer)}"
                )

        dashboard = self.monitor.dashboard_snapshot()

        arr = np.asarray(self.latencies_ms, dtype=float)
        summary = {
            "events_processed": int(n),
            "target_latency_ms": float(target),
            "latency_avg_ms": float(np.mean(arr)) if len(arr) else 0.0,
            "latency_p95_ms": float(np.quantile(arr, 0.95)) if len(arr) else 0.0,
            "latency_max_ms": float(np.max(arr)) if len(arr) else 0.0,
            "latency_under_target_rate": float(np.mean(arr <= target)) if len(arr) else 1.0,
            "buffer_size_final": int(len(self.buffer)),
            "dashboard": dashboard,
        }

        out_path = os.path.join(self.streaming_dir, "stream_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\nStreaming simulation complete.")
        print(f"Summary saved to: {out_path}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud-stream simulation.")
    parser.add_argument("--n-events", type=int, default=1500)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=500)
    args = parser.parse_args()

    cfg = StreamConfig(
        n_events=args.n_events,
        realtime=bool(args.realtime),
        max_buffer_size=args.buffer_size,
    )

    simulator = TransactionStreamSimulator(_project_root_from_file(), cfg)
    simulator.run()


if __name__ == "__main__":
    main()
