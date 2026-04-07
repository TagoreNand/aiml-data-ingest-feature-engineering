"""src/monitoring/metrics.py — Latency, error rate, and custom business metric tracking."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from collections import deque
from threading import Lock
from typing import Deque


@dataclass
class LatencyWindow:
    """Rolling window of latency measurements (milliseconds)."""

    window_size: int = 1000
    _values: Deque[float] = field(init=False)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=self.window_size)

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._values.append(latency_ms)

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self._values:
                return 0.0
                
            sorted_vals = sorted(self._values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def p50(self) -> float: return self.percentile(50)
    @property
    def p95(self) -> float: return self.percentile(95)
    @property
    def p99(self) -> float: return self.percentile(99)
    @property
    def mean(self) -> float:
        with self._lock:
            return sum(self._values) / len(self._values) if self._values else 0.0


class MetricsCollector:
    """Thread-safe in-process metrics collector."""

    def __init__(self) -> None:
        self._requests: int = 0
        self._errors: int = 0
        self._latency = LatencyWindow()
        self._lock = Lock()
        self._start_time = time.time()

    def record_request(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self._requests += 1
            if error:
                self._errors += 1
        self._latency.record(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            requests = self._requests
            errors = self._errors
        uptime = time.time() - self._start_time
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(uptime, 1),
            "total_requests": requests,
            "total_errors": errors,
            "error_rate": errors / max(requests, 1),
            "throughput_rps": round(requests / max(uptime, 1), 2),
            "latency": {
                "mean_ms": round(self._latency.mean, 2),
                "p50_ms": round(self._latency.p50, 2),
                "p95_ms": round(self._latency.p95, 2),
                "p99_ms": round(self._latency.p99, 2),
            },
        }

    def check_alerts(self, config: dict) -> list[str]:
        alerts = []
        mon = config.get("monitoring", {}).get("performance", {})
        threshold_latency = mon.get("latency_p99_threshold_ms", 200)
        threshold_error = mon.get("error_rate_threshold", 0.01)
        snap = self.snapshot()
        if snap["latency"]["p99_ms"] > threshold_latency:
            alerts.append(f"p99 latency {snap['latency']['p99_ms']}ms exceeds threshold {threshold_latency}ms")
        if snap["error_rate"] > threshold_error:
            alerts.append(f"Error rate {snap['error_rate']:.2%} exceeds threshold {threshold_error:.2%}")
        return alerts


# Singleton for in-process sharing
_COLLECTOR = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _COLLECTOR
