"""tests/unit/test_monitoring.py — Unit tests for drift detection and metrics."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.monitoring.metrics import MetricsCollector, LatencyWindow
from src.utils.schema import DriftReport


# ── LatencyWindow tests ───────────────────────────────────────────────────────

class TestLatencyWindow:
    def test_empty_window(self):
        w = LatencyWindow()
        assert w.p50 == 0.0
        assert w.mean == 0.0

    def test_percentiles(self):
        w = LatencyWindow(window_size=100)
        for i in range(1, 101):
            w.record(float(i))
        assert 49.0 <= w.p50 <= 51.0
        assert w.p95 >= 94.0
        assert w.p99 >= 98.0

    def test_rolling_eviction(self):
        w = LatencyWindow(window_size=5)
        for i in range(10):
            w.record(float(i))
        # Only last 5 values kept
        assert w.mean == pytest.approx((5 + 6 + 7 + 8 + 9) / 5, abs=1e-6)


# ── MetricsCollector tests ────────────────────────────────────────────────────

class TestMetricsCollector:
    def test_initial_snapshot(self):
        c = MetricsCollector()
        snap = c.snapshot()
        assert snap["total_requests"] == 0
        assert snap["error_rate"] == 0.0

    def test_request_recording(self):
        c = MetricsCollector()
        c.record_request(50.0)
        c.record_request(100.0)
        c.record_request(200.0, error=True)
        snap = c.snapshot()
        assert snap["total_requests"] == 3
        assert snap["total_errors"] == 1
        assert pytest.approx(snap["error_rate"], abs=1e-6) == 1 / 3

    def test_alert_latency(self):
        c = MetricsCollector()
        for _ in range(100):
            c.record_request(500.0)  # 500ms, well above threshold
        cfg = {"monitoring": {"performance": {"latency_p99_threshold_ms": 200, "error_rate_threshold": 0.5}}}
        alerts = c.check_alerts(cfg)
        assert any("latency" in a for a in alerts)

    def test_alert_error_rate(self):
        c = MetricsCollector()
        for _ in range(10):
            c.record_request(10.0, error=True)
        cfg = {"monitoring": {"performance": {"latency_p99_threshold_ms": 1000, "error_rate_threshold": 0.05}}}
        alerts = c.check_alerts(cfg)
        assert any("Error rate" in a for a in alerts)

    def test_no_alerts_clean_traffic(self):
        c = MetricsCollector()
        for _ in range(50):
            c.record_request(10.0, error=False)
        cfg = {"monitoring": {"performance": {"latency_p99_threshold_ms": 200, "error_rate_threshold": 0.05}}}
        assert c.check_alerts(cfg) == []


# ── DriftReport schema ────────────────────────────────────────────────────────

class TestDriftReport:
    def test_no_drift(self):
        r = DriftReport(
            timestamp=datetime.now(timezone.utc),
            dataset_drift=False,
            drift_score=0.05,
            drifted_columns=[],
            details={},
        )
        assert r.dataset_drift is False
        assert r.drifted_columns == []

    def test_drift_with_columns(self):
        r = DriftReport(
            timestamp=datetime.now(timezone.utc),
            dataset_drift=True,
            drift_score=0.42,
            drifted_columns=["age", "income", "score"],
            details={"psi_scores": {"age": 0.31, "income": 0.55}},
        )
        assert len(r.drifted_columns) == 3
        assert r.details["psi_scores"]["income"] == 0.55
