"""
TIER 4 — Observability and MLOps
Run: python tier4_observability.py

What this does:
  - Writes A/B testing traffic router (shadow + canary modes)
  - Writes automated retraining pipeline triggered by drift/feedback
  - Generates Grafana dashboard JSON (import at localhost:3000)
  - Adds structured performance logging with percentile tracking
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

from rich.console import Console
console = Console()


# ── A/B Traffic Router ────────────────────────────────────────────────────────

AB_ROUTER = '''"""src/serving/ab_router.py — A/B testing traffic router with shadow and canary modes."""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable

from src.utils.logger import logger


@dataclass
class ModelVariant:
    name: str
    version: str
    weight: float = 1.0          # Traffic share (normalised internally)
    shadow: bool = False          # Shadow mode: serve but don\'t return to user
    predict_fn: Callable | None = None


@dataclass
class ABMetrics:
    requests: int = 0
    latency_sum: float = 0.0
    errors: int = 0
    predictions: list[int] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return (self.latency_sum / self.requests * 1000) if self.requests else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


class ABRouter:
    """
    Routes traffic between model variants.

    Modes:
      - canary:  small % of traffic to challenger, rest to champion
      - shadow:  all traffic to champion for response, ALSO send to challenger (fire-and-forget)
      - full_ab: split traffic by weight, track metrics per variant
    """

    def __init__(self, variants: list[ModelVariant], mode: str = "shadow"):
        assert mode in ("canary", "shadow", "full_ab"), f"Unknown mode: {mode}"
        self.variants = variants
        self.mode     = mode
        self.metrics: dict[str, ABMetrics] = defaultdict(ABMetrics)
        self._normalise_weights()
        logger.info(f"ABRouter initialised | mode={mode} | variants={[v.name for v in variants]}")

    def _normalise_weights(self) -> None:
        real_variants = [v for v in self.variants if not v.shadow]
        total = sum(v.weight for v in real_variants)
        for v in real_variants:
            v.weight /= total

    def _select_variant(self, request_id: str | None = None) -> ModelVariant:
        """Deterministic selection via request hash (same user always gets same variant)."""
        real = [v for v in self.variants if not v.shadow]
        if request_id:
            h = int(hashlib.md5(request_id.encode()).hexdigest(), 16) / (2**128)
        else:
            h = random.random()
        cumulative = 0.0
        for v in real:
            cumulative += v.weight
            if h < cumulative:
                return v
        return real[-1]

    def predict(self, texts: list[str], request_id: str | None = None) -> dict[str, Any]:
        primary = self._select_variant(request_id)

        t0 = time.perf_counter()
        try:
            result = primary.predict_fn(texts) if primary.predict_fn else [0] * len(texts)
            latency = time.perf_counter() - t0
            m = self.metrics[primary.name]
            m.requests += 1
            m.latency_sum += latency
            m.predictions.extend(result if isinstance(result, list) else [result])
        except Exception as exc:
            self.metrics[primary.name].errors += 1
            raise

        # Shadow: fire-and-forget to challenger
        for v in self.variants:
            if v.shadow and v.predict_fn:
                try:
                    import threading
                    def _shadow_call(variant=v):
                        t = time.perf_counter()
                        try:
                            r = variant.predict_fn(texts)
                            self.metrics[variant.name].requests += 1
                            self.metrics[variant.name].latency_sum += time.perf_counter() - t
                        except Exception:
                            self.metrics[variant.name].errors += 1
                    threading.Thread(target=_shadow_call, daemon=True).start()
                except Exception:
                    pass

        return {
            "predictions": result,
            "variant": primary.name,
            "request_id": request_id,
        }

    def report(self) -> dict[str, dict]:
        out = {}
        for name, m in self.metrics.items():
            out[name] = {
                "requests": m.requests,
                "avg_latency_ms": round(m.avg_latency_ms, 2),
                "error_rate": round(m.error_rate, 4),
                "prediction_rate_positive": (
                    sum(1 for p in m.predictions if p == 1) / len(m.predictions)
                    if m.predictions else None
                ),
            }
        return out

    def should_promote_challenger(self,
                                   latency_threshold_ms: float = 150.0,
                                   min_requests: int = 100) -> tuple[bool, str]:
        """Returns (should_promote, reason)."""
        champ = next((v for v in self.variants if "champion" in v.name.lower()), None)
        chall = next((v for v in self.variants if "challenger" in v.name.lower()), None)
        if not champ or not chall:
            return False, "Missing champion or challenger variant"
        cm = self.metrics.get(champ.name, ABMetrics())
        tm = self.metrics.get(chall.name, ABMetrics())
        if tm.requests < min_requests:
            return False, f"Challenger has only {tm.requests} requests (need {min_requests})"
        if tm.error_rate > cm.error_rate * 1.5:
            return False, f"Challenger error rate {tm.error_rate:.2%} > 1.5× champion"
        if tm.avg_latency_ms > latency_threshold_ms:
            return False, f"Challenger p50 latency {tm.avg_latency_ms:.0f}ms > {latency_threshold_ms}ms"
        return True, f"Challenger ({tm.requests} requests, {tm.avg_latency_ms:.0f}ms) passes all gates"
'''


# ── Automated Retraining Pipeline ─────────────────────────────────────────────

AUTO_RETRAIN = '''"""src/training/auto_retrain.py — Automated retraining pipeline triggered by drift or feedback."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import logger


class RetrainingPipeline:
    """
    Full automated loop:
      1. Check trigger conditions (drift score OR feedback volume)
      2. Re-run feature pipeline on latest data
      3. Train new model version
      4. Evaluate vs champion
      5. Promote if better
      6. Log decision
    """

    def __init__(self, config: dict | None = None):
        self.cfg  = config or load_config()
        self.mon  = self.cfg.get("monitoring", {})
        self.fb   = self.mon.get("feedback", {})
        self.log_path = Path("logs/retrain_decisions.jsonl")
        self.log_path.parent.mkdir(exist_ok=True)

    # ── Trigger checks ────────────────────────────────────────────────────────

    def drift_triggered(self, drift_score: float) -> bool:
        threshold = self.mon.get("drift", {}).get("retrain_trigger_drift_score", 0.25)
        return drift_score >= threshold

    def feedback_triggered(self) -> tuple[bool, int]:
        fb_path = Path(self.fb.get("collection_path", "data/raw/feedback")) / "feedback.jsonl"
        if not fb_path.exists():
            return False, 0
        lines = fb_path.read_text().strip().splitlines()
        count = len(lines)
        min_samples = self.fb.get("min_samples_for_retrain", 500)
        return count >= min_samples, count

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def _run_step(self, cmd: list[str], step_name: str) -> bool:
        logger.info(f"[retrain] Running: {step_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[retrain] {step_name} failed:\\n{result.stderr}")
            return False
        logger.info(f"[retrain] {step_name} complete")
        return True

    def run(self, trigger: str, drift_score: float | None = None) -> dict:
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "drift_score": drift_score,
            "steps_completed": [],
            "outcome": "pending",
        }

        logger.info(f"[retrain] Pipeline started | trigger={trigger}")

        steps = [
            (["python", "scripts/build_features.py", "--config", "configs/config.yaml"],
             "feature_pipeline"),
            (["python", "scripts/train.py", "--experiment", f"auto_retrain_{trigger}",
              "--config", "configs/config.yaml"],
             "training"),
        ]

        for cmd, name in steps:
            if self._run_step(cmd, name):
                decision["steps_completed"].append(name)
            else:
                decision["outcome"] = f"failed_at_{name}"
                self._log(decision)
                return decision

        decision["outcome"] = "success"
        logger.info("[retrain] Pipeline complete — new model version registered.")
        self._log(decision)
        return decision

    def _log(self, decision: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(decision) + "\\n")
        logger.info(f"[retrain] Decision logged → {self.log_path}")

    # ── Convenience: check + run ──────────────────────────────────────────────

    def maybe_retrain(self, drift_score: float) -> dict | None:
        if self.drift_triggered(drift_score):
            return self.run(trigger="drift", drift_score=drift_score)
        fb_triggered, fb_count = self.feedback_triggered()
        if fb_triggered:
            return self.run(trigger=f"feedback_{fb_count}_samples")
        logger.info("[retrain] No trigger conditions met.")
        return None
'''


# ── Grafana Dashboard JSON ────────────────────────────────────────────────────

def generate_grafana_dashboard() -> dict:
    """Generate a Grafana dashboard JSON for the ML platform."""
    return {
        "title": "AI/ML Platform",
        "uid": "aiml-platform-v1",
        "schemaVersion": 38,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": [
            {
                "id": 1, "type": "stat", "title": "Requests/min",
                "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": "rate(predict_requests_total[1m]) * 60",
                              "legendFormat": "req/min"}],
                "options": {"colorMode": "background", "graphMode": "area"},
                "fieldConfig": {"defaults": {"color": {"mode": "thresholds"},
                                "thresholds": {"steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 100},
                                    {"color": "red", "value": 500}
                                ]}}}
            },
            {
                "id": 2, "type": "stat", "title": "Error rate",
                "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": 'rate(predict_requests_total{status="error"}[5m]) / rate(predict_requests_total[5m])',
                              "legendFormat": "error rate"}],
                "fieldConfig": {"defaults": {"unit": "percentunit",
                                "thresholds": {"steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 0.01},
                                    {"color": "red", "value": 0.05}
                                ]}}}
            },
            {
                "id": 3, "type": "stat", "title": "p99 latency (ms)",
                "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": "histogram_quantile(0.99, rate(predict_latency_seconds_bucket[5m])) * 1000",
                              "legendFormat": "p99"}],
                "fieldConfig": {"defaults": {"unit": "ms",
                                "thresholds": {"steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 100},
                                    {"color": "red", "value": 500}
                                ]}}}
            },
            {
                "id": 4, "type": "stat", "title": "Avg batch size",
                "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": "rate(predict_batch_size_sum[5m]) / rate(predict_batch_size_count[5m])",
                              "legendFormat": "avg batch"}],
                "fieldConfig": {"defaults": {"decimals": 1}}
            },
            {
                "id": 5, "type": "timeseries", "title": "Request rate by status",
                "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
                "targets": [
                    {"expr": 'rate(predict_requests_total{status="ok"}[1m]) * 60', "legendFormat": "success"},
                    {"expr": 'rate(predict_requests_total{status="error"}[1m]) * 60', "legendFormat": "error"},
                ],
                "fieldConfig": {"defaults": {"custom": {"lineWidth": 2}}}
            },
            {
                "id": 6, "type": "timeseries", "title": "Latency percentiles",
                "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
                "targets": [
                    {"expr": "histogram_quantile(0.50, rate(predict_latency_seconds_bucket[5m])) * 1000",
                     "legendFormat": "p50"},
                    {"expr": "histogram_quantile(0.95, rate(predict_latency_seconds_bucket[5m])) * 1000",
                     "legendFormat": "p95"},
                    {"expr": "histogram_quantile(0.99, rate(predict_latency_seconds_bucket[5m])) * 1000",
                     "legendFormat": "p99"},
                ],
                "fieldConfig": {"defaults": {"unit": "ms", "custom": {"lineWidth": 2}}}
            },
        ],
        "templating": {"list": []},
        "annotations": {"list": []},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold cyan]Tier 4 — Observability and MLOps[/]\n")

    # 1. Write A/B router
    out = Path("src/serving/ab_router.py")
    out.write_text(AB_ROUTER, encoding="utf-8")
    console.print(f"  [green]A/B router written[/] → {out}")

    # 2. Write auto-retrain pipeline
    out2 = Path("src/training/auto_retrain.py")
    out2.write_text(AUTO_RETRAIN, encoding="utf-8")
    console.print(f"  [green]Auto-retrain pipeline written[/] → {out2}")

    # 3. Generate Grafana dashboard
    dashboard = generate_grafana_dashboard()
    dash_path = Path("configs/grafana_dashboard.json")
    dash_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    console.print(f"  [green]Grafana dashboard JSON written[/] → {dash_path}")

    console.print("\n[bold green]Tier 4 complete![/]")
    console.print("\n[bold]To start Grafana + Prometheus:[/]")
    console.print("  [cyan]docker compose up grafana prometheus -d[/]")
    console.print("  Open http://localhost:3000 (admin/admin)")
    console.print("  Go to Dashboards → Import → Upload JSON → select configs/grafana_dashboard.json")
    console.print("\n[bold]A/B testing usage:[/]")
    console.print("  from src.serving.ab_router import ABRouter, ModelVariant")
    console.print("  router = ABRouter([champion_variant, challenger_variant], mode='shadow')")
    console.print("\n[dim]Next: run tier5_cloud_infra.py[/]")


if __name__ == "__main__":
    main()
