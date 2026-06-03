"""src/serving/ab_router.py — A/B testing traffic router with shadow and canary modes."""
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
    shadow: bool = False          # Shadow mode: serve but don't return to user
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
        except Exception:
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
                            variant.predict_fn(texts)
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

