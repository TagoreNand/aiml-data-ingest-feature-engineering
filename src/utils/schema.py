"""src/utils/schema.py — Pydantic schemas shared across layers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class RawRecord(BaseModel):
    id: str
    source: str
    timestamp: datetime
    payload: dict[str, Any]
    schema_version: str = "1.0"


# ── Features ──────────────────────────────────────────────────────────────────

class FeatureRow(BaseModel):
    entity_id: str
    feature_values: dict[str, float | int | str]
    event_timestamp: datetime


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    model_name: str
    task: str
    num_labels: int = 2
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    experiment_name: str = "default_experiment"


class ModelMetrics(BaseModel):
    accuracy: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    loss: float | None = None
    auc: float | None = None

    def passes_threshold(self, thresholds: dict[str, float]) -> bool:
        for metric, min_val in thresholds.items():
            actual = getattr(self, metric, None)
            if actual is None or actual < min_val:
                return False
        return True


# ── Serving ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1, max_length=64)
    model_version: str = "champion"


class PredictResponse(BaseModel):
    predictions: list[Any]
    model_version: str
    latency_ms: float


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float


# ── Monitoring ────────────────────────────────────────────────────────────────

class DriftReport(BaseModel):
    timestamp: datetime
    dataset_drift: bool
    drift_score: float
    drifted_columns: list[str]
    details: dict[str, Any]


class FeedbackRecord(BaseModel):
    prediction_id: str
    true_label: Any
    predicted_label: Any
    timestamp: datetime
    source: str = "human"
