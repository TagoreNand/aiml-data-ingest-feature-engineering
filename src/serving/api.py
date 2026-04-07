"""src/serving/api.py — FastAPI serving endpoint with batching, health, and metrics."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from transformers import AutoTokenizer

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import PredictRequest, PredictResponse


# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction latency", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2])


# ── App state ─────────────────────────────────────────────────────────────────
_model: Any = None
_tokenizer: Any = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _config
    _config = load_config()
    logger.info("Loading model for serving...")

    try:
        from src.training.registry import ModelRegistry
        registry = ModelRegistry(config=_config)
        _model = registry.load_champion()
        _model.eval()
    except Exception as exc:
        logger.warning(f"Could not load from registry ({exc}). Using stub model.")
        _model = None

    model_name = _config.get("training", {}).get("model_name", "bert-base-uncased")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Model and tokenizer ready.")
    yield
    logger.info("Shutting down serving layer.")


app = FastAPI(
    title="AI/ML Platform API",
    version="1.0.0",
    description="Online inference, RAG, and health endpoints.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Inference ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    t0 = time.perf_counter()
    try:
        predictions = _run_inference(request.inputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        REQUEST_COUNT.labels(status="ok").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)
        return PredictResponse(
            predictions=predictions,
            model_version=request.model_version,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _run_inference(texts: list[str]) -> list[Any]:
    if _model is None or _tokenizer is None:
        # Stub: return random labels
        import random
        return [random.randint(0, 1) for _ in texts]

    cfg = _config.get("training", {})
    max_length = cfg.get("max_length", 128)
    device = next(_model.parameters()).device

    encodings = _tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        out = _model(**encodings)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        preds = logits.argmax(-1).cpu().tolist()
    return preds


# ── Feedback collection ───────────────────────────────────────────────────────
@app.post("/feedback", tags=["feedback"])
async def feedback(request: Request):
    body = await request.json()
    from src.utils.schema import FeedbackRecord
    from datetime import datetime, timezone
    from pathlib import Path


    record = FeedbackRecord(
        prediction_id=body.get("prediction_id", "unknown"),
        true_label=body.get("true_label"),
        predicted_label=body.get("predicted_label"),
        timestamp=datetime.now(timezone.utc),
        source=body.get("source", "api"),
    )
    feedback_path = Path(_config.get("monitoring", {}).get("feedback", {}).get("collection_path", "data/raw/feedback"))
    feedback_path.mkdir(parents=True, exist_ok=True)
    out = feedback_path / "feedback.jsonl"
    with out.open("a") as fh:
        fh.write(record.model_dump_json() + "\n")
    return {"status": "recorded"}
