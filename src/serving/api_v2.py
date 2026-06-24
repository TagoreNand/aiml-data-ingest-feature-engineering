"""src/serving/api_v2.py — Production FastAPI with auth, batching, ONNX, versioning."""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from transformers import AutoTokenizer

from src.utils.config import load_config
from src.utils.logger import logger


# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("predict_requests_total", "Total prediction requests", ["status", "version"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction latency",
                             buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
BATCH_SIZE_HIST = Histogram("predict_batch_size", "Batch sizes", buckets=[1, 2, 4, 8, 16, 32, 64])


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1, max_length=64)
    model_version: str = "champion"
    return_probabilities: bool = False


class Prediction(BaseModel):
    label: int
    confidence: float
    probabilities: list[float] | None = None


class PredictResponse(BaseModel):
    request_id: str
    predictions: list[Prediction]
    model_version: str
    latency_ms: float


# ── API key auth ──────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production: load from env / secrets manager
VALID_API_KEYS = {"dev-key-12345", "prod-key-abcde"}

async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)):
    if api_key is None or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Async batching queue ──────────────────────────────────────────────────────
class BatchQueue:
    """Collects individual requests and processes them in micro-batches."""

    def __init__(self, max_batch: int = 32, timeout_ms: float = 10.0):
        self.max_batch = max_batch
        self.timeout   = timeout_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()

    async def submit(self, texts: list[str]) -> list[Prediction]:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((texts, future))
        return await future

    async def run(self, inference_fn) -> None:
        while True:
            batch_items = []
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.timeout)
                batch_items.append(item)
                while len(batch_items) < self.max_batch:
                    try:
                        item = self._queue.get_nowait()
                        batch_items.append(item)
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                await asyncio.sleep(0.001)
                continue

            if not batch_items:
                continue

            all_texts = []
            offsets   = [0]
            for texts, _ in batch_items:
                all_texts.extend(texts)
                offsets.append(offsets[-1] + len(texts))

            BATCH_SIZE_HIST.observe(len(all_texts))

            try:
                all_preds = inference_fn(all_texts)
                for i, (_, future) in enumerate(batch_items):
                    preds = all_preds[offsets[i]:offsets[i + 1]]
                    if not future.done():
                        future.set_result(preds)
            except Exception as exc:
                for _, future in batch_items:
                    if not future.done():
                        future.set_exception(exc)


# ── App state ─────────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_config    = {}
_batch_q   = BatchQueue(max_batch=32, timeout_ms=10)
_use_onnx  = False
_onnx_sess = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _config, _use_onnx, _onnx_sess
    _config = load_config()
    model_name = _config.get("training", {}).get("model_name", "bert-base-uncased")


    logger.info("Loading model for serving...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try ONNX first
    onnx_path = Path("models/model.onnx")
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            _onnx_sess = ort.InferenceSession(str(onnx_path),
                                               providers=["CPUExecutionProvider"])
            _use_onnx = True
            logger.info("ONNX Runtime session loaded.")
        except Exception as e:
            logger.warning(f"ONNX load failed ({e}), falling back to PyTorch.")

    # PyTorch fallback
    if not _use_onnx:
        try:
            from src.training.registry import ModelRegistry
            registry = ModelRegistry(config=_config)
            _model = registry.load_champion()
            _model.eval()
            logger.info("PyTorch champion model loaded.")
        except Exception as e:
            logger.warning(f"Registry load failed ({e}). Using stub.")
            _model = None

    logger.info(f"Serving backend: {'ONNX' if _use_onnx else 'PyTorch'}")

    # Start batching worker
    asyncio.create_task(_batch_q.run(_run_inference))
    logger.info("Batch queue worker started.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="AI/ML Platform API v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Middleware: request ID injection ──────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Inference function ────────────────────────────────────────────────────────
def _run_inference(texts: list[str]) -> list[Prediction]:
    cfg = _config.get("training", {})
    max_length = cfg.get("max_length", 128)

    enc = _tokenizer(texts, truncation=True, padding="max_length",
                     max_length=max_length, return_tensors="pt")

    if _use_onnx and _onnx_sess is not None:
        outputs = _onnx_sess.run(None, {
            "input_ids":      enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        })
        logits = torch.tensor(outputs[0])
    elif _model is not None:
        with torch.no_grad():
            out = _model(**enc)
            logits = out["logits"] if isinstance(out, dict) else out.logits
    else:
        import random
        return [Prediction(label=random.randint(0, 1), confidence=0.5) for _ in texts]

    probs  = torch.softmax(logits, dim=-1)
    labels = logits.argmax(-1).tolist()
    return [
        Prediction(
            label=labels[i],
            confidence=float(probs[i, labels[i]]),
            probabilities=probs[i].tolist(),
        )
        for i in range(len(texts))
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "ok",
        "backend": "onnx" if _use_onnx else "pytorch",
        "model_loaded": _model is not None or _use_onnx,
    }


@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
async def predict_v1(request: Request, body: PredictRequest,
                     api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    try:
        predictions = await _batch_q.submit(body.inputs)
        if not body.return_probabilities:
            for p in predictions:
                p.probabilities = None
        latency_ms = (time.perf_counter() - t0) * 1000
        REQUEST_COUNT.labels(status="ok", version="v1").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)
        return PredictResponse(
            request_id=request_id,
            predictions=predictions,
            model_version=body.model_version,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(status="error", version="v1").inc()
        logger.exception(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# Backwards-compatible /predict endpoint (no auth required)
@app.post("/predict", tags=["inference (legacy)"])
async def predict_legacy(body: PredictRequest):
    t0 = time.perf_counter()
    preds = _run_inference(body.inputs)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"predictions": [p.label for p in preds],
            "model_version": body.model_version,
            "latency_ms": round(latency_ms, 2)}


@app.post("/feedback", tags=["feedback"])
async def feedback(request: Request):
    body = await request.json()
    feedback_path = Path("data/raw/feedback")
    feedback_path.mkdir(parents=True, exist_ok=True)
    import json
    with (feedback_path / "feedback.jsonl").open("a") as f:
        f.write(json.dumps({**body, "ts": time.time()}) + "\n")
    return {"status": "recorded"}


