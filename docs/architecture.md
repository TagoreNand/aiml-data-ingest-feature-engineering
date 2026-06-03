# Architecture & Runbooks

## System Overview

The platform is structured as five integrated layers connected by a feedback loop. Each layer is independently deployable and observable.

```
Ingest → Store → Train → Serve → Observe → (retrain trigger) → Train
```

## Component Descriptions

### Ingestion Layer
- **BatchIngestor**: Reads Parquet/CSV files from a local or cloud path on a schedule.
- **KafkaIngestor**: Consumes from a Kafka topic for real-time event data.
- **APIIngestor**: Pages through a REST API with rate limiting and retry logic.

All ingestors write `RawRecord` objects as newline-delimited JSON to `data/raw/`.

### Feature Layer
- **FeatureTransformer**: Stateful fit/transform with standard scaling, label encoding, and custom transforms.
- **FeaturePipeline**: Loads raw data → validates → engineers features → splits → materialises to Parquet in `data/features/`.

### Training Layer
- **Trainer**: PyTorch training loop with MLflow tracking, early stopping, and checkpoint saving.
- **HPORunner**: Optuna study over learning rate, batch size, epochs, warmup, and weight decay.
- **ModelRegistry**: Wraps MLflow Model Registry for version promotion via champion/challenger aliases.

### Serving Layer
- **FastAPI app** (`src/serving/api.py`): `/predict`, `/feedback`, `/health`, `/metrics` endpoints.
- **RAGPipeline**: FAISS-backed retrieval + OpenAI LLM generation.

### Monitoring Layer
- **DriftDetector**: Evidently-based dataset drift report with PSI fallback.
- **RetrainingTrigger**: Fires when drift score or feedback volume exceeds threshold.
- **MetricsCollector**: Rolling p50/p95/p99 latency, error rate, throughput.

---

## Runbooks

### Deploy a new model to production
1. Train: `python scripts/train.py --experiment my_exp`
2. Evaluate: `python scripts/evaluate.py --version <version>`
3. If metrics pass, the registry promotes automatically. If not, promote manually:
   ```
   python -c "
   from src.training.registry import ModelRegistry
   from src.utils.schema import ModelMetrics
   r = ModelRegistry()
   r.try_promote('<version>', ModelMetrics(f1=0.92))
   "
   ```

### Rollback to previous champion
```python
from src.training.registry import ModelRegistry
r = ModelRegistry()
r.client.set_registered_model_alias(r.model_name, "champion", "<previous_version>")
```

### Force a retrain
```python
from src.monitoring.drift import RetrainingTrigger
RetrainingTrigger().trigger_retrain("manual trigger")
```

### Add documents to the RAG index
```python
from src.serving.rag import RAGPipeline
rag = RAGPipeline()
rag.add_documents(["doc text 1", "doc text 2"], metadata=[{"source": "wiki"}, {"source": "wiki"}])
```

### Scale the serving API
```bash
uvicorn src.serving.api:app --workers 8 --host 0.0.0.0 --port 8000
```

Or via Docker Compose, set `UVICORN_WORKERS=8` in the `api` service environment.

---

## Alerting Thresholds (defaults)

| Metric | Threshold |
|--------|-----------|
| p99 latency | 200 ms |
| Error rate | 1% |
| PSI drift score | 0.20 per column |
| Dataset drift share | 25% of columns |
| Feedback samples for retrain | 500 |

All configurable in `configs/config.yaml` under `monitoring`.
