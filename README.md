# Production AI/ML Platform

A fully integrated, production-grade machine learning platform covering the entire lifecycle: data ingestion → feature engineering → distributed training → model serving → monitoring → automated retraining.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  INGEST   Kafka · Batch ETL · REST APIs · Unstructured  │
├─────────────────────────────────────────────────────────┤
│  STORE    Data Lake · Feature Store · Vector DB          │
├─────────────────────────────────────────────────────────┤
│  TRAIN    Distributed PyTorch · MLflow · HPO · Registry  │
├─────────────────────────────────────────────────────────┤
│  SERVE    Online API · RAG Pipeline · Batch Inference    │
├─────────────────────────────────────────────────────────┤
│  OBSERVE  Drift · Latency · RLHF · Auto-retrain loop    │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
aiml_project/
├── configs/               # YAML configs for all environments
├── data/
│   ├── raw/               # Landing zone for ingested data
│   ├── processed/         # Cleaned, validated datasets
│   └── features/          # Materialised feature sets
├── src/
│   ├── ingestion/         # Streaming, batch, API, unstructured ingestors
│   ├── features/          # Feature engineering & feature store client
│   ├── training/          # Model definitions, trainer, HPO, registry
│   ├── serving/           # FastAPI server, RAG pipeline, batch inference
│   ├── monitoring/        # Drift detection, metrics, feedback loop
│   └── utils/             # Logging, config loader, schema validation
├── notebooks/             # Exploratory analysis & prototyping
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/               # CLI entry points (train, evaluate, deploy)
├── docs/                  # Architecture diagrams, runbooks
├── models/                # Serialised model artefacts (git-ignored)
└── logs/                  # Runtime logs (git-ignored)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp configs/config.example.yaml configs/config.yaml

# 3. Run ingestion
python scripts/run_ingestion.py --source batch --config configs/config.yaml

# 4. Build features
python scripts/build_features.py --config configs/config.yaml

# 5. Train a model
python scripts/train.py --experiment my_experiment --config configs/config.yaml

# 6. Serve the model
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# 7. Run monitoring
python scripts/run_monitoring.py --config configs/config.yaml
```

## Key Technologies

| Layer       | Tools                                      |
|-------------|--------------------------------------------|
| Ingestion   | Apache Kafka, Apache Spark, Requests       |
| Storage     | Parquet/Delta Lake, Feast, FAISS/Weaviate  |
| Training    | PyTorch, Hugging Face, MLflow, Optuna      |
| Serving     | FastAPI, LangChain, Ray Serve              |
| Monitoring  | Evidently, Prometheus, custom drift logic  |
| Infra       | Docker, Kubernetes, GitHub Actions         |

## Configuration

All runtime behaviour is controlled via `configs/config.yaml`. See `configs/config.example.yaml` for all options.

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## License

MIT
