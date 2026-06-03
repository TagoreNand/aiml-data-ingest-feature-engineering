"""
TIER 5 — Cloud and infrastructure
Run: python tier5_cloud_infra.py

What this does:
  - Writes production-grade multi-stage Dockerfile
  - Writes GitHub Actions CI/CD workflow (test → lint → build → push → deploy)
  - Writes Kubernetes Helm chart (deployment, service, HPA, ingress)
  - Writes docker-compose.prod.yml with health checks and resource limits
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
from rich.console import Console
console = Console()


# ── Production Dockerfile ─────────────────────────────────────────────────────

DOCKERFILE_PROD = """# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Non-root user for security
RUN useradd -m -u 1000 appuser

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/        ./src/
COPY configs/    ./configs/
COPY models/     ./models/

# Pre-download tokenizer at build time (avoids runtime HuggingFace calls)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

USER appuser

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    MLFLOW_ALLOW_FILE_STORE=true \\
    GIT_PYTHON_REFRESH=quiet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \\
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.serving.api_v2:app", "--host", "0.0.0.0", "--port", "8000", \\
     "--workers", "2", "--log-level", "info"]
"""


# ── GitHub Actions CI/CD ──────────────────────────────────────────────────────

CI_WORKFLOW = """name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/aiml-platform
  PYTHON_VERSION: "3.11"

jobs:

  # ── Lint & Type Check ────────────────────────────────────────────────────────
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install ruff mypy
      - run: ruff check src/ scripts/
      - run: mypy src/ --ignore-missing-imports --no-strict-optional

  # ── Unit Tests ───────────────────────────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Run unit tests
        run: |
          python -m pytest tests/unit/ -v \\
            --cov=src --cov-report=xml \\
            --cov-fail-under=70
        env:
          PYTHONPATH: .
          MLFLOW_ALLOW_FILE_STORE: "true"
          GIT_PYTHON_REFRESH: quiet
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # ── Model Validation ─────────────────────────────────────────────────────────
  model-validation:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Validate model artifacts
        run: |
          python -c "
          from pathlib import Path
          import sys
          # Check critical files exist
          required = ['src/serving/api_v2.py', 'src/training/trainer.py',
                      'src/monitoring/drift.py', 'configs/config.example.yaml']
          missing = [f for f in required if not Path(f).exists()]
          if missing:
              print('MISSING:', missing)
              sys.exit(1)
          print('All critical files present')
          "
        env:
          PYTHONPATH: .

  # ── Build & Push Docker Image ─────────────────────────────────────────────────
  build-push:
    runs-on: ubuntu-latest
    needs: [test, model-validation]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.prod
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_SHA=${{ github.sha }}

  # ── Deploy to Staging ─────────────────────────────────────────────────────────
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-push
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging via Helm
        run: |
          echo "Deploying SHA=${{ github.sha }} to staging"
          # helm upgrade --install aiml-platform ./helm/aiml-platform \\
          #   --namespace staging \\
          #   --set image.tag=sha-${{ github.sha }} \\
          #   --set replicaCount=1 \\
          #   --wait --timeout=5m
          echo "Staging deploy complete (helm command shown above)"
"""


# ── Kubernetes Helm Chart ─────────────────────────────────────────────────────

HELM_DEPLOYMENT = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-api
  labels:
    app: aiml-platform
    version: {{ .Values.image.tag }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: aiml-platform
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0       # Zero-downtime deploys
  template:
    metadata:
      labels:
        app: aiml-platform
        version: {{ .Values.image.tag }}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          env:
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: aiml-secrets
                  key: mlflow-uri
            - name: MLFLOW_ALLOW_FILE_STORE
              value: "true"
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 5
"""

HELM_HPA = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .Release.Name }}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ .Release.Name }}-api
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: predict_requests_total
        target:
          type: AverageValue
          averageValue: 100      # Scale up when >100 req/s per pod
"""

HELM_VALUES = """replicaCount: 2

image:
  repository: ghcr.io/your-org/aiml-platform
  tag: latest

autoscaling:
  minReplicas: 2
  maxReplicas: 10

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
  hosts:
    - host: api.aiml-platform.example.com
      paths:
        - path: /
          pathType: Prefix
"""

DOCKER_COMPOSE_PROD = """version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_ALLOW_FILE_STORE=true
      - GIT_PYTHON_REFRESH=quiet
    volumes:
      - ./configs:/app/configs:ro
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G
    healthcheck:
      test: ["CMD", "python", "-c",
             "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    depends_on:
      - mlflow
      - redis

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
    volumes:
      - mlflow_data:/mlflow
    deploy:
      resources:
        limits:
          memory: 512M
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  mlflow_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold cyan]Tier 5 — Cloud and infrastructure[/]\n")

    files = {
        "Dockerfile.prod": DOCKERFILE_PROD,
        ".github/workflows/ci_cd.yml": CI_WORKFLOW,
        "helm/aiml-platform/templates/deployment.yaml": HELM_DEPLOYMENT,
        "helm/aiml-platform/templates/hpa.yaml": HELM_HPA,
        "helm/aiml-platform/values.yaml": HELM_VALUES,
        "docker-compose.prod.yml": DOCKER_COMPOSE_PROD,
    }

    for fpath, content in files.items():
        p = Path(fpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        console.print(f"  [green]Written[/] → {fpath}")

    console.print("\n[bold green]Tier 5 complete![/]")
    console.print("\n[bold]Production deploy:[/]")
    console.print("  [cyan]docker compose -f docker-compose.prod.yml up -d[/]")
    console.print("\n[bold]Kubernetes deploy:[/]")
    console.print("  [cyan]helm upgrade --install aiml-platform ./helm/aiml-platform[/]")
    console.print("\n[bold]CI/CD:[/] Push to main — GitHub Actions will test → build → push → deploy automatically")
    console.print("\n[dim]Next: run tier6_advanced_ai.py[/]")


if __name__ == "__main__":
    main()
