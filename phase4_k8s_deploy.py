"""
PHASE 4 — Real Kubernetes Cluster Rollout
Run: python phase4_k8s_deploy.py

What this adds:
  - Complete Kubernetes manifests (no placeholders)
  - Automated deploy script with health checks
  - Rolling update strategy with automatic rollback
  - ConfigMap for environment-specific config
  - Secret management template
  - Namespace setup
  - Monitoring stack integration (Prometheus ServiceMonitor)
  - GitHub Actions deploy step (real, not echo)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
from rich.console import Console
console = Console()


K8S_NAMESPACE = """apiVersion: v1
kind: Namespace
metadata:
  name: aiml-platform
  labels:
    app: aiml-platform
    managed-by: helm
"""

K8S_CONFIGMAP = """apiVersion: v1
kind: ConfigMap
metadata:
  name: aiml-platform-config
  namespace: aiml-platform
data:
  MLFLOW_TRACKING_URI: "sqlite:///mlruns.db"
  MLFLOW_ALLOW_FILE_STORE: "true"
  GIT_PYTHON_REFRESH: "quiet"
  LOG_LEVEL: "info"
  PYTHONUNBUFFERED: "1"
  WORKERS: "2"
  PORT: "8000"
"""

K8S_SECRET_TEMPLATE = """apiVersion: v1
kind: Secret
metadata:
  name: aiml-platform-secrets
  namespace: aiml-platform
type: Opaque
stringData:
  # Replace with real values — use Sealed Secrets or External Secrets Operator in production
  API_KEY_1: "prod-key-replace-me"
  API_KEY_2: "prod-key-replace-me"
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
  REDIS_PASSWORD: ""
"""

K8S_DEPLOYMENT = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiml-platform-api
  namespace: aiml-platform
  labels:
    app: aiml-platform
    component: api
    version: "{{ .Values.image.tag }}"
  annotations:
    deployment.kubernetes.io/revision: "1"
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: aiml-platform
      component: api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # Zero-downtime rolling updates
  template:
    metadata:
      labels:
        app: aiml-platform
        component: api
        version: "{{ .Values.image.tag }}"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: aiml-platform
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          envFrom:
            - configMapRef:
                name: aiml-platform-config
            - secretRef:
                name: aiml-platform-secrets
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
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 20
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
          startupProbe:
            httpGet:
              path: /health
              port: http
            failureThreshold: 30
            periodSeconds: 10
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
            - name: logs
              mountPath: /app/logs
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: aiml-platform-models
        - name: logs
          emptyDir: {}
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: aiml-platform
"""

K8S_SERVICE = """apiVersion: v1
kind: Service
metadata:
  name: aiml-platform-api
  namespace: aiml-platform
  labels:
    app: aiml-platform
    component: api
spec:
  type: ClusterIP
  selector:
    app: aiml-platform
    component: api
  ports:
    - name: http
      port: 80
      targetPort: http
      protocol: TCP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aiml-platform-ingress
  namespace: aiml-platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.aiml-platform.example.com
      secretName: aiml-platform-tls
  rules:
    - host: api.aiml-platform.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: aiml-platform-api
                port:
                  name: http
"""

K8S_HPA = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aiml-platform-hpa
  namespace: aiml-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aiml-platform-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
"""

K8S_PVC = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: aiml-platform-models
  namespace: aiml-platform
spec:
  accessModes:
    - ReadWriteMany    # Multiple pods can read simultaneously
  storageClassName: efs-sc  # AWS EFS for shared storage; use gcs-fuse for GKE
  resources:
    requests:
      storage: 10Gi
"""

K8S_SERVICE_MONITOR = """apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aiml-platform
  namespace: aiml-platform
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: aiml-platform
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
"""

DEPLOY_SCRIPT = '''#!/usr/bin/env python3
"""
scripts/deploy_k8s.py — Production Kubernetes deployment with health checks + rollback.

Usage:
  python scripts/deploy_k8s.py --image ghcr.io/org/aiml-platform:sha-abc123
  python scripts/deploy_k8s.py --image ... --namespace staging --dry-run
"""
from __future__ import annotations

import subprocess
import sys
import time
import typer
from pathlib import Path

app = typer.Typer()


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, print it, return result."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def wait_for_rollout(deployment: str, namespace: str,
                     timeout: int = 300) -> bool:
    """Wait for a deployment rollout to complete."""
    print(f"Waiting for rollout: {deployment} (timeout={timeout}s)")
    result = run([
        "kubectl", "rollout", "status",
        f"deployment/{deployment}",
        f"--namespace={namespace}",
        f"--timeout={timeout}s",
    ], check=False)
    return result.returncode == 0


def get_current_image(deployment: str, namespace: str) -> str | None:
    """Get the current image tag of a deployment."""
    result = run([
        "kubectl", "get", "deployment", deployment,
        f"--namespace={namespace}",
        "-o", "jsonpath={.spec.template.spec.containers[0].image}",
    ], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def health_check(namespace: str, retries: int = 5) -> bool:
    """Port-forward and check /health endpoint."""
    import socket, threading, http.client

    for attempt in range(retries):
        try:
            # Forward port 8000 locally
            fwd = subprocess.Popen([
                "kubectl", "port-forward",
                f"--namespace={namespace}",
                "service/aiml-platform-api",
                "18000:80",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Wait for port-forward to establish

            conn = http.client.HTTPConnection("localhost", 18000, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            fwd.terminate()

            if resp.status == 200:
                print(f"Health check PASSED (attempt {attempt + 1})")
                return True
            print(f"Health check failed: HTTP {resp.status}")
        except Exception as e:
            print(f"Health check attempt {attempt + 1} failed: {e}")
            try:
                fwd.terminate()
            except Exception:
                pass
        time.sleep(5)

    return False


@app.command()
def deploy(
    image: str = typer.Option(..., help="Full image URI to deploy"),
    namespace: str = typer.Option("aiml-platform", help="Kubernetes namespace"),
    chart: str = typer.Option("./helm/aiml-platform", help="Helm chart path"),
    dry_run: bool = typer.Option(False, help="Dry run — show what would happen"),
    timeout: int = typer.Option(300, help="Rollout timeout in seconds"),
    skip_health: bool = typer.Option(False, help="Skip health check after deploy"),
):
    """Deploy to Kubernetes with health checks and automatic rollback."""

    print(f"\\n{'='*60}")
    print(f"Deploying: {image}")
    print(f"Namespace: {namespace}")
    print(f"Dry run:   {dry_run}")
    print(f"{'='*60}\\n")

    # Save current image for rollback
    current_image = get_current_image("aiml-platform-api", namespace)
    print(f"Current image: {current_image or 'none (first deploy)'}")

    # Parse tag from image
    tag = image.split(":")[-1] if ":" in image else "latest"
    repo = image.rsplit(":", 1)[0] if ":" in image else image

    # Build helm command
    helm_cmd = [
        "helm", "upgrade", "--install", "aiml-platform", chart,
        f"--namespace={namespace}",
        "--create-namespace",
        f"--set=image.repository={repo}",
        f"--set=image.tag={tag}",
        "--set=replicaCount=2",
        f"--timeout={timeout}s",
        "--wait",
        "--atomic",  # Automatic rollback on failure
    ]

    if dry_run:
        helm_cmd.append("--dry-run")

    try:
        run(helm_cmd)
    except RuntimeError as e:
        print(f"\\nDeploy FAILED: {e}")
        print("Helm --atomic flag will have triggered automatic rollback.")
        sys.exit(1)

    if dry_run:
        print("\\nDry run complete — no changes made.")
        return

    # Wait for rollout
    if not wait_for_rollout("aiml-platform-api", namespace, timeout):
        print("\\nRollout timed out! Rolling back...")
        run(["helm", "rollback", "aiml-platform", "0",
             f"--namespace={namespace}"])
        sys.exit(1)

    # Health check
    if not skip_health:
        if not health_check(namespace):
            print("\\nHealth check FAILED! Rolling back...")
            run(["helm", "rollback", "aiml-platform", "0",
                 f"--namespace={namespace}"])
            sys.exit(1)

    print(f"\\n✅ Deployment successful!")
    print(f"   Image:     {image}")
    print(f"   Namespace: {namespace}")


if __name__ == "__main__":
    app()
'''

GITHUB_ACTIONS_DEPLOY = """  # ── Deploy to Production ──────────────────────────────────────────────────────
  deploy-production:
    runs-on: ubuntu-latest
    needs: build-push
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install kubectl
        uses: azure/setup-kubectl@v4
        with:
          version: 'v1.29.0'

      - name: Install Helm
        uses: azure/setup-helm@v4
        with:
          version: 'v3.14.0'

      - name: Configure kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBECONFIG }}" | base64 -d > ~/.kube/config
          kubectl cluster-info

      - name: Deploy to production
        run: |
          python scripts/deploy_k8s.py \\
            --image ghcr.io/${{ github.repository }}/aiml-platform:sha-${{ github.sha }} \\
            --namespace aiml-platform \\
            --timeout 300

      - name: Verify deployment
        run: |
          kubectl get pods -n aiml-platform
          kubectl get hpa  -n aiml-platform
"""


def main() -> None:
    console.print("[bold cyan]Phase 4 — Kubernetes Real Cluster Rollout[/]\n")

    files = {
        "helm/aiml-platform/templates/namespace.yaml":       K8S_NAMESPACE,
        "helm/aiml-platform/templates/configmap.yaml":       K8S_CONFIGMAP,
        "helm/aiml-platform/templates/secret.yaml":          K8S_SECRET_TEMPLATE,
        "helm/aiml-platform/templates/deployment.yaml":      K8S_DEPLOYMENT,
        "helm/aiml-platform/templates/service.yaml":         K8S_SERVICE,
        "helm/aiml-platform/templates/hpa.yaml":             K8S_HPA,
        "helm/aiml-platform/templates/pvc.yaml":             K8S_PVC,
        "helm/aiml-platform/templates/servicemonitor.yaml":  K8S_SERVICE_MONITOR,
        "scripts/deploy_k8s.py":                             DEPLOY_SCRIPT,
    }

    for fpath, content in files.items():
        p = Path(fpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        console.print(f"  [green]Written[/] → {fpath}")

    # Append production deploy step to CI/CD workflow
    ci_path = Path(".github/workflows/ci_cd.yml")
    if ci_path.exists():
        content = ci_path.read_text(encoding="utf-8")
        if "deploy-production" not in content:
            content += "\n" + GITHUB_ACTIONS_DEPLOY
            ci_path.write_text(content, encoding="utf-8")
            console.print(f"  [green]Updated[/] → .github/workflows/ci_cd.yml")

    console.print("\n[bold green]Phase 4 complete![/]")
    console.print("\nNew Kubernetes manifests:")
    console.print("  namespace.yaml      — isolated namespace")
    console.print("  configmap.yaml      — environment config")
    console.print("  secret.yaml         — credential template")
    console.print("  deployment.yaml     — zero-downtime rolling deploy")
    console.print("  service.yaml        — ClusterIP + Ingress + TLS")
    console.print("  hpa.yaml            — auto-scale 2→20 pods")
    console.print("  pvc.yaml            — shared model storage (EFS/GCS)")
    console.print("  servicemonitor.yaml — Prometheus scraping")
    console.print("  scripts/deploy_k8s.py — deploy with health check + rollback")
    console.print("\nDeploy command:")
    console.print("  [cyan]kubectl create namespace aiml-platform[/]")
    console.print("  [cyan]helm upgrade --install aiml-platform ./helm/aiml-platform --namespace aiml-platform[/]")
    console.print("  [cyan]python scripts/deploy_k8s.py --image ghcr.io/org/aiml-platform:latest[/]")


if __name__ == "__main__":
    main()
