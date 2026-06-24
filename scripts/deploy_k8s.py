#!/usr/bin/env python3
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
    import http.client

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

    print(f"\n{'='*60}")
    print(f"Deploying: {image}")
    print(f"Namespace: {namespace}")
    print(f"Dry run:   {dry_run}")
    print(f"{'='*60}\n")

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
        print(f"\nDeploy FAILED: {e}")
        print("Helm --atomic flag will have triggered automatic rollback.")
        sys.exit(1)

    if dry_run:
        print("\nDry run complete — no changes made.")
        return

    # Wait for rollout
    if not wait_for_rollout("aiml-platform-api", namespace, timeout):
        print("\nRollout timed out! Rolling back...")
        run(["helm", "rollback", "aiml-platform", "0",
             f"--namespace={namespace}"])
        sys.exit(1)

    # Health check
    if not skip_health:
        if not health_check(namespace):
            print("\nHealth check FAILED! Rolling back...")
            run(["helm", "rollback", "aiml-platform", "0",
                 f"--namespace={namespace}"])
            sys.exit(1)

    print("\n✅ Deployment successful!")
    print(f"   Image:     {image}")
    print(f"   Namespace: {namespace}")


if __name__ == "__main__":
    app()
