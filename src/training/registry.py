"""src/training/registry.py — Model registry: promote, alias, load champion."""
from __future__ import annotations


import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import ModelMetrics


class ModelRegistry:
    """Thin wrapper around MLflow Model Registry for version lifecycle management."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        mlflow_cfg = self.cfg.get("training", {}).get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
        self.client = MlflowClient()
        self.reg_cfg = self.cfg.get("model_registry", {})
        self.model_name = self.cfg["project"]["name"]

    # ── Register ──────────────────────────────────────────────────────────────

    def register(self, run_id: str, artifact_path: str = "model") -> str:
        """Register a run's model artefact and return the version string."""
        uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri=uri, name=self.model_name)
        version = result.version
        logger.info(f"Registered model '{self.model_name}' version {version} from run {run_id}.")
        return version

    # ── Evaluate & promote ────────────────────────────────────────────────────

    def try_promote(self, version: str, metrics: ModelMetrics) -> bool:
        """Promote version to 'champion' if it beats the threshold."""
        thresholds = self.reg_cfg.get("promote_threshold", {})
        if not metrics.passes_threshold(thresholds):
            logger.warning(f"Version {version} did not meet promotion thresholds {thresholds}.")
            return False

        staging_alias = self.reg_cfg.get("staging_alias", "challenger")
        champion_alias = self.reg_cfg.get("production_alias", "champion")

        try:
            # Archive existing champion to challenger
            current = self.client.get_model_version_by_alias(self.model_name, champion_alias)
            self.client.set_registered_model_alias(self.model_name, staging_alias, current.version)
        except Exception:
            pass  # No current champion

        self.client.set_registered_model_alias(self.model_name, champion_alias, version)
        logger.info(f"Version {version} promoted to '{champion_alias}'.")
        return True

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_champion(self) -> object:
        alias = self.reg_cfg.get("production_alias", "champion")
        uri = f"models:/{self.model_name}@{alias}"
        logger.info(f"Loading model from {uri}")
        return mlflow.pytorch.load_model(uri)

    def load_version(self, version: str) -> object:
        uri = f"models:/{self.model_name}/{version}"
        return mlflow.pytorch.load_model(uri)
