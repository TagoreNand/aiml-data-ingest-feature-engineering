"""src/training/hpo.py — Hyperparameter optimisation with Optuna."""
from __future__ import annotations

from typing import Any

import optuna
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import TrainingConfig
from src.training.trainer import Trainer


class HPORunner:
    """Runs Optuna hyperparameter search and returns the best config."""

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, config: dict | None = None) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or load_config()
        self.hpo_cfg = self.cfg.get("training", {}).get("hpo", {})

    def _objective(self, trial: optuna.Trial) -> float:
        hp = TrainingConfig(
            model_name=self.cfg["training"]["model_name"],
            task=self.cfg["training"]["task"],
            num_labels=self.cfg["training"].get("num_labels", 2),
            max_length=self.cfg["training"].get("max_length", 128),
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            num_epochs=trial.suggest_int("num_epochs", 2, 8),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.05, 0.2),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            gradient_clip=trial.suggest_float("gradient_clip", 0.5, 2.0),
            experiment_name=f"hpo_trial_{trial.number}",
        )
        trainer = Trainer(cfg=hp, config=self.cfg)
        _, metrics = trainer.train(self.train_loader, self.val_loader, run_name=f"hpo_trial_{trial.number}")

        metric_name = self.hpo_cfg.get("metric", "val_f1")
        return getattr(metrics, metric_name.replace("val_", ""), 0.0) or 0.0

    def run(self) -> dict[str, Any]:
        direction = self.hpo_cfg.get("direction", "maximize")
        n_trials = self.hpo_cfg.get("n_trials", 20)
        timeout = self.hpo_cfg.get("timeout_seconds", 3600)
        sampler_name = self.hpo_cfg.get("sampler", "tpe")

        sampler = optuna.samplers.TPESampler() if sampler_name == "tpe" else optuna.samplers.RandomSampler()
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout)

        logger.info(f"HPO complete. Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params
