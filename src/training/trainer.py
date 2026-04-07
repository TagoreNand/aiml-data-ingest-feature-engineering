"""src/training/trainer.py — Training loop with MLflow tracking, checkpointing, and early stopping."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import ModelMetrics, TrainingConfig
from src.training.models import build_model


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[Any], tokenizer: Any, max_length: int = 128) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.counter: int = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrainingConfig | None = None, config: dict | None = None) -> None:
        self.raw_cfg = config or load_config()
        self.cfg = cfg or self._build_training_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer using device: {self.device}")

    def _build_training_config(self) -> TrainingConfig:
        tc = self.raw_cfg.get("training", {})
        hp = tc.get("hyperparams", {})
        return TrainingConfig(
            model_name=tc.get("model_name", "bert-base-uncased"),
            task=tc.get("task", "text_classification"),
            num_labels=tc.get("num_labels", 2),
            max_length=tc.get("max_length", 128),
            learning_rate=hp.get("learning_rate", 2e-5),
            batch_size=hp.get("batch_size", 32),
            num_epochs=hp.get("num_epochs", 10),
            warmup_ratio=hp.get("warmup_ratio", 0.1),
            weight_decay=hp.get("weight_decay", 0.01),
            gradient_clip=hp.get("gradient_clip", 1.0),
            experiment_name=self.raw_cfg.get("training", {}).get("mlflow", {}).get("experiment_name", "default"),
        )

    # ── Core training step ────────────────────────────────────────────────────

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: Any, scheduler: Any) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, model: nn.Module, loader: DataLoader) -> tuple[float, ModelMetrics]:
        model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out["loss"].item()
            preds = out["logits"].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)
        accuracy = (all_preds_arr == all_labels_arr).mean()

        # Binary F1
        tp = ((all_preds_arr == 1) & (all_labels_arr == 1)).sum()
        fp = ((all_preds_arr == 1) & (all_labels_arr == 0)).sum()
        fn = ((all_preds_arr == 0) & (all_labels_arr == 1)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return avg_loss, ModelMetrics(accuracy=float(accuracy), f1=float(f1),
                                      precision=float(precision), recall=float(recall), loss=avg_loss)

    # ── Full training run ─────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: str | None = None,
    ) -> tuple[nn.Module, ModelMetrics]:
        mlflow_cfg = self.raw_cfg.get("training", {}).get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(self.cfg.experiment_name)

        model = build_model(self.cfg.task, self.cfg.model_name, self.cfg.num_labels).to(self.device)
        total_steps = len(train_loader) * self.cfg.num_epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        optimizer = AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        early_stop = EarlyStopping(patience=3)
        best_metrics = ModelMetrics(loss=float("inf"))

        with mlflow.start_run(run_name=run_name or self.cfg.experiment_name):
            mlflow.log_params(self.cfg.model_dump())

            for epoch in range(1, self.cfg.num_epochs + 1):
                t0 = time.time()
                train_loss = self._train_epoch(model, train_loader, optimizer, scheduler)
                val_loss, metrics = self._eval_epoch(model, val_loader)
                elapsed = time.time() - t0

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": metrics.f1 or 0,
                    "val_accuracy": metrics.accuracy or 0,
                }, step=epoch)

                logger.info(
                    f"Epoch {epoch}/{self.cfg.num_epochs} "
                    f"| train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"f1={metrics.f1:.4f} acc={metrics.accuracy:.4f} | {elapsed:.1f}s"
                )

                if (metrics.loss or float("inf")) < (best_metrics.loss or float("inf")):
                    best_metrics = metrics
                    ckpt_path = Path("models") / "best_checkpoint.pt"
                    ckpt_path.parent.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), ckpt_path)
                    mlflow.log_artifact(str(ckpt_path))

                if early_stop.step(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.model_dump().items() if v is not None})

        # Load best checkpoint
        best_ckpt = Path("models/best_checkpoint.pt")
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        return model, best_metrics
