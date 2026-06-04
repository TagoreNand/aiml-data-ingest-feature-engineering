"""src/training/distributed.py — Distributed training with Ray Train + Ray Tune."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.utils.logger import logger


# ── Ray Train: Distributed Training ──────────────────────────────────────────

def train_fn_per_worker(config: dict) -> None:
    """
    Training function executed on EACH Ray worker.
    Ray automatically handles:
      - Device placement (GPU/CPU per worker)
      - Gradient synchronisation across workers
      - Fault tolerance and worker restarts
    """
    import ray.train as ray_train
    from ray.train import Checkpoint

    from src.training.models import build_model
    from src.training.trainer import TextDataset
    from sklearn.metrics import f1_score, accuracy_score
    import pandas as pd

    # Each worker gets its own device
    device = ray_train.torch.get_device()
    logger.info(f"Worker on device: {device}")

    # Load data
    train_df = pd.read_parquet(config["train_path"])
    val_df   = pd.read_parquet(config["val_path"])

    text_col  = "text"  if "text"  in train_df.columns else train_df.columns[0]
    label_col = "label" if "label" in train_df.columns else train_df.columns[-1]
    train_df[label_col] = train_df[label_col].astype(int)
    val_df[label_col]   = val_df[label_col].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_ds  = TextDataset(train_df[text_col].tolist(),
                             train_df[label_col].tolist(),
                             tokenizer, config["max_length"])
    val_ds    = TextDataset(val_df[text_col].tolist(),
                             val_df[label_col].tolist(),
                             tokenizer, config["max_length"])

    # Ray handles distributed sampling automatically
    train_loader = ray_train.torch.prepare_data_loader(
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    # Build model and wrap with DDP (Distributed Data Parallel)
    model = build_model(config["task"], config["model_name"], config["num_labels"])
    model = ray_train.torch.prepare_model(model)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                      weight_decay=config["weight_decay"])
    total_steps   = len(train_loader) * config["num_epochs"]
    warmup_steps  = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn   = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, config["num_epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            out  = model(**batch, labels=labels)
            loss = out["loss"] if isinstance(out, dict) else loss_fn(out.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ── Eval ──
        model.eval()
        all_preds, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                out = model(**batch, labels=labels)
                logits = out["logits"] if isinstance(out, dict) else out.logits
                val_loss += loss_fn(logits, labels).item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        # Save checkpoint if best
        checkpoint = None
        if val_f1 > best_f1:
            best_f1 = val_f1
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(model.state_dict(), os.path.join(tmpdir, "model.pt"))
                checkpoint = Checkpoint.from_directory(tmpdir)

        # Report metrics back to Ray Tune / Train
        ray_train.report(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_f1": val_f1,
                "val_acc": val_acc,
            },
            checkpoint=checkpoint,
        )


class DistributedTrainer:
    """
    Orchestrates distributed training using Ray Train.
    
    Modes:
      - num_workers=1: single GPU (same as regular training)
      - num_workers=N: N-GPU data parallel training
      - use_gpu=True:  each worker gets one GPU
      - use_gpu=False: CPU training (useful for debugging)
    """

    def __init__(self, config: dict, num_workers: int = 1, use_gpu: bool = False):
        self.config     = config
        self.num_workers = num_workers
        self.use_gpu    = use_gpu and torch.cuda.is_available()

    def train(self, experiment_name: str = "distributed_run") -> dict:
        try:
            import ray
            from ray import train as ray_train
            from ray.train.torch import TorchTrainer
            from ray.train import RunConfig, ScalingConfig
        except ImportError:
            raise RuntimeError("Install Ray: pip install ray[train]")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info(f"Ray initialised | nodes: {len(ray.nodes())}")

        tc = self.config["training"]
        hp = tc["hyperparams"]

        train_config = {
            "model_name":    tc["model_name"],
            "task":          tc["task"],
            "num_labels":    tc["num_labels"],
            "max_length":    tc.get("max_length", 128),
            "learning_rate": hp["learning_rate"],
            "batch_size":    hp["batch_size"],
            "num_epochs":    hp["num_epochs"],
            "weight_decay":  hp["weight_decay"],
            "train_path":    "data/features/train.parquet",
            "val_path":      "data/features/val.parquet",
        }

        scaling = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker={
                "CPU": 2,
                "GPU": 1 if self.use_gpu else 0,
            },
        )

        trainer = TorchTrainer(
            train_loop_per_worker=train_fn_per_worker,
            train_loop_config=train_config,
            scaling_config=scaling,
            run_config=RunConfig(
                name=experiment_name,
                storage_path=str(Path("ray_results").absolute()),
            ),
        )

        logger.info(f"Starting distributed training | "
                    f"workers={self.num_workers} gpu={self.use_gpu}")
        result = trainer.fit()

        best_metrics = result.metrics
        best_checkpoint = result.checkpoint
        logger.info(f"Training complete | best val_f1={best_metrics.get('val_f1', 0):.4f}")

        # Save best model locally
        if best_checkpoint:
            checkpoint_dir = best_checkpoint.to_directory("models/distributed_best")
            logger.info(f"Best checkpoint saved → {checkpoint_dir}")

        return best_metrics


# ── Ray Tune: Hyperparameter Optimisation ─────────────────────────────────────

class RayTuneHPO:
    """
    Hyperparameter optimisation using Ray Tune.
    Searches over learning rate, batch size, and weight decay
    using ASHA (Asynchronous Successive Halving) scheduler —
    automatically kills poor configurations early.
    """

    def __init__(self, config: dict, num_samples: int = 10,
                 max_epochs: int = 5, num_workers: int = 1):
        self.config      = config
        self.num_samples = num_samples
        self.max_epochs  = max_epochs
        self.num_workers = num_workers

    def run(self) -> dict:
        try:
            import ray
            from ray import tune
            from ray.train.torch import TorchTrainer
            from ray.train import RunConfig, ScalingConfig
            from ray.tune.schedulers import ASHAScheduler
            from ray.tune import TuneConfig
        except ImportError:
            raise RuntimeError("Install Ray: pip install ray[tune]")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        tc = self.config["training"]

        # Search space — Ray Tune will sample from these distributions
        search_space = {
            "model_name":    tc["model_name"],
            "task":          tc["task"],
            "num_labels":    tc["num_labels"],
            "max_length":    tc.get("max_length", 128),
            "weight_decay":  tc["hyperparams"]["weight_decay"],
            "train_path":    "data/features/train.parquet",
            "val_path":      "data/features/val.parquet",
            # ── Tunable hyperparameters ──
            "learning_rate": tune.loguniform(1e-5, 5e-4),
            "batch_size":    tune.choice([8, 16, 32]),
            "num_epochs":    self.max_epochs,
        }

        # ASHA: kill bad trials early, keep good ones
        scheduler = ASHAScheduler(
            metric="val_f1",
            mode="max",
            max_t=self.max_epochs,
            grace_period=1,
            reduction_factor=2,
        )

        trainer = TorchTrainer(
            train_loop_per_worker=train_fn_per_worker,
            train_loop_config=search_space,
            scaling_config=ScalingConfig(num_workers=self.num_workers),
        )

        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": search_space},
            tune_config=TuneConfig(
                scheduler=scheduler,
                num_samples=self.num_samples,
                metric="val_f1",
                mode="max",
            ),
            run_config=RunConfig(
                name="ray_tune_hpo",
                storage_path=str(Path("ray_results").absolute()),
            ),
        )

        logger.info(f"Starting Ray Tune HPO | samples={self.num_samples}")
        results = tuner.fit()
        best   = results.get_best_result(metric="val_f1", mode="max")

        logger.info(f"Best config: {best.config}")
        logger.info(f"Best val_f1: {best.metrics.get('val_f1', 0):.4f}")
        return {"best_config": best.config, "best_metrics": best.metrics}
