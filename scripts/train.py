#!/usr/bin/env python
"""scripts/train.py — CLI entry point for model training."""
from __future__ import annotations

import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="Train a model.")
console = Console()


@app.command()
def main(
    experiment: str = typer.Option("default_experiment", help="MLflow experiment name"),
    config: Path = typer.Option("configs/config.yaml", help="Path to config YAML"),
    run_hpo: bool = typer.Option(False, help="Run hyperparameter optimisation first"),
    dry_run: bool = typer.Option(False, help="Load data and model but skip training"),
):
    from src.utils.config import load_config
    from src.utils.logger import logger
    from src.features.pipeline import FeaturePipeline
    from src.training.trainer import Trainer, TextDataset
    from src.training.registry import ModelRegistry
    from src.utils.schema import ModelMetrics
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import pandas as pd

    cfg = load_config(config)
    cfg["training"]["mlflow"]["experiment_name"] = experiment
    console.print(f"[bold cyan]Training run:[/] {experiment}")

    # 1. Ensure features exist
    feat_path = Path(cfg["data"]["features_path"])
    if not (feat_path / "train.parquet").exists():
        console.print("[yellow]Features not found — running feature pipeline...[/]")
        FeaturePipeline(config=cfg).run()

    train_df = pd.read_parquet(feat_path / "train.parquet")
    val_df = pd.read_parquet(feat_path / "val.parquet")

    if dry_run:
        console.print(f"[green]Dry run OK[/]: train={len(train_df):,} val={len(val_df):,}")
        return

    # 2. Build dataloaders
    model_name = cfg["training"]["model_name"]
    max_length = cfg["training"].get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_col = "text" if "text" in train_df.columns else train_df.columns[0]
    label_col = "label" if "label" in train_df.columns else train_df.columns[-1]

    train_ds = TextDataset(train_df[text_col].tolist(), train_df[label_col].tolist(), tokenizer, max_length)
    val_ds = TextDataset(val_df[text_col].tolist(), val_df[label_col].tolist(), tokenizer, max_length)
    bs = cfg["training"]["hyperparams"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    # 3. Optionally run HPO first
    if run_hpo:
        from src.training.hpo import HPORunner
        console.print("[bold]Running HPO...[/]")
        best_params = HPORunner(train_loader, val_loader, config=cfg).run()
        cfg["training"]["hyperparams"].update(best_params)

    # 4. Train
    trainer = Trainer(config=cfg)
    model, metrics = trainer.train(train_loader, val_loader, run_name=experiment)
    console.print(f"[green]Training complete.[/] val_f1={metrics.f1:.4f} val_acc={metrics.accuracy:.4f}")

    # 5. Register & promote
    import mlflow
    mlflow.set_tracking_uri(cfg["training"]["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"], max_results=1)
    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        registry = ModelRegistry(config=cfg)
        version = registry.register(run_id)
        promoted = registry.try_promote(version, metrics)
        if promoted:
            console.print(f"[bold green]Model version {version} promoted to champion![/]")
        else:
            console.print(f"[yellow]Model version {version} registered as challenger (thresholds not met).[/]")


if __name__ == "__main__":
    app()
