#!/usr/bin/env python
"""scripts/evaluate.py — Evaluate a registered model version against the test split."""
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Evaluate a model against the test set.")
console = Console()


@app.command()
def main(
    version: str = typer.Option("champion", help="Model version or alias (e.g. champion, 3)"),
    config: Path = typer.Option("configs/config.yaml", help="Path to config YAML"),
):
    import pandas as pd
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from src.utils.config import load_config
    from src.training.registry import ModelRegistry
    from src.training.trainer import TextDataset, Trainer

    cfg = load_config(config)
    feat_path = Path(cfg["data"]["features_path"])
    test_df = pd.read_parquet(feat_path / "test.parquet")

    model_name = cfg["training"]["model_name"]
    max_length = cfg["training"].get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_col = "text" if "text" in test_df.columns else test_df.columns[0]
    label_col = "label" if "label" in test_df.columns else test_df.columns[-1]

    test_ds = TextDataset(test_df[text_col].tolist(), test_df[label_col].tolist(), tokenizer, max_length)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["hyperparams"]["batch_size"])

    registry = ModelRegistry(config=cfg)
    model = registry.load_champion() if version == "champion" else registry.load_version(version)

    trainer = Trainer(config=cfg)
    _, metrics = trainer._eval_epoch(model, test_loader)

    table = Table(title=f"Evaluation — version: {version}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.model_dump().items():
        if v is not None:
            table.add_row(k, f"{v:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
