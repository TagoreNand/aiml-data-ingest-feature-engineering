#!/usr/bin/env python
"""scripts/build_features.py — CLI entry point for feature engineering."""
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="Run the feature engineering pipeline.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option("configs/config.yaml", help="Path to config YAML"),
):
    from src.utils.config import load_config
    from src.features.pipeline import FeaturePipeline

    cfg = load_config(config)
    console.print("[bold cyan]Building features...[/]")
    pipeline = FeaturePipeline(config=cfg)
    paths = pipeline.run()
    for split, path in paths.items():
        console.print(f"  [green]{split}[/] → {path}")
    console.print("[green]Feature pipeline complete.[/]")


if __name__ == "__main__":
    app()
