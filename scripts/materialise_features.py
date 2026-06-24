"""Materialise features to online (Redis) + offline (Parquet) stores."""
from __future__ import annotations
import sys
import typer
from pathlib import Path

sys.path.insert(0, ".")

def main(config: Path = typer.Option("configs/config.yaml"),
         version: str = typer.Option("latest")):
    import pandas as pd
    from rich.console import Console
    from src.utils.config import load_config
    from src.features.store import FeatureStore, get_default_feature_views

    console = Console()
    cfg = load_config(config)
    store = FeatureStore(cfg)

    console.print("[bold cyan]Materialising features...[/]")

    for view in get_default_feature_views():
        store.register_view(view)

    # Load feature data
    feat_path = Path("data/features/train.parquet")
    if not feat_path.exists():
        console.print("[red]Run feature pipeline first![/]")
        raise typer.Exit(1)

    df = pd.read_parquet(feat_path)
    for view in get_default_feature_views():
        try:
            store.materialise(view, df, version=version)
            console.print(f"  [green]✓[/] {view.name} → online + offline")
        except Exception as e:
            console.print(f"  [yellow]⚠[/] {view.name}: {e}")

    console.print("[bold green]Materialisation complete![/]")

if __name__ == "__main__":
    typer.run(main)
