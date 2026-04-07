#!/usr/bin/env python
"""scripts/run_ingestion.py — CLI entry point for data ingestion."""
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="Run a data ingestor.")
console = Console()


@app.command()
def main(
    source: str = typer.Argument("batch", help="Ingestor type: batch | kafka | api"),
    config: Path = typer.Option("configs/config.yaml", help="Path to config YAML"),
):
    from src.utils.config import load_config
    from src.ingestion.ingestors import get_ingestor

    cfg = load_config(config)
    console.print(f"[bold cyan]Running ingestor:[/] {source}")
    ingestor = get_ingestor(source, config=cfg)
    count = ingestor.run()
    console.print(f"[green]Done.[/] {count:,} records ingested.")


if __name__ == "__main__":
    app()
