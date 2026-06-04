"""Launch distributed training with Ray."""
from __future__ import annotations
import sys
import typer
from pathlib import Path

sys.path.insert(0, ".")

def main(
    workers: int = typer.Option(1, help="Number of Ray workers"),
    gpu: bool = typer.Option(False, help="Use GPU per worker"),
    hpo: bool = typer.Option(False, help="Run Ray Tune HPO"),
    samples: int = typer.Option(10, help="HPO trials"),
    config: Path = typer.Option("configs/config.yaml"),
):
    from src.utils.config import load_config
    from src.training.distributed import DistributedTrainer, RayTuneHPO
    from rich.console import Console
    console = Console()

    cfg = load_config(config)

    if hpo:
        console.print(f"[bold cyan]Ray Tune HPO | samples={samples}[/]")
        tuner = RayTuneHPO(cfg, num_samples=samples, num_workers=workers)
        result = tuner.run()
        console.print(f"Best val_f1: [green]{result['best_metrics'].get('val_f1', 0):.4f}[/]")
    else:
        console.print(f"[bold cyan]Distributed training | workers={workers} gpu={gpu}[/]")
        trainer = DistributedTrainer(cfg, num_workers=workers, use_gpu=gpu)
        metrics = trainer.train()
        console.print(f"Training complete | val_f1=[green]{metrics.get('val_f1', 0):.4f}[/]")

if __name__ == "__main__":
    typer.run(main)
