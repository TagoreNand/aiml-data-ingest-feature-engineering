#!/usr/bin/env python
"""scripts/run_monitoring.py — CLI entry point for drift detection and alerting."""
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="Run drift detection and performance monitoring.")
console = Console()


@app.command()
def main(
    current: Path = typer.Argument(..., help="Path to current production data (parquet)"),
    config: Path = typer.Option("configs/config.yaml", help="Path to config YAML"),
):
    import pandas as pd
    from src.utils.config import load_config
    from src.monitoring.drift import DriftDetector, RetrainingTrigger
    from src.monitoring.metrics import get_collector

    cfg = load_config(config)
    console.print("[bold cyan]Running drift detection...[/]")

    current_df = pd.read_parquet(current)
    detector = DriftDetector(config=cfg)
    report = detector.run_evidently(current_df)
    report_path = detector.save_report(report)

    console.print(f"  dataset_drift : [{'red' if report.dataset_drift else 'green'}]{report.dataset_drift}[/]")
    console.print(f"  drift_score   : {report.drift_score:.3f}")
    if report.drifted_columns:
        console.print(f"  drifted cols  : {', '.join(report.drifted_columns)}")
    console.print(f"  Report saved  : {report_path}")

    trigger = RetrainingTrigger(config=cfg)
    should, reason = trigger.should_retrain(report)
    if should:
        console.print(f"[bold yellow]Retraining triggered:[/] {reason}")
        trigger.trigger_retrain(reason)
    else:
        console.print(f"[green]No retraining needed:[/] {reason}")

    snap = get_collector().snapshot()
    console.print("\n[bold]Performance snapshot:[/]")
    console.print(f"  requests : {snap['total_requests']:,}")
    console.print(f"  errors   : {snap['error_rate']:.2%}")
    console.print(f"  p99 lat  : {snap['latency']['p99_ms']} ms")

    alerts = get_collector().check_alerts(cfg)
    for alert in alerts:
        console.print(f"  [bold red]ALERT:[/] {alert}")


if __name__ == "__main__":
    app()
