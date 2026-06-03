"""
TIER 1 — Real dataset + data validation
Run: python tier1_real_data.py

What this does:
  - Downloads IMDb (50k movie reviews, balanced sentiment)
  - Validates data with Great Expectations-style checks
  - Saves as clean CSV to data/raw/imdb_dataset.csv
  - Runs feature pipeline to produce train/val/test parquet splits
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()


# ── Step 1: Load real dataset ─────────────────────────────────────────────────

def load_imdb() -> pd.DataFrame:
    console.print("[bold cyan]Loading IMDb dataset from HuggingFace...[/]")
    import urllib.request, json, io
    # Use the datasets library with explicit trust
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", trust_remote_code=True)
    train_df = pd.DataFrame(ds["train"])
    test_df  = pd.DataFrame(ds["test"])
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.rename(columns={"text": "text", "label": "label"})
    df["id"] = [f"imdb_{i}" for i in range(len(df))]
    console.print(f"  Loaded [green]{len(df):,}[/] reviews | labels: {df['label'].value_counts().to_dict()}")
    return df


# ── Step 2: Data validation suite ────────────────────────────────────────────

class DataValidator:
    """Lightweight validation suite — production replacement for Great Expectations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: list[tuple[str, bool, str]] = []

    def _check(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    def validate(self) -> bool:
        df = self.df

        # Schema checks
        self._check("required columns present",
                    {"text", "label", "id"}.issubset(df.columns),
                    f"columns: {list(df.columns)}")

        # Nulls
        null_pct = df["text"].isnull().mean()
        self._check("text null rate < 1%", null_pct < 0.01, f"{null_pct:.2%} nulls")

        # Label distribution
        label_counts = df["label"].value_counts(normalize=True)
        min_class_pct = label_counts.min()
        self._check("label balance > 30%", min_class_pct > 0.30,
                    f"min class: {min_class_pct:.1%}")

        # Text length
        lengths = df["text"].str.len()
        self._check("median text length > 50 chars", lengths.median() > 50,
                    f"median: {lengths.median():.0f} chars")

        # Duplicates
        dup_rate = df.duplicated(subset=["text"]).mean()
        self._check("duplicate rate < 5%", dup_rate < 0.05, f"{dup_rate:.2%} duplicates")

        # Label dtype
        self._check("label is numeric", pd.api.types.is_numeric_dtype(df["label"]),
                    f"dtype: {df['label'].dtype}")

        # ID uniqueness
        self._check("IDs are unique", df["id"].nunique() == len(df),
                    f"{df['id'].nunique()} unique / {len(df)} total")

        return all(passed for _, passed, _ in self.results)

    def report(self) -> None:
        table = Table(title="Data validation report")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail", style="dim")
        for name, passed, detail in self.results:
            status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
            table.add_row(name, status, detail)
        console.print(table)


# ── Step 3: Sample for faster local training ──────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Take a balanced sample for faster local iteration."""
    return (df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(n=n // 2, random_state=42))
              .reset_index(drop=True))


# ── Step 4: Save and run feature pipeline ────────────────────────────────────

def main() -> None:
    # 1. Load
    df = load_imdb()

    # 2. Validate
    console.print("\n[bold cyan]Running data validation...[/]")
    validator = DataValidator(df)
    all_passed = validator.validate()
    validator.report()
    if not all_passed:
        console.print("[bold red]Validation failed — fix data issues before proceeding.[/]")
        sys.exit(1)
    console.print("[green]All checks passed.[/]\n")

    # 3. Sample for local dev (use full 50k for production)
    sample_df = stratified_sample(df, n=2000)
    console.print(f"[bold cyan]Using stratified sample:[/] {len(sample_df):,} rows "
                  f"({sample_df['label'].value_counts().to_dict()})")

    # 4. Save
    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)
    out = raw_path / "imdb_dataset.csv"
    sample_df.to_csv(out, index=False)
    console.print(f"[green]Saved[/] → {out}")

    # 5. Also save full dataset for reference
    full_out = raw_path / "imdb_full.csv"
    df[["id", "text", "label"]].to_csv(full_out, index=False)
    console.print(f"[green]Saved full dataset[/] → {full_out} ({len(df):,} rows)")

    # 6. Run feature pipeline
    console.print("\n[bold cyan]Running feature pipeline...[/]")
    from src.utils.config import load_config
    from src.features.pipeline import FeaturePipeline

    cfg = load_config("configs/config.yaml")
    # Point batch source to data/raw
    cfg["ingestion"]["batch"]["source_path"] = "data/raw"
    cfg["ingestion"]["batch"]["format"] = "csv"

    pipeline = FeaturePipeline(config=cfg)

    # Load directly from CSV (bypass ingestor for cleanliness)
    df_feat = pd.read_csv(out)
    df_feat = pipeline.validate(df_feat)
    df_feat = pipeline.engineer(df_feat)
    train, val, test = pipeline.split(df_feat)
    train_t = pipeline.transformer.fit_transform(train)
    val_t   = pipeline.transformer.transform(val)
    test_t  = pipeline.transformer.transform(test)
    paths = {
        "train": pipeline.materialise(train_t, "train"),
        "val":   pipeline.materialise(val_t,   "val"),
        "test":  pipeline.materialise(test_t,  "test"),
    }

    # Save reference dataset for drift monitoring
    ref_path = Path("data/processed/reference_dataset.parquet")
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    train_t.to_parquet(ref_path, index=False)

    console.print("\n[bold green]Tier 1 complete![/]")
    table = Table(title="Feature splits")
    table.add_column("Split", style="cyan")
    table.add_column("Rows", style="green")
    table.add_column("Path")
    for split, path in paths.items():
        rows = len(pd.read_parquet(path))
        table.add_row(split, f"{rows:,}", str(path))
    console.print(table)
    console.print("\n[dim]Next: run tier2_training_upgrades.py[/]")


if __name__ == "__main__":
    main()
