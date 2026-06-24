"""
TIER 2 — Training infrastructure upgrades
Run: python tier2_training_upgrades.py

What this does:
  - Adds LoRA (trains only 1% of parameters — much faster)
  - Mixed precision fp16 training (2x memory reduction)
  - Encoder layer freezing strategy
  - Label smoothing loss
  - Cosine LR schedule with warmup
  - Saves upgraded model to MLflow
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


# ── LoRA implementation (no peft dependency needed) ───────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation — adds two small matrices A and B to a frozen linear layer.
    
    Instead of updating W (d×k), we learn A (d×r) and B (r×k) where r << d.
    This reduces trainable params by ~100x.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in  = linear.in_features
        d_out = linear.out_features

        # Freeze original weights
        for p in linear.parameters():
            p.requires_grad = False

        # LoRA matrices — A initialized with kaiming, B with zeros
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def apply_lora(model: nn.Module, rank: int = 8, target_modules: list[str] | None = None) -> nn.Module:
    """Replace target linear layers with LoRA-wrapped versions."""
    target_modules = target_modules or ["query", "value"]
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name, attr = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, attr, LoRALinear(module, rank=rank))
                replaced += 1
    console.print(f"  LoRA applied to [green]{replaced}[/] linear layers (rank={rank})")
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Label smoothing loss ──────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """Prevents overconfidence — distributes 'epsilon' probability to other classes."""

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        smooth_labels = torch.full_like(log_probs, self.epsilon / (self.num_classes - 1))
        smooth_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.epsilon)
        return (-smooth_labels * log_probs).sum(-1).mean()


# ── Upgraded trainer ──────────────────────────────────────────────────────────

class UpgradedTrainer:

    def __init__(self, config: dict):
        self.cfg = config
        self.tc = config["training"]
        self.hp = self.tc["hyperparams"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()   # fp16 only on GPU
        self.scaler = GradScaler(enabled=self.use_amp)
        console.print(f"  Device: [cyan]{self.device}[/] | AMP: [cyan]{self.use_amp}[/]")

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              run_name: str = "tier2_lora") -> tuple:
        from src.training.models import build_model
        from src.training.trainer import TextDataset
        from sklearn.metrics import f1_score, accuracy_score

        mlflow_cfg = self.tc.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db"))
        mlflow.set_experiment("tier2_lora_training")

        # Build base model
        model = build_model(
            self.tc["task"],
            self.tc["model_name"],
            self.tc["num_labels"]
        ).to(self.device)

        # Freeze bottom 8 encoder layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "encoder"):
            for layer in model.encoder.encoder.layer[:8]:
                for p in layer.parameters():
                    p.requires_grad = False
            console.print("  Froze bottom [cyan]8[/] encoder layers")

        # Apply LoRA to query and value projections
        model = apply_lora(model, rank=8, target_modules=["query", "value"])

        total, trainable = count_parameters(model)
        pct = 100 * trainable / total
        console.print(f"  Parameters: [green]{trainable:,}[/] trainable / {total:,} total ([green]{pct:.1f}%[/])")

        # Loss, optimizer, scheduler
        loss_fn = LabelSmoothingLoss(num_classes=self.tc["num_labels"], epsilon=0.1)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.hp["learning_rate"],
            weight_decay=self.hp["weight_decay"],
        )
        total_steps = len(train_loader) * self.hp["num_epochs"]
        warmup_steps = int(total_steps * self.hp.get("warmup_ratio", 0.1))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1, best_state, patience, pat_counter = 0.0, None, 3, 0

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "model": self.tc["model_name"],
                "lora_rank": 8,
                "frozen_layers": 8,
                "label_smoothing": 0.1,
                "scheduler": "cosine",
                "amp": self.use_amp,
                "trainable_pct": round(pct, 2),
            })

            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          BarColumn(), console=console) as progress:
                task = progress.add_task("Training", total=self.hp["num_epochs"])

                for epoch in range(1, self.hp["num_epochs"] + 1):
                    # ── Train ──
                    model.train()
                    train_loss = 0.0
                    for batch in train_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        labels = batch.pop("labels")
                        optimizer.zero_grad()
                        with autocast(enabled=self.use_amp):
                            out = model(**batch, labels=labels)
                            loss = loss_fn(out["logits"], labels)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        train_loss += loss.item()
                    train_loss /= len(train_loader)

                    # ── Eval ──
                    model.eval()
                    all_preds, all_labels, val_loss = [], [], 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            labels = batch.pop("labels")
                            with autocast(enabled=self.use_amp):
                                out = model(**batch, labels=labels)
                            val_loss += loss_fn(out["logits"], labels).item()
                            all_preds.extend(out["logits"].argmax(-1).cpu().tolist())
                            all_labels.extend(labels.cpu().tolist())
                    val_loss /= len(val_loader)
                    val_f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
                    val_acc = accuracy_score(all_labels, all_preds)

                    mlflow.log_metrics({
                        "train_loss": train_loss, "val_loss": val_loss,
                        "val_f1": val_f1, "val_acc": val_acc,
                    }, step=epoch)

                    progress.advance(task)
                    console.print(
                        f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} f1=[green]{val_f1:.4f}[/] acc={val_acc:.4f}"
                    )

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        pat_counter = 0
                    else:
                        pat_counter += 1
                        if pat_counter >= patience:
                            console.print(f"  [yellow]Early stopping at epoch {epoch}[/]")
                            break

            # Load best weights and save
            if best_state:
                model.load_state_dict(best_state)

            ckpt = Path("models/tier2_lora_best.pt")
            ckpt.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            mlflow.pytorch.log_model(model, artifact_path="model")
            console.print(f"\n  Best val_f1=[bold green]{best_f1:.4f}[/]")
            console.print(f"  Model saved → {ckpt}")

        return model, best_f1, run.info.run_id


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold cyan]Tier 2 — Training infrastructure upgrades[/]\n")

    feat_path = Path("data/features")
    if not (feat_path / "train.parquet").exists():
        console.print("[red]Run tier1_real_data.py first![/]")
        sys.exit(1)

    from src.utils.config import load_config
    from src.training.trainer import TextDataset

    cfg = load_config("configs/config.yaml")

    train_df = pd.read_parquet(feat_path / "train.parquet")
    val_df   = pd.read_parquet(feat_path / "val.parquet")

    model_name = cfg["training"]["model_name"]
    max_length = cfg["training"].get("max_length", 128)
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    text_col  = "text"  if "text"  in train_df.columns else train_df.columns[0]
    label_col = "label" if "label" in train_df.columns else train_df.columns[-1]

    # Cast labels to int
    train_df[label_col] = train_df[label_col].astype(int)
    val_df[label_col]   = val_df[label_col].astype(int)

    train_ds = TextDataset(train_df[text_col].tolist(), train_df[label_col].tolist(), tokenizer, max_length)
    val_ds   = TextDataset(val_df[text_col].tolist(),   val_df[label_col].tolist(),   tokenizer, max_length)

    bs = min(cfg["training"]["hyperparams"]["batch_size"], 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)

    trainer = UpgradedTrainer(cfg)
    model, best_f1, run_id = trainer.train(train_loader, val_loader)

    console.print(f"\n[bold green]Tier 2 complete![/]")
    console.print(f"  val_f1={best_f1:.4f} | MLflow run_id={run_id}")
    console.print("\n[dim]Next: run tier3_serving_upgrades.py[/]")


if __name__ == "__main__":
    main()
