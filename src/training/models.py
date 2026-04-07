"""src/training/models.py — Model definitions for classification, regression, and generation."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    PreTrainedModel,
)

from src.utils.logger import logger


# ── Text Classification ───────────────────────────────────────────────────────

class TextClassifier(nn.Module):
    """Transformer encoder + classification head with optional freeze."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

    def freeze_encoder(self, n_layers: int = 0) -> None:
        """Freeze the embeddings and optionally the first n_layers encoder layers."""
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in self.encoder.encoder.layer[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info(f"Froze embeddings + first {n_layers} encoder layers.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(self.dropout(pooled))

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss() if self.num_labels > 1 else nn.BCEWithLogitsLoss()
            result["loss"] = loss_fn(logits, labels)
        return result


# ── Regression ────────────────────────────────────────────────────────────────

class TextRegressor(nn.Module):
    """Transformer encoder + regression head."""

    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        preds = self.head(pooled).squeeze(-1)
        result: dict[str, torch.Tensor] = {"predictions": preds}
        if labels is not None:
            result["loss"] = nn.MSELoss()(preds, labels.float())
        return result


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(task: str, model_name: str, num_labels: int = 2) -> nn.Module:
    """Return the appropriate model for a given task."""
    if task == "text_classification":
        return TextClassifier(model_name, num_labels=num_labels)
    elif task == "regression":
        return TextRegressor(model_name)
    elif task == "generation":
        return AutoModelForCausalLM.from_pretrained(model_name)
    else:
        # Fallback to HF AutoModel for sequence classification
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
