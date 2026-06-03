"""src/training/reward_model.py — RLHF reward model trained on preference pairs."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from src.utils.logger import logger


class PreferencePairDataset(Dataset):
    """
    Dataset of (chosen, rejected) response pairs.
    
    Format: list of dicts with keys:
      'prompt':   the input prompt
      'chosen':   the preferred response (human-rated better)
      'rejected': the non-preferred response
    """
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        def enc(text):
            return self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )
        chosen_enc   = enc(pair["prompt"] + " " + pair["chosen"])
        rejected_enc = enc(pair["prompt"] + " " + pair["rejected"])
        return {
            "chosen_input_ids":      chosen_enc["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_input_ids":      rejected_enc["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
        }


class RewardModel(nn.Module):
    """
    Scalar reward predictor on top of a pretrained transformer.
    Trained with Bradley-Terry loss on preference pairs.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:, 0]   # [CLS]
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(chosen_rewards: torch.Tensor,
                        rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Preference loss: maximise p(chosen > rejected).
    log(sigmoid(r_chosen - r_rejected))
    """
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()


class RewardTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, pairs: list[dict], epochs: int = 3,
              lr: float = 1e-5, batch_size: int = 8) -> RewardModel:
        dataset = PreferencePairDataset(pairs, self.tokenizer)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = RewardModel(self.model_name).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                chosen_rewards = model(
                    batch["chosen_input_ids"], batch["chosen_attention_mask"]
                )
                rejected_rewards = model(
                    batch["rejected_input_ids"], batch["rejected_attention_mask"]
                )
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (chosen_rewards > rejected_rewards).sum().item()

            acc = correct / len(dataset)
            logger.info(f"Reward model epoch {epoch}/{epochs} | "
                        f"loss={total_loss/len(loader):.4f} | accuracy={acc:.4f}")

        return model

    def score(self, model: RewardModel, texts: list[str]) -> list[float]:
        """Score a list of texts — higher = more preferred by humans."""
        model.eval()
        enc = self.tokenizer(texts, truncation=True, padding=True,
                              max_length=256, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            rewards = model(**enc)
        return rewards.cpu().tolist()
