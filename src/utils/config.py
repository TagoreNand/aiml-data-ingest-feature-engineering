"""src/utils/config.py — YAML config loader with env-var override support."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG: dict[str, Any] | None = None


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config, substituting ${ENV_VAR} placeholders with env values."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Copy configs/config.example.yaml to configs/config.yaml first."
        )

    raw = path.read_text()
    # Simple env-var substitution: ${MY_VAR} → os.environ["MY_VAR"]
    for key, val in os.environ.items():
        raw = raw.replace(f"${{{key}}}", val)

    _CONFIG = yaml.safe_load(raw)
    return _CONFIG


def get(path: str, default: Any = None, config: dict | None = None) -> Any:
    """Dot-path accessor, e.g. get('training.hyperparams.learning_rate')."""
    cfg = config or load_config()
    keys = path.split(".")
    for k in keys:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(k, default)
        if cfg is default:
            return default
    return cfg
