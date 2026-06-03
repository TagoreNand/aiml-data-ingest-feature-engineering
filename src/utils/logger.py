"""src/utils/logger.py — Structured JSON logging via loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from src.utils.config import get


def setup_logger() -> None:
    """Configure loguru based on config.yaml logging section."""
    _logger.remove()

    level: str = get("logging.level", "INFO")
    fmt: str = get("logging.format", "text")
    log_file: str = get("logging.output", "logs/app.log")
    rotation: str = get("logging.rotation", "100 MB")
    retention: str = get("logging.retention", "30 days")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        log_format = (
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSS}","level":"{level}",'
            '"name":"{name}","message":"{message}","extra":{extra}}}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        )

    _logger.add(sys.stderr, level=level, format=log_format, colorize=(fmt != "json"))
    _logger.add(
        log_file,
        level=level,
        format=log_format,
        rotation=rotation,
        retention=retention,
        enqueue=True,
    )


# Call once at import time so all modules share the same logger.
setup_logger()

logger = _logger
