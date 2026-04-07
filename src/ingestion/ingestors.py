"""src/ingestion/ingestors.py — Data ingestion from multiple sources."""
from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import RawRecord


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseIngestor(ABC):
    """All ingestors produce RawRecord objects and land them in data/raw/."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.raw_path = Path(self.cfg["data"]["raw_path"])
        self.raw_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def ingest(self) -> Iterator[RawRecord]:
        """Yield RawRecord instances from the source."""

    def run(self) -> int:
        """Run ingestion and persist records as newline-delimited JSON."""
        count = 0
        out_file = self.raw_path / f"{self.__class__.__name__}_{int(time.time())}.jsonl"
        with out_file.open("w") as fh:
            for record in self.ingest():
                fh.write(record.model_dump_json() + "\n")
                count += 1
        logger.info(f"{self.__class__.__name__} wrote {count} records → {out_file}")
        return count


# ── Batch ETL ─────────────────────────────────────────────────────────────────

class BatchIngestor(BaseIngestor):
    """Reads Parquet/CSV files from a local or cloud path."""

    def ingest(self) -> Iterator[RawRecord]:
        source_path = Path(self.cfg["ingestion"]["batch"]["source_path"])
        files = list(source_path.glob("**/*.parquet")) + list(source_path.glob("**/*.csv"))
        logger.info(f"BatchIngestor found {len(files)} file(s) at {source_path}")

        for fpath in files:
            df = (
                pd.read_parquet(fpath)
                if fpath.suffix == ".parquet"
                else pd.read_csv(fpath)
            )
            for _, row in df.iterrows():
                yield RawRecord(
                    id=str(row.get("id", uuid.uuid4())),
                    source=str(fpath),
                    timestamp=datetime.now(timezone.utc),
                    payload=row.to_dict(),
                )


# ── Streaming (Kafka) ─────────────────────────────────────────────────────────

class KafkaIngestor(BaseIngestor):
    """Consumes messages from a Kafka topic and yields RawRecords."""

    def ingest(self) -> Iterator[RawRecord]:
        try:
            from kafka import KafkaConsumer  # type: ignore
        except ImportError:
            raise RuntimeError("Install kafka-python: pip install kafka-python")

        kafka_cfg = self.cfg["ingestion"]["kafka"]
        consumer = KafkaConsumer(
            kafka_cfg["topic"],
            bootstrap_servers=kafka_cfg["bootstrap_servers"],
            group_id=kafka_cfg["group_id"],
            auto_offset_reset=kafka_cfg.get("auto_offset_reset", "earliest"),
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=5000,
        )
        logger.info(f"KafkaIngestor consuming from topic '{kafka_cfg['topic']}'")
        for msg in consumer:
            yield RawRecord(
                id=str(uuid.uuid4()),
                source=f"kafka:{kafka_cfg['topic']}:{msg.partition}:{msg.offset}",
                timestamp=datetime.fromtimestamp(msg.timestamp / 1000, tz=timezone.utc),
                payload=msg.value,
            )
        consumer.close()


# ── REST API ──────────────────────────────────────────────────────────────────

class APIIngestor(BaseIngestor):
    """Pages through a REST API and yields RawRecords."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_page(self, session: requests.Session, url: str, params: dict) -> dict:
        resp = session.get(url, params=params, timeout=self.cfg["ingestion"]["api"]["timeout_seconds"])
        resp.raise_for_status()
        return resp.json()

    def ingest(self) -> Iterator[RawRecord]:
        api_cfg = self.cfg["ingestion"]["api"]
        base_url = api_cfg["base_url"]
        min_interval = 1.0 / api_cfg.get("rate_limit_rps", 10)

        with requests.Session() as session:
            page, has_more = 1, True
            while has_more:
                t0 = time.monotonic()
                data = self._fetch_page(session, base_url, {"page": page, "per_page": 100})
                items = data.get("items", data.get("results", [data]))
                for item in items:
                    yield RawRecord(
                        id=str(item.get("id", uuid.uuid4())),
                        source=base_url,
                        timestamp=datetime.now(timezone.utc),
                        payload=item,
                    )
                has_more = bool(data.get("next_page") or data.get("has_more"))
                page += 1
                elapsed = time.monotonic() - t0
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_ingestor(source: str, config: dict | None = None) -> BaseIngestor:
    mapping = {
        "batch": BatchIngestor,
        "kafka": KafkaIngestor,
        "api": APIIngestor,
    }
    cls = mapping.get(source)
    if cls is None:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(mapping)}")
    return cls(config=config)
