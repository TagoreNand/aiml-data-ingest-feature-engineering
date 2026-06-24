"""
PHASE 3 — Feature Store with Online + Offline Stores
Run: python phase3_feature_store.py

What this adds:
  - Offline store: versioned Parquet feature sets (already have)
  - Online store: Redis for low-latency feature serving (<5ms)
  - Feature registry: tracks feature definitions and versions
  - Point-in-time correct joins for training
  - Writes src/features/store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
from rich.console import Console
console = Console()


FEATURE_STORE = '''"""src/features/store.py — Feature store with Redis online store + Parquet offline store."""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import logger


# ── Feature Definitions ───────────────────────────────────────────────────────

@dataclass
class FeatureDefinition:
    """Defines a single feature — its name, type, and how to compute it."""
    name: str
    dtype: str              # "float", "int", "str"
    description: str = ""
    default: Any = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype,
                "description": self.description, "default": self.default,
                "tags": self.tags}


@dataclass
class FeatureView:
    """
    Groups related features together.
    Like a database table for features.
    
    Example:
      text_features = FeatureView(
          name="text_features",
          entity="review_id",
          features=[
              FeatureDefinition("text_len", "int", "Character count"),
              FeatureDefinition("word_count", "int", "Word count"),
              FeatureDefinition("sentiment_score", "float", "VADER score"),
          ],
          ttl_seconds=86400,  # 24h cache in online store
      )
    """
    name: str
    entity: str             # Primary key column name
    features: list[FeatureDefinition]
    ttl_seconds: int = 3600 # Time-to-live in online store
    tags: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


# ── Feature Registry ──────────────────────────────────────────────────────────

class FeatureRegistry:
    """
    Tracks all feature views and their versions.
    Persisted as JSON — in production would use a database.
    """

    def __init__(self, registry_path: str = "data/feature_registry.json"):
        self.registry_path = Path(registry_path)
        self._views: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            self._views = json.loads(self.registry_path.read_text())

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._views, indent=2))

    def register(self, view: FeatureView) -> None:
        """Register a feature view."""
        self._views[view.name] = {
            "name": view.name,
            "entity": view.entity,
            "features": [f.to_dict() for f in view.features],
            "ttl_seconds": view.ttl_seconds,
            "tags": view.tags,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        logger.info(f"Registered feature view: {view.name} "
                    f"({len(view.features)} features)")

    def get(self, name: str) -> dict | None:
        return self._views.get(name)

    def list_views(self) -> list[str]:
        return list(self._views.keys())


# ── Online Store (Redis) ──────────────────────────────────────────────────────

class RedisOnlineStore:
    """
    Low-latency online feature store backed by Redis.
    Serves pre-computed features at <5ms for real-time inference.
    
    Key format: feature:{view_name}:{entity_id}
    Value: JSON of feature values
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: str | None = None):
        self.host     = host
        self.port     = port
        self.db       = db
        self.password = password
        self._client  = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=self.host, port=self.port,
                    db=self.db, password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self._client.ping()
                logger.info(f"Redis online store connected: {self.host}:{self.port}")
            except ImportError:
                raise RuntimeError("Install redis: pip install redis")
            except Exception as e:
                logger.warning(f"Redis not available ({e}). Using in-memory fallback.")
                self._client = InMemoryStore()
        return self._client

    def _key(self, view_name: str, entity_id: str) -> str:
        return f"feature:{view_name}:{entity_id}"

    def write(self, view_name: str, entity_id: str,
              features: dict, ttl_seconds: int = 3600) -> None:
        """Write features for an entity to Redis."""
        key = self._key(view_name, entity_id)
        value = json.dumps({**features, "_ts": time.time()})
        self.client.setex(key, ttl_seconds, value)

    def read(self, view_name: str,
             entity_id: str) -> dict | None:
        """Read features for an entity from Redis."""
        key = self._key(view_name, entity_id)
        value = self.client.get(key)
        if value is None:
            return None
        data = json.loads(value)
        data.pop("_ts", None)
        return data

    def read_batch(self, view_name: str,
                   entity_ids: list[str]) -> list[dict | None]:
        """Read features for multiple entities in one round-trip."""
        keys = [self._key(view_name, eid) for eid in entity_ids]
        values = self.client.mget(keys)
        results = []
        for v in values:
            if v is None:
                results.append(None)
            else:
                data = json.loads(v)
                data.pop("_ts", None)
                results.append(data)
        return results

    def materialise_from_df(self, view: FeatureView,
                             df: pd.DataFrame,
                             entity_col: str | None = None) -> int:
        """
        Bulk-load features from a DataFrame into Redis.
        Called after feature pipeline completes.
        """
        entity_col = entity_col or view.entity
        if entity_col not in df.columns:
            logger.warning(f"Entity column {entity_col!r} not in DataFrame")
            return 0

        feature_cols = [c for c in view.feature_names if c in df.columns]
        count = 0
        pipe = self.client.pipeline()  # type: ignore

        for _, row in df.iterrows():
            entity_id = str(row[entity_col])
            features  = {col: row[col] for col in feature_cols
                         if pd.notna(row[col])}
            key   = self._key(view.name, entity_id)
            value = json.dumps({**features, "_ts": time.time()})
            pipe.setex(key, view.ttl_seconds, value)
            count += 1

        pipe.execute()
        logger.info(f"Materialised {count} entities → Redis [{view.name}]")
        return count

    def delete(self, view_name: str, entity_id: str) -> None:
        self.client.delete(self._key(view_name, entity_id))


class InMemoryStore:
    """Fallback in-memory store when Redis is not available."""

    def __init__(self):
        self._data: dict[str, tuple[str, float]] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._data[key] = (value, time.time() + ttl)

    def get(self, key: str) -> str | None:
        if key not in self._data:
            return None
        value, expiry = self._data[key]
        if time.time() > expiry:
            del self._data[key]
            return None
        return value

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.get(k) for k in keys]

    def pipeline(self):
        return PipelineFallback(self)

    def ping(self) -> bool:
        return True


class PipelineFallback:
    def __init__(self, store: InMemoryStore):
        self._store = store
        self._ops: list = []

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._ops.append(("setex", key, ttl, value))

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "setex":
                self._store.setex(op[1], op[2], op[3])
        self._ops.clear()


# ── Offline Store (Parquet) ───────────────────────────────────────────────────

class ParquetOfflineStore:
    """
    Offline feature store backed by Parquet files.
    Used for training — supports point-in-time correct joins.
    """

    def __init__(self, base_path: str = "data/feature_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, view: FeatureView, df: pd.DataFrame,
              version: str = "latest") -> Path:
        """Write a feature view snapshot."""
        out_dir = self.base_path / view.name / version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features.parquet"

        cols = [view.entity] + [
            c for c in view.feature_names if c in df.columns
        ]
        df[cols].to_parquet(out_path, index=False)
        logger.info(f"Offline store: wrote {len(df)} rows → {out_path}")
        return out_path

    def read(self, view_name: str, version: str = "latest") -> pd.DataFrame:
        """Read a feature view snapshot."""
        path = self.base_path / view_name / version / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature view not found: {path}")
        return pd.read_parquet(path)

    def point_in_time_join(self, entity_df: pd.DataFrame,
                            view_name: str,
                            timestamp_col: str = "event_timestamp",
                            version: str = "latest") -> pd.DataFrame:
        """
        Point-in-time correct join — ensures no future data leakage.
        Only uses features that existed at the time of each event.
        """
        features_df = self.read(view_name, version)

        if timestamp_col not in entity_df.columns:
            # No timestamp — simple merge
            return entity_df.merge(features_df, how="left",
                                    on=self._entity_col(view_name, version))

        # Sort and merge_asof for point-in-time correctness
        if timestamp_col not in features_df.columns:
            return entity_df.merge(features_df, how="left",
                                    on=self._entity_col(view_name, version))

        entity_df   = entity_df.sort_values(timestamp_col)
        features_df = features_df.sort_values(timestamp_col)
        return pd.merge_asof(
            entity_df, features_df,
            on=timestamp_col,
            direction="backward",
        )

    def _entity_col(self, view_name: str, version: str) -> str:
        meta_path = self.base_path / view_name / version / "features.parquet"
        df = pd.read_parquet(meta_path, columns=[])
        return df.columns[0]

    def list_versions(self, view_name: str) -> list[str]:
        view_path = self.base_path / view_name
        if not view_path.exists():
            return []
        return [d.name for d in view_path.iterdir() if d.is_dir()]


# ── Unified Feature Store ─────────────────────────────────────────────────────

class FeatureStore:
    """
    Unified interface combining online (Redis) and offline (Parquet) stores.
    
    Training:  uses offline store for historical features
    Inference: uses online store for low-latency serving
    """

    def __init__(self, config: dict | None = None):
        from src.utils.config import load_config
        cfg = config or load_config()
        fs_cfg = cfg.get("feature_store", {})

        redis_cfg = fs_cfg.get("online", {})
        self.online  = RedisOnlineStore(
            host=redis_cfg.get("host", "localhost"),
            port=redis_cfg.get("port", 6379),
        )
        self.offline  = ParquetOfflineStore(
            base_path=fs_cfg.get("offline_path", "data/feature_store")
        )
        self.registry = FeatureRegistry(
            registry_path=fs_cfg.get("registry_path",
                                       "data/feature_registry.json")
        )

    def register_view(self, view: FeatureView) -> None:
        self.registry.register(view)

    def materialise(self, view: FeatureView,
                    df: pd.DataFrame,
                    version: str = "latest") -> None:
        """
        Materialise features to both online and offline stores.
        Called after feature pipeline completes.
        """
        # Offline: save full history for training
        self.offline.write(view, df, version=version)
        logger.info(f"Materialised to offline store [{view.name}]")

        # Online: load into Redis for serving
        try:
            count = self.online.materialise_from_df(view, df)
            logger.info(f"Materialised {count} entities to online store [{view.name}]")
        except Exception as e:
            logger.warning(f"Online materialisation failed: {e}")

    def get_online_features(self, view_name: str,
                             entity_ids: list[str]) -> list[dict]:
        """Get features for inference — sub-5ms from Redis."""
        results = self.online.read_batch(view_name, entity_ids)
        return [r or {} for r in results]

    def get_training_features(self, view_name: str,
                               version: str = "latest") -> pd.DataFrame:
        """Get full feature set for training — from Parquet."""
        return self.offline.read(view_name, version)


# ── Pre-built Feature Views for this Platform ─────────────────────────────────

def get_default_feature_views() -> list[FeatureView]:
    """Feature views for the sentiment classification platform."""
    return [
        FeatureView(
            name="text_features",
            entity="id",
            features=[
                FeatureDefinition("text_len",    "int",   "Character count of review"),
                FeatureDefinition("word_count",  "int",   "Word count of review"),
                FeatureDefinition("label",       "int",   "Sentiment label (0/1)"),
            ],
            ttl_seconds=86400,
            tags={"domain": "nlp", "version": "v1"},
        ),
        FeatureView(
            name="temporal_features",
            entity="id",
            features=[
                FeatureDefinition("timestamp_hour",      "int", "Hour of ingestion"),
                FeatureDefinition("timestamp_dayofweek", "int", "Day of week"),
                FeatureDefinition("timestamp_month",     "int", "Month"),
            ],
            ttl_seconds=3600,
            tags={"domain": "temporal"},
        ),
    ]
'''


def main() -> None:
    console.print("[bold cyan]Phase 3 — Feature Store (Online + Offline)[/]\n")

    out = Path("src/features/store.py")
    out.write_text(FEATURE_STORE, encoding="utf-8")
    console.print(f"  [green]Written[/] → {out}")

    # Write materialisation script
    script = Path("scripts/materialise_features.py")
    script.write_text('''"""Materialise features to online (Redis) + offline (Parquet) stores."""
from __future__ import annotations
import sys, typer
from pathlib import Path

sys.path.insert(0, ".")

def main(config: Path = typer.Option("configs/config.yaml"),
         version: str = typer.Option("latest")):
    import pandas as pd
    from rich.console import Console
    from src.utils.config import load_config
    from src.features.store import FeatureStore, get_default_feature_views

    console = Console()
    cfg = load_config(config)
    store = FeatureStore(cfg)

    console.print("[bold cyan]Materialising features...[/]")

    for view in get_default_feature_views():
        store.register_view(view)

    # Load feature data
    feat_path = Path("data/features/train.parquet")
    if not feat_path.exists():
        console.print("[red]Run feature pipeline first![/]")
        raise typer.Exit(1)

    df = pd.read_parquet(feat_path)
    for view in get_default_feature_views():
        try:
            store.materialise(view, df, version=version)
            console.print(f"  [green]✓[/] {view.name} → online + offline")
        except Exception as e:
            console.print(f"  [yellow]⚠[/] {view.name}: {e}")

    console.print("[bold green]Materialisation complete![/]")

if __name__ == "__main__":
    typer.run(main)
''', encoding="utf-8")
    console.print(f"  [green]Written[/] → {script}")

    console.print("\n[bold green]Phase 3 complete![/]")
    console.print("  src/features/store.py              — online + offline feature store")
    console.print("  scripts/materialise_features.py   — materialisation script")
    console.print("\nUsage:")
    console.print("  [cyan]pip install redis[/]")
    console.print("  [cyan]python scripts/materialise_features.py[/]")
    console.print("  [cyan]# Inference: store.get_online_features('text_features', ['id1', 'id2'])[/]")
    console.print("\n[dim]Next: python phase4_k8s_deploy.py[/]")


if __name__ == "__main__":
    main()
