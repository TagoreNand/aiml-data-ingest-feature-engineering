"""src/features/pipeline.py — Feature engineering, validation, and store materialisation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import FeatureRow


# ── Transformations ───────────────────────────────────────────────────────────

class FeatureTransformer:
    """Stateful transformer that fits on training data and transforms any split."""

    def __init__(self) -> None:
        self._scalers: dict[str, StandardScaler | MinMaxScaler] = {}
        self._encoders: dict[str, LabelEncoder] = {}
        self._custom: list[tuple[str, Callable]] = []

    def add_scaler(self, col: str, method: str = "standard") -> "FeatureTransformer":
        self._scalers[col] = StandardScaler() if method == "standard" else MinMaxScaler()
        return self

    def add_encoder(self, col: str) -> "FeatureTransformer":
        self._encoders[col] = LabelEncoder()
        return self

    def add_custom(self, name: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "FeatureTransformer":
        self._custom.append((name, fn))
        return self

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        for col, scaler in self._scalers.items():
            if col in df.columns:
                scaler.fit(df[[col]])
        for col, enc in self._encoders.items():
            if col in df.columns:
                enc.fit(df[col].astype(str))
        logger.info("FeatureTransformer fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, scaler in self._scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        for col, enc in self._encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col].astype(str))
        for name, fn in self._custom:
            df = fn(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ── Feature Pipeline ──────────────────────────────────────────────────────────

class FeaturePipeline:
    """Orchestrates raw → processed → feature materialisation."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.processed_path = Path(self.cfg["data"]["processed_path"])
        self.features_path = Path(self.cfg["data"]["features_path"])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.transformer = FeatureTransformer()

    # ── Step 1: load & validate ───────────────────────────────────────────────

    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        raw_path = Path(path or self.cfg["data"]["raw_path"])
        files = list(raw_path.glob("**/*.jsonl")) + list(raw_path.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No raw data found at {raw_path}")
        dfs = []
        for f in files:
            if f.suffix == ".jsonl":
                dfs.append(pd.read_json(f, lines=True))
            else:
                dfs.append(pd.read_parquet(f))
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df):,} rows from {len(files)} file(s).")
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, handle nulls, enforce dtypes."""
        n_before = len(df)
        df = df.drop_duplicates()
        df = df.dropna(how="all")
        logger.info(f"Validation: {n_before - len(df):,} rows removed. {len(df):,} remain.")
        return df

    # ── Step 2: engineer features ─────────────────────────────────────────────

    @staticmethod
    def _add_text_length(df: pd.DataFrame) -> pd.DataFrame:
        if "text" in df.columns:
            df["text_len"] = df["text"].fillna("").str.len()
            df["word_count"] = df["text"].fillna("").str.split().str.len()
        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
            df[f"{col}_hour"] = pd.to_datetime(df[col]).dt.hour
            df[f"{col}_dayofweek"] = pd.to_datetime(df[col]).dt.dayofweek
            df[f"{col}_month"] = pd.to_datetime(df[col]).dt.month
        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_text_length(df)
        df = self._add_time_features(df)
        self.transformer.add_custom("text_features", lambda x: x)
        logger.info(f"Feature engineering complete. Columns: {list(df.columns)}")
        return df

    # ── Step 3: split & materialise ───────────────────────────────────────────

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seed = self.cfg["data"]["random_seed"]
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        train_end = int(n * self.cfg["data"]["train_split"])
        val_end = train_end + int(n * self.cfg["data"]["val_split"])
        return df[:train_end], df[train_end:val_end], df[val_end:]

    def materialise(self, df: pd.DataFrame, split: str = "train") -> Path:
        out = self.features_path / f"{split}.parquet"
        # Flatten payload dict if present
        if "payload" in df.columns:
            payload_df = pd.json_normalize(df["payload"].tolist())
            df = pd.concat([df.drop(columns=["payload"]), payload_df], axis=1)
        df.to_parquet(out, index=False)
        logger.info(f"Materialised {len(df):,} rows → {out}")
        return out

    def to_feature_rows(self, df: pd.DataFrame, entity_col: str = "id") -> list[FeatureRow]:
        rows = []
        for _, r in df.iterrows():
            entity = str(r.get(entity_col, "unknown"))
            vals = {k: v for k, v in r.items() if k != entity_col and isinstance(v, (int, float, str))}
            rows.append(FeatureRow(entity_id=entity, feature_values=vals, event_timestamp=datetime.now(timezone.utc)))
        return rows

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self) -> dict[str, Path]:
        df = self.load_raw()
        df = self.validate(df)
        df = self.engineer(df)
        train, val, test = self.split(df)
        train_transformed = self.transformer.fit_transform(train)
        val_transformed = self.transformer.transform(val)
        test_transformed = self.transformer.transform(test)
        return {
            "train": self.materialise(train_transformed, "train"),
            "val": self.materialise(val_transformed, "val"),
            "test": self.materialise(test_transformed, "test"),
        }
