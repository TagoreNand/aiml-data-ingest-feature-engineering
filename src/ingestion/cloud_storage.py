"""src/ingestion/cloud_storage.py — S3/GCS cloud storage integration."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from src.utils.logger import logger


class S3Storage:
    """
    S3-compatible cloud storage client.
    Works with AWS S3, MinIO, and any S3-compatible store.
    
    Required env vars:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    Or use IAM roles in production (no keys needed).
    """

    def __init__(self, bucket: str | None = None, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                )
                logger.info("S3 client initialised.")
            except ImportError:
                raise RuntimeError("Install boto3: pip install boto3")
        return self._client

    def parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse s3://bucket/prefix/key → (bucket, key)."""
        path = s3_path.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def list_files(self, s3_path: str,
                   extensions: list[str] | None = None) -> list[str]:
        """List all files under an S3 prefix."""
        extensions = extensions or [".parquet", ".csv", ".jsonl"]
        bucket, prefix = self.parse_s3_path(s3_path)
        paginator = self.client.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if any(key.endswith(ext) for ext in extensions):
                    files.append(f"s3://{bucket}/{key}")
        logger.info(f"Found {len(files)} files in {s3_path}")
        return files

    def read_dataframe(self, s3_path: str) -> pd.DataFrame:
        """Read a Parquet or CSV file directly from S3 into a DataFrame."""
        bucket, key = self.parse_s3_path(s3_path)
        logger.info(f"Reading s3://{bucket}/{key}")
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        if key.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body))
        elif key.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body))
        elif key.endswith(".jsonl"):
            return pd.read_json(io.BytesIO(body), lines=True)
        else:
            raise ValueError(f"Unsupported file type: {key}")

    def upload_file(self, local_path: str | Path,
                    s3_path: str) -> str:
        """Upload a local file to S3."""
        bucket, key = self.parse_s3_path(s3_path)
        self.client.upload_file(str(local_path), bucket, key)
        logger.info(f"Uploaded {local_path} → s3://{bucket}/{key}")
        return s3_path

    def upload_dataframe(self, df: pd.DataFrame,
                         s3_path: str,
                         format: str = "parquet") -> str:
        """Upload a DataFrame directly to S3."""
        bucket, key = self.parse_s3_path(s3_path)
        buf = io.BytesIO()
        if format == "parquet":
            df.to_parquet(buf, index=False)
        else:
            df.to_csv(buf, index=False)
        buf.seek(0)
        self.client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        logger.info(f"Uploaded DataFrame ({len(df)} rows) → s3://{bucket}/{key}")
        return s3_path

    def download_file(self, s3_path: str,
                      local_path: str | Path) -> Path:
        """Download a file from S3 to local disk."""
        bucket, key = self.parse_s3_path(s3_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded s3://{bucket}/{key} → {local_path}")
        return local_path

    def sync_to_local(self, s3_path: str,
                      local_dir: str | Path,
                      extensions: list[str] | None = None) -> list[Path]:
        """Download all matching files from S3 to a local directory."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        files = self.list_files(s3_path, extensions)
        local_files = []
        for s3_file in files:
            _, key = self.parse_s3_path(s3_file)
            filename = Path(key).name
            local_path = self.download_file(s3_file, local_dir / filename)
            local_files.append(local_path)
        return local_files

    def model_exists(self, s3_path: str) -> bool:
        """Check if a model artifact exists in S3."""
        try:
            bucket, key = self.parse_s3_path(s3_path)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False


class GCSStorage:
    """
    Google Cloud Storage client.
    
    Required: GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON.
    Or use Workload Identity in GKE (no keys needed).
    """

    def __init__(self, bucket: str | None = None):
        self.bucket_name = bucket
        self._client = None
        self._bucket = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
                logger.info("GCS client initialised.")
            except ImportError:
                raise RuntimeError(
                    "Install google-cloud-storage: pip install google-cloud-storage"
                )
        return self._client

    def parse_gcs_path(self, gcs_path: str) -> tuple[str, str]:
        """Parse gs://bucket/prefix → (bucket, prefix)."""
        path = gcs_path.replace("gs://", "")
        parts = path.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def read_dataframe(self, gcs_path: str) -> pd.DataFrame:
        """Read a file from GCS into a DataFrame."""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        if blob_name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        return pd.read_csv(io.BytesIO(data))

    def upload_dataframe(self, df: pd.DataFrame,
                         gcs_path: str,
                         format: str = "parquet") -> str:
        """Upload a DataFrame to GCS."""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        buf = io.BytesIO()
        if format == "parquet":
            df.to_parquet(buf, index=False)
        else:
            df.to_csv(buf, index=False)
        buf.seek(0)
        blob.upload_from_file(buf)
        logger.info(f"Uploaded DataFrame → {gcs_path}")
        return gcs_path


class CloudArtifactStore:
    """
    Unified artifact store — saves/loads models, features,
    and reports to/from cloud storage.
    
    In production this replaces local file paths everywhere.
    """

    def __init__(self, config: dict):
        storage_cfg = config.get("storage", {})
        self.backend = storage_cfg.get("backend", "local")
        self.bucket = storage_cfg.get("bucket", "")
        self.prefix = storage_cfg.get("prefix", "aiml-platform")
        self._s3 = None
        self._gcs = None

    @property
    def s3(self) -> S3Storage:
        if self._s3 is None:
            self._s3 = S3Storage(bucket=self.bucket)
        return self._s3

    @property
    def gcs(self) -> GCSStorage:
        if self._gcs is None:
            self._gcs = GCSStorage(bucket=self.bucket)
        return self._gcs

    def save_features(self, df: pd.DataFrame,
                      split: str,
                      version: str = "latest") -> str:
        """Save feature split to cloud or local."""
        if self.backend == "s3":
            path = f"s3://{self.bucket}/{self.prefix}/features/{version}/{split}.parquet"
            return self.s3.upload_dataframe(df, path)
        elif self.backend == "gcs":
            path = f"gs://{self.bucket}/{self.prefix}/features/{version}/{split}.parquet"
            return self.gcs.upload_dataframe(df, path)
        else:
            local = Path(f"data/features/{split}.parquet")
            local.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(local, index=False)
            return str(local)

    def load_features(self, split: str,
                      version: str = "latest") -> pd.DataFrame:
        """Load feature split from cloud or local."""
        if self.backend == "s3":
            path = f"s3://{self.bucket}/{self.prefix}/features/{version}/{split}.parquet"
            return self.s3.read_dataframe(path)
        elif self.backend == "gcs":
            path = f"gs://{self.bucket}/{self.prefix}/features/{version}/{split}.parquet"
            return self.gcs.read_dataframe(path)
        else:
            return pd.read_parquet(f"data/features/{split}.parquet")

    def save_drift_report(self, report_json: str,
                          timestamp: str) -> str:
        """Save drift report to cloud."""
        filename = f"drift_report_{timestamp}.json"
        if self.backend == "s3":
            bucket_name, _ = self.s3.parse_s3_path(f"s3://{self.bucket}/")
            key = f"{self.prefix}/reports/{filename}"
            self.s3.client.put_object(
                Bucket=bucket_name, Key=key, Body=report_json
            )
            return f"s3://{self.bucket}/{key}"
        else:
            path = Path(f"logs/{filename}")
            path.write_text(report_json)
            return str(path)
