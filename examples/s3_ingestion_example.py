"""Example: Ingest data from S3."""
import os, sys
sys.path.insert(0, ".")

# Set credentials (use IAM roles in production)
os.environ["AWS_ACCESS_KEY_ID"]     = "your-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"

from src.utils.config import load_config
from src.ingestion.cloud_storage import S3Storage, CloudArtifactStore

# Direct S3 operations
s3 = S3Storage()

# List files in a bucket
files = s3.list_files("s3://my-bucket/raw/")
print(f"Found {len(files)} files")

# Read a DataFrame from S3
# df = s3.read_dataframe("s3://my-bucket/raw/data.parquet")

# Upload features to S3
cfg = load_config()
cfg["storage"] = {"backend": "s3", "bucket": "my-bucket", "prefix": "aiml-platform"}
store = CloudArtifactStore(cfg)

# In production: store.save_features(train_df, "train", version="v1")
# In production: train_df = store.load_features("train", version="v1")
print("Cloud artifact store ready!")
