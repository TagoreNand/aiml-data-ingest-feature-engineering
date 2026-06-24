import pandas as pd
df = pd.read_parquet('data/features/train.parquet')
df.to_parquet('data/processed/reference_dataset.parquet', index=False)
print(f'Reference dataset created with {len(df)} rows.')
