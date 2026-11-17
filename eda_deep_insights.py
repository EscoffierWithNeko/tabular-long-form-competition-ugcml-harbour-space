import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
train = pd.read_csv(WORKSPACE_DIR / 'train.csv')
test = pd.read_csv(WORKSPACE_DIR / 'test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train geo_cell unique: {train['geo_cell'].nunique()}, Test geo_cell unique: {test['geo_cell'].nunique()}")
common_cells = set(train['geo_cell']).intersection(set(test['geo_cell']))
print(f"Common geo_cells: {len(common_cells)} / {test['geo_cell'].nunique()} test cells present in train")

# Target price distribution
print("Price quantiles (train):")
print(train['price'].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print("Price <= 0 count:", (train['price'] <= 0).sum())

# Date columns check
print("\nFirst review sample:", train['first_review'].dropna().head(3).tolist())
print("Last review sample:", train['last_review'].dropna().head(3).tolist())
print("Missing first_review in train:", train['first_review'].isna().sum(), "test:", test['first_review'].isna().sum())

# Bathrooms text check
print("\nTop 10 bathrooms_text:")
print(train['bathrooms_text'].value_counts().head(10))
