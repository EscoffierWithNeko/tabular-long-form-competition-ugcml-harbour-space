import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=== geo_cell analysis ===")
print(f"train geo_cell nunique: {train['geo_cell'].nunique()}, nulls: {train['geo_cell'].isnull().sum()}")
print(f"test geo_cell nunique: {test['geo_cell'].nunique()}, nulls: {test['geo_cell'].isnull().sum()}")
tr_cells = set(train['geo_cell'].dropna().unique())
te_cells = set(test['geo_cell'].dropna().unique())
overlap = len(tr_cells & te_cells)
print(f"geo_cell overlap: {overlap} / test unique ({len(te_cells)}): {overlap / len(te_cells):.2%}")
print("Sample geo_cells:", train['geo_cell'].head(5).tolist())

print("\n=== host_location analysis ===")
print(f"train host_location nunique: {train['host_location'].nunique()}")
print(f"test host_location nunique: {test['host_location'].nunique()}")
tr_locs = set(train['host_location'].dropna().unique())
te_locs = set(test['host_location'].dropna().unique())
print(f"host_location overlap: {len(tr_locs & te_locs)} / test unique ({len(te_locs)}): {len(tr_locs & te_locs) / len(te_locs):.2%}")
print("Top 5 host_location:\n", train['host_location'].value_counts().head(5))

print("\n=== Date columns analysis ===")
print("first_review sample:", train['first_review'].dropna().head(3).tolist())
print("last_review sample:", train['last_review'].dropna().head(3).tolist())

print("\n=== Target Price analysis ===")
print("price min/max/mean/median/std:")
print(train['price'].describe())
print(f"log1p(price) std: {np.log1p(train['price']).std():.5f}, skew: {np.log1p(train['price']).skew():.5f}")
