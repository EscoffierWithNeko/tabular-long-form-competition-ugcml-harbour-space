import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', nrows=5000)
test = pd.read_csv('test.csv', nrows=1000)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

print('\n=== COLUMNS AND TYPES ===')
for col in train.columns:
    n_missing = train[col].isna().sum()
    sample_val = str(train[col].dropna().iloc[0]) if n_missing < len(train) else 'ALL_NAN'
    if len(sample_val) > 40:
        sample_val = sample_val[:37] + '...'
    print(f'{col:35s} | {str(train[col].dtype):10s} | missing: {n_missing:5d} ({n_missing/len(train):.1%}) | sample: {sample_val}')

print('\n=== LATITUDE & LONGITUDE CHECK ===')
geo_cols = [c for c in train.columns if 'lat' in c.lower() or 'lon' in c.lower() or 'geo' in c.lower() or 'coord' in c.lower()]
print('Geo columns:', geo_cols)
for gc in geo_cols:
    print(train[gc].describe())

print('\n=== TEXT COLUMNS CHECK ===')
text_candidates = ['name', 'description', 'neighborhood_overview', 'host_about', 'amenities', 'host_location']
for tc in text_candidates:
    if tc in train.columns:
        print(f'{tc}: len min={train[tc].dropna().astype(str).str.len().min()}, max={train[tc].dropna().astype(str).str.len().max()}, mean={train[tc].dropna().astype(str).str.len().mean():.1f}')
