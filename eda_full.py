import sys
from pathlib import Path
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', nrows=2000)
print(f'Train total shape: (full train is large, preview has {train.shape[1]} cols)')
print('=== ALL 53 COLUMNS ===')
for idx, col in enumerate(train.columns):
    dtype = train[col].dtype
    nulls = train[col].isnull().sum()
    uniq = train[col].nunique()
    sample = str(train[col].dropna().iloc[0])[:30] if uniq > 0 else 'EMPTY'
    print(f'[{idx:02d}] {col:<40} | {str(dtype):<10} | nulls={nulls:<5} | uniq={uniq:<6} | {sample}')

print('\n=== Testing ML Libraries ===')
for lib in ['lightgbm', 'xgboost', 'catboost', 'sklearn', 'scipy']:
    try:
        __import__(lib)
        print(f'  [OK] {lib}')
    except Exception as e:
        print(f'  [FAIL] {lib}: {e}')

print('\n=== Dataset Total Row Counts ===')
train_len = sum(1 for _ in open('train.csv', encoding='utf-8', errors='ignore')) - 1
test_len = sum(1 for _ in open('test.csv', encoding='utf-8', errors='ignore')) - 1
print(f'Train total rows: {train_len}, Test total rows: {test_len}')
