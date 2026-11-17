import sys
import importlib

for pkg in ['lightgbm', 'xgboost', 'catboost', 'sklearn', 'scipy', 'optuna']:
    try:
        mod = importlib.import_module(pkg)
        print(f'{pkg}: available, version {getattr(mod, "__version__", "unknown")}')
    except ImportError:
        print(f'{pkg}: NOT available')

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f'Train shape: {train.shape}, Test shape: {test.shape}')
print('Price min, median, mean, max:', train['price'].min(), train['price'].median(), train['price'].mean(), train['price'].max())
print('Price quantiles (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):')
print(np.quantile(train['price'], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
print('Zero or negative prices:', (train['price'] <= 0).sum())
