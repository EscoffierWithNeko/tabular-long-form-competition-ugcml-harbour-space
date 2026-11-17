import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)

print(f"Python: {sys.version}")

try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
except ImportError as e:
    print(f"LightGBM not found: {e}")

try:
    import catboost as cb
    print(f"CatBoost version: {cb.__version__}")
except ImportError as e:
    print(f"CatBoost not found: {e}")

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"XGBoost not found: {e}")

# Quick benchmark on synthetic dataset to test training time
X = np.random.randn(2000, 30).astype(np.float32)
y = np.random.randn(2000).astype(np.float32)

print("\n--- Benchmark LightGBM ---")
t0 = time.time()
lgb_mod = lgb.LGBMRegressor(n_estimators=100, n_jobs=4, verbose=-1, random_state=42)
lgb_mod.fit(X, y)
print(f"LightGBM 100 trees fit time: {time.time() - t0:.3f}s")

try:
    print("\n--- Benchmark CatBoost ---")
    t0 = time.time()
    cb_mod = cb.CatBoostRegressor(iterations=100, thread_count=4, verbose=0, random_seed=42)
    cb_mod.fit(X, y)
    print(f"CatBoost 100 trees fit time: {time.time() - t0:.3f}s")
except Exception as e:
    print(f"CatBoost benchmark failed: {e}")

try:
    print("\n--- Benchmark XGBoost ---")
    t0 = time.time()
    xgb_mod = xgb.XGBRegressor(n_estimators=100, n_jobs=4, random_state=42)
    xgb_mod.fit(X, y)
    print(f"XGBoost 100 trees fit time: {time.time() - t0:.3f}s")
except Exception as e:
    print(f"XGBoost benchmark failed: {e}")
