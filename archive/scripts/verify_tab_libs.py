import sys
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"LightGBM: {lgb.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"CatBoost: {cb.__version__}")

# Test basic fitting of XGBoost and CatBoost
X = np.random.randn(100, 5)
y = np.random.randn(100)

lgb_model = lgb.LGBMRegressor(n_estimators=10, verbosity=-1)
lgb_model.fit(X, y)
print("LGBM fit successfully!")

xgb_model = xgb.XGBRegressor(n_estimators=10, verbosity=0)
xgb_model.fit(X, y)
print("XGBoost fit successfully!")

cb_model = cb.CatBoostRegressor(iterations=10, verbose=0)
cb_model.fit(X, y)
print("CatBoost fit successfully!")
print("All tabular models work perfectly!")
