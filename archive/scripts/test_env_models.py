import sys
print(f"Python: {sys.version}")

try:
    import lightgbm as lgb
    print(f"LightGBM: {lgb.__version__}")
except ImportError as e:
    print(f"LightGBM not installed: {e}")

try:
    import xgboost as xgb
    print(f"XGBoost: {xgb.__version__}")
except ImportError as e:
    print(f"XGBoost not installed: {e}")

try:
    import catboost as cb
    print(f"CatBoost: {cb.__version__}")
except ImportError as e:
    print(f"CatBoost not installed: {e}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"PyTorch not installed: {e}")

try:
    import sklearn
    print(f"Scikit-Learn: {sklearn.__version__}")
except ImportError as e:
    print(f"Scikit-Learn not installed: {e}")
