import sys
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import sklearn

print(f'Python: {sys.version}')
print(f'LightGBM: {lgb.__version__}')
print(f'XGBoost: {xgb.__version__}')
print(f'CatBoost: {cb.__version__}')
print(f'Scikit-Learn: {sklearn.__version__}')
print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
