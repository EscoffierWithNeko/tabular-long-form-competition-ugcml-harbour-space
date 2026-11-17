import sys
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")

pkgs = ['lightgbm', 'xgboost', 'catboost', 'sklearn', 'scipy', 'optuna']
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"{p}: {ver}")
    except ImportError as e:
        print(f"{p}: NOT INSTALLED ({e})")
