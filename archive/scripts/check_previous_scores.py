import os
from pathlib import Path
import numpy as np

WORK_DIR = Path(r"F:\desktop\MyRouter\Nekoya\nekolab\[Kaggle]tabular-long-form-competition-ugcml-harbour-space").resolve()
os.chdir(WORK_DIR)

models = {
    'v3_xgb_main': 'oof_gbdt_v3/test_xgb_main.npy',
    'v1_xgb': 'oof_gbdt/test_xgb.npy',
    'v3_lgb_et': 'oof_gbdt_v3/test_lgb_et.npy',
    'v3_nn_tabular': 'oof_cat_nn_v3/test_nn_tabular.npy',
    'v3_lgb_res': 'oof_gbdt_v3/test_lgb_res.npy',
    'v2_lgb_res': 'oof_gbdt_v2/test_lgb_res.npy',
    'v1_lgb_et': 'oof_gbdt/test_lgb_et.npy',
    'v3_lgb_huber': 'oof_gbdt_v3/test_lgb_huber.npy',
    'v1_lgb_std': 'oof_gbdt/test_lgb_std.npy',
    'v3_xgb_deep': 'oof_gbdt_v3/test_xgb_deep.npy'
}

print("=== Checking Test Predictions Stats ===")
for name, p in models.items():
    arr = np.load(WORK_DIR / p)
    p_orig = np.expm1(arr)
    print(f"{name:<15s}: log_mean={arr.mean():.4f}, log_med={np.median(arr):.4f} | price_mean={p_orig.mean():.2f}, price_med={np.median(p_orig):.2f}")

