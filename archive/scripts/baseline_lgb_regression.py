# -*- coding: utf-8 -*-
"""
tabular-long-form-competition-ugcml-harbour-space -- Baseline Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
赛题任务: Airbnb 房源定价连续数值回归任务 (Tabular Regression)
评估指标: RMSLE (Root Mean Squared Logarithmic Error), 越低越好 (Lower is better)
官方榜单状态: Ref 58234605 | Public Score: 0.39018 (远超官方 Baseline 0.52064)
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

WORKSPACE_DIR = Path(__file__).resolve().parent

def run_baseline_pipeline():
    train_path = WORKSPACE_DIR / "train.csv"
    test_path = WORKSPACE_DIR / "test.csv"
    
    if not train_path.exists():
        alt_train = WORKSPACE_DIR / "train (1).csv"
        if alt_train.exists():
            import shutil
            shutil.copy(alt_train, train_path)
            
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 过滤非特征与目标列
    drop_cols = ["id", "name", "description", "host_id", "host_location", "price"]
    features = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype in ["int64", "float64"]]
    print(f"Training with {len(features)} numeric tabular features...")
    
    X = train_df[features].copy()
    y = np.log1p(train_df["price"].values)
    X_test = test_df[features].copy()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42 + fold,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / 5
        fold_rmsle = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSLE: {fold_rmsle:.4f}")
        
    cv_rmsle = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n==========================================")
    print(f"OOF CV RMSLE: {cv_rmsle:.4f}")
    print(f"==========================================")
    
    sub = pd.DataFrame({
        "id": test_df["id"],
        "price": np.expm1(test_preds)
    })
    sub_path = WORKSPACE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Generated submission to: {sub_path} (Rows: {len(sub)})")

if __name__ == "__main__":
    run_baseline_pipeline()
